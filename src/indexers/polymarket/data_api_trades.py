"""Indexer for the Polymarket Data API trade tape, fetched per market."""

import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.common.indexer import Indexer
from src.indexers.polymarket.client import FULL_HISTORY_START, PolymarketClient
from src.indexers.polymarket.models import DataApiTrade

DATA_DIR = Path("data/polymarket/data_api_trades")
MARKETS_DIR = Path("data/polymarket/markets")
CURSOR_FILE = Path("data/polymarket/.data_api_trades_cursor")
BATCH_SIZE = 10000
MARKETS_PER_REQUEST = 20  # condition IDs per `market` query parameter
MIN_WINDOW_SIZE = 1  # a window can't be split below one second


def _trade_key(trade: DataApiTrade) -> tuple:
    """Identity of a trade row; the Data API has no explicit trade ID."""
    return (
        trade.transaction_hash,
        trade.proxy_wallet,
        trade.asset,
        trade.side,
        trade.timestamp,
        trade.size,
        trade.price,
    )


def _load_condition_ids() -> list[str]:
    """Condition IDs from the markets dataset, in stable dataset order."""
    files = sorted(MARKETS_DIR.glob("markets_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
    ids: list[str] = []
    seen: set[str] = set()
    for path in files:
        for cid in pd.read_parquet(path, columns=["condition_id"])["condition_id"]:
            if cid and cid not in seen:
                seen.add(cid)
                ids.append(cid)
    return ids


def _read_cursor() -> tuple[int, Optional[int]]:
    """Parse the cursor file into (market index, optional window resume timestamp)."""
    try:
        idx_part, _, ts_part = CURSOR_FILE.read_text().strip().partition(",")
        return int(idx_part), int(ts_part) if ts_part else None
    except (ValueError, TypeError):
        return 0, None


class PolymarketDataApiTradesIndexer(Indexer):
    """Backfills the public trade tape from data-api.polymarket.com per market.

    The API only honors `start`/`end` on market-scoped queries (the unscoped
    market-wide tape silently ignores them) and caps `offset` at 10,000, so
    deep history is unreachable without scoping. The indexer iterates
    condition IDs from the markets dataset in batches, fetches each batch's
    full trade history, and recursively splits any timestamp window that
    overflows the offset cap.
    """

    def __init__(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        markets_per_request: int = MARKETS_PER_REQUEST,
        limit: int = 1000,
    ):
        super().__init__(
            name="polymarket_data_api_trades",
            description="Backfills the Polymarket Data API trade tape to parquet files",
        )
        self._start = start
        self._end = end
        self._markets_per_request = markets_per_request
        self._limit = limit

    def run(self) -> None:
        """Backfill Data API trades for every known market.

        Progress is cursored as `<market index>[,<window timestamp>]` so both
        an interrupt between markets and one inside a large market's window
        walk resume without refetching completed work.
        """
        condition_ids = _load_condition_ids()
        if not condition_ids:
            print(f"No markets found in {MARKETS_DIR}; run the polymarket_markets indexer first")
            return

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        client = PolymarketClient()

        # Explicit full history by default: scoped queries without `start`
        # (or with 0) are silently limited to the API's ~3-year window.
        start = self._start if self._start is not None else FULL_HISTORY_START
        end = self._end if self._end is not None else int(time.time())

        market_idx, resume_ts = 0, None
        if CURSOR_FILE.exists():
            market_idx, resume_ts = _read_cursor()
            if market_idx or resume_ts:
                suffix = f" at timestamp {resume_ts}" if resume_ts else ""
                print(f"Resuming from market {market_idx}{suffix}")

        print(f"Fetching Data API trades for {len(condition_ids)} markets, timestamps {start} to {end}")

        def iter_leaf_windows(market: str, w_start: int, w_end: int):
            """Yield (window end, trades) in timestamp order for windows under the offset cap."""
            trades, truncated = client.get_data_trades_window(
                w_start, w_end, limit=self._limit, taker_only=True, market=market
            )
            if truncated and w_end - w_start + 1 > MIN_WINDOW_SIZE:
                mid = (w_start + w_end) // 2
                yield from iter_leaf_windows(market, w_start, mid)
                yield from iter_leaf_windows(market, mid + 1, w_end)
                return
            if truncated:
                tqdm.write(
                    f"Warning: window {w_start}-{w_end} overflows the offset cap "
                    "at minimum window size; trades beyond the cap are dropped"
                )
            yield w_end, trades

        def get_next_chunk_idx():
            existing = list(DATA_DIR.glob("trades_*.parquet"))
            if not existing:
                return 0
            indices = []
            for f in existing:
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    try:
                        indices.append(int(parts[1]))
                    except ValueError:
                        pass
            return max(indices) + BATCH_SIZE if indices else 0

        buffer = []
        total_saved = 0

        def save_batch(trades_batch):
            nonlocal total_saved
            if not trades_batch:
                return
            chunk_idx = get_next_chunk_idx()
            chunk_path = DATA_DIR / f"trades_{chunk_idx}_{chunk_idx + BATCH_SIZE}.parquet"
            pd.DataFrame(trades_batch).to_parquet(chunk_path)
            total_saved += len(trades_batch)
            tqdm.write(f"Saved {len(trades_batch)} trades to {chunk_path.name}")

        pbar = tqdm(
            total=len(condition_ids),
            initial=min(market_idx, len(condition_ids)),
            desc="Backfilling",
            unit=" markets",
        )

        interrupted = False
        try:
            while market_idx < len(condition_ids):
                batch = condition_ids[market_idx : market_idx + self._markets_per_request]
                batch_start = resume_ts if resume_ts is not None else start
                resume_ts = None
                fetched_at = datetime.utcnow()

                # Keys seen in the previous window guard against the API
                # returning boundary rows in two adjacent windows.
                prev_keys: set = set()
                for w_end, trades in iter_leaf_windows(",".join(batch), batch_start, end):
                    window_keys: set = set()
                    for trade in trades:
                        key = _trade_key(trade)
                        if key in prev_keys or key in window_keys:
                            continue
                        window_keys.add(key)
                        record = asdict(trade)
                        record["_fetched_at"] = fetched_at
                        buffer.append(record)
                    prev_keys = window_keys

                    while len(buffer) >= BATCH_SIZE:
                        save_batch(buffer[:BATCH_SIZE])
                        buffer = buffer[BATCH_SIZE:]

                    if w_end < end:
                        CURSOR_FILE.write_text(f"{market_idx},{w_end + 1}")

                market_idx += len(batch)
                pbar.update(len(batch))
                pbar.set_postfix(buffer=len(buffer), saved=total_saved)
                CURSOR_FILE.write_text(str(market_idx))

        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted. Progress saved.")
        finally:
            pbar.close()

        # Save remaining trades
        if buffer:
            save_batch(buffer)

        # Only clean up cursor on successful completion
        if not interrupted and CURSOR_FILE.exists():
            CURSOR_FILE.unlink()

        client.close()
        print(f"\nBackfill complete: {total_saved} trades saved")
