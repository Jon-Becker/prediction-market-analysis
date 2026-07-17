"""Indexer for the Polymarket Data API market-wide trade tape."""

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
CURSOR_FILE = Path("data/polymarket/.data_api_trades_cursor")
BATCH_SIZE = 10000
WINDOW_SIZE = 86400  # seconds per timestamp window
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


class PolymarketDataApiTradesIndexer(Indexer):
    """Backfills the public market-wide trade tape from data-api.polymarket.com."""

    def __init__(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        window_size: int = WINDOW_SIZE,
        limit: int = 1000,
    ):
        super().__init__(
            name="polymarket_data_api_trades",
            description="Backfills the Polymarket Data API trade tape to parquet files",
        )
        self._start = start
        self._end = end
        self._window_size = window_size
        self._limit = limit

    def run(self) -> None:
        """Backfill all Data API trades by walking timestamp windows.

        The API caps `offset` at 10,000, so the tape is fetched in [start, end]
        timestamp windows; any window that overflows the cap is split in half
        recursively until every returned trade is covered.
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        client = PolymarketClient()

        # Determine starting timestamp; default is explicit full history (start=1),
        # since omitting `start` silently limits the API to the last ~3 years.
        start = self._start
        if start is None:
            if CURSOR_FILE.exists():
                try:
                    start = int(CURSOR_FILE.read_text().strip())
                    print(f"Resuming from timestamp {start}")
                except (ValueError, TypeError):
                    start = FULL_HISTORY_START
            else:
                start = FULL_HISTORY_START

        end = self._end
        if end is None:
            end = int(time.time())

        print(f"Fetching Data API trades from timestamp {start} to {end}")

        def fetch_window(w_start: int, w_end: int) -> list[DataApiTrade]:
            trades, truncated = client.get_data_trades_window(w_start, w_end, limit=self._limit, taker_only=True)
            if not truncated:
                return trades
            if w_end - w_start + 1 <= MIN_WINDOW_SIZE:
                tqdm.write(
                    f"Warning: window {w_start}-{w_end} overflows the offset cap "
                    "at minimum window size; trades beyond the cap are dropped"
                )
                return trades
            mid = (w_start + w_end) // 2
            return fetch_window(w_start, mid) + fetch_window(mid + 1, w_end)

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

        # Build list of timestamp windows
        ranges = []
        current = start
        while current <= end:
            w_end = min(current + self._window_size - 1, end)
            ranges.append((current, w_end))
            current = w_end + 1

        pbar = tqdm(total=len(ranges), desc="Backfilling", unit=" windows")

        # Keys seen in the previous window guard against the API returning
        # boundary rows in two adjacent windows.
        prev_keys: set = set()

        interrupted = False
        try:
            for w_start, w_end in ranges:
                fetched_at = datetime.utcnow()
                window_keys: set = set()

                for trade in fetch_window(w_start, w_end):
                    key = _trade_key(trade)
                    if key in prev_keys or key in window_keys:
                        continue
                    window_keys.add(key)
                    record = asdict(trade)
                    record["_fetched_at"] = fetched_at
                    buffer.append(record)

                prev_keys = window_keys

                pbar.update(1)
                pbar.set_postfix(
                    timestamp=w_end,
                    buffer=len(buffer),
                    saved=total_saved,
                )

                while len(buffer) >= BATCH_SIZE:
                    save_batch(buffer[:BATCH_SIZE])
                    buffer = buffer[BATCH_SIZE:]

                CURSOR_FILE.write_text(str(w_end + 1))

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
