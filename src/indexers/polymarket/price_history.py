"""Indexer for Polymarket CLOB price history data."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from tqdm import tqdm

from src.common.indexer import Indexer
from src.indexers.polymarket.client import PolymarketClient

DATA_DIR = Path("data/polymarket/price_history")
MARKETS_DIR = Path("data/polymarket/markets")
BATCH_SIZE = 10000
FIDELITY_MINUTES = 60
WINDOW_SECONDS = 30 * 24 * 60 * 60  # max span per /prices-history request; longer ranges get truncated
END_PADDING_SECONDS = 7 * 24 * 60 * 60  # markets can keep trading past their scheduled end until resolution
EARLIEST_TS = 1577836800  # 2020-01-01, before the first Polymarket market


def _to_epoch_seconds(value: Optional[datetime], default: int) -> int:
    try:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp())
    except (AttributeError, TypeError, ValueError, OverflowError):
        return default


def _discover_token_windows() -> dict[str, tuple[int, int]]:
    """Map every CLOB token ID in the markets dataset to its (start_ts, end_ts) fetch window."""
    rows = duckdb.sql(f"SELECT clob_token_ids, created_at, end_date FROM '{MARKETS_DIR}/markets_*.parquet'").fetchall()
    now_ts = int(datetime.now(timezone.utc).timestamp())

    windows: dict[str, tuple[int, int]] = {}
    for clob_token_ids, created_at, end_date in rows:
        try:
            token_ids = json.loads(clob_token_ids)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(token_ids, list):
            continue

        start_ts = _to_epoch_seconds(created_at, EARLIEST_TS)
        end_ts = min(_to_epoch_seconds(end_date, now_ts) + END_PADDING_SECONDS, now_ts)
        end_ts = max(end_ts, start_ts + 1)

        for token_id in token_ids:
            if not isinstance(token_id, str) or not token_id:
                continue
            if token_id in windows:
                prev_start, prev_end = windows[token_id]
                windows[token_id] = (min(prev_start, start_ts), max(prev_end, end_ts))
            else:
                windows[token_id] = (start_ts, end_ts)

    return windows


class PolymarketPriceHistoryIndexer(Indexer):
    """Fetches and stores Polymarket CLOB price history data."""

    def __init__(self, max_workers: int = 10):
        super().__init__(
            name="polymarket_price_history",
            description="Backfills Polymarket price history data to parquet files",
        )
        self._max_workers = max_workers

    def run(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if not list(MARKETS_DIR.glob("markets_*.parquet")):
            print(f"No markets data found in {MARKETS_DIR}, run the polymarket_markets indexer first")
            return

        token_windows = _discover_token_windows()
        print(f"Found {len(token_windows)} unique tokens across all markets")

        # Load existing token IDs for deduplication (this is also the resume mechanism)
        existing_tokens: set[str] = set()
        parquet_files = list(DATA_DIR.glob("prices_*.parquet"))
        if parquet_files:
            print("Loading existing tokens for deduplication...")
            try:
                result = duckdb.sql(f"SELECT DISTINCT token_id FROM '{DATA_DIR}/prices_*.parquet'").fetchall()
                existing_tokens = {row[0] for row in result}
                print(f"Found {len(existing_tokens)} existing tokens")
            except Exception:
                pass

        tokens_to_process = [(t, w) for t, w in sorted(token_windows.items()) if t not in existing_tokens]
        print(
            f"Skipped {len(token_windows) - len(tokens_to_process)} already processed, "
            f"{len(tokens_to_process)} to fetch"
        )

        if not tokens_to_process:
            print("Nothing to process")
            return

        all_points: list[dict] = []
        total_points_saved = 0
        next_chunk_idx = 0

        # Calculate next chunk index
        if parquet_files:
            indices = []
            for f in parquet_files:
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    try:
                        indices.append(int(parts[1]))
                    except ValueError:
                        pass
            if indices:
                next_chunk_idx = max(indices) + BATCH_SIZE

        def save_batch(points_batch: list[dict]) -> int:
            nonlocal next_chunk_idx
            if not points_batch:
                return 0
            chunk_path = DATA_DIR / f"prices_{next_chunk_idx}_{next_chunk_idx + BATCH_SIZE}.parquet"
            pd.DataFrame(points_batch).to_parquet(chunk_path)
            next_chunk_idx += BATCH_SIZE
            return len(points_batch)

        def fetch_token_history(token_id: str, start_ts: int, end_ts: int) -> list[dict]:
            """Fetch the full price history for a single token in bounded windows."""
            client = PolymarketClient()
            try:
                fetched_at = datetime.utcnow()
                records: list[dict] = []
                seen_ts: set[int] = set()
                cursor = start_ts
                while cursor < end_ts:
                    window_end = min(cursor + WINDOW_SECONDS, end_ts)
                    points = client.get_price_history(
                        token_id,
                        interval=None,
                        fidelity=FIDELITY_MINUTES,
                        start_ts=cursor,
                        end_ts=window_end,
                    )
                    for point in points:
                        if point.timestamp not in seen_ts:
                            seen_ts.add(point.timestamp)
                            records.append({**asdict(point), "_fetched_at": fetched_at})
                    cursor = window_end
                return records
            finally:
                client.close()

        # Concurrent fetching
        pbar = tqdm(total=len(tokens_to_process), desc="Fetching price history")
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(fetch_token_history, token, start, end): token
                for token, (start, end) in tokens_to_process
            }

            for future in as_completed(futures):
                token = futures[future]
                try:
                    points_data = future.result()
                    if points_data:
                        all_points.extend(points_data)

                    pbar.update(1)
                    pbar.set_postfix(buffer=len(all_points), saved=total_points_saved, last=token[-20:])

                    # Save in batches
                    while len(all_points) >= BATCH_SIZE:
                        saved = save_batch(all_points[:BATCH_SIZE])
                        total_points_saved += saved
                        all_points = all_points[BATCH_SIZE:]

                except Exception as e:
                    pbar.update(1)
                    tqdm.write(f"Error fetching {token}: {e}")

        pbar.close()

        # Save remaining
        if all_points:
            total_points_saved += save_batch(all_points)

        print(
            f"\nBackfill price history complete: {len(tokens_to_process)} tokens processed, "
            f"{total_points_saved} points saved"
        )
