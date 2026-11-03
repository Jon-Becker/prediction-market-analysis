"""Indexer for Kalshi historical markets data (the /historical/markets endpoint)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from src.common.indexer import Indexer
from src.common.storage import ParquetStorage
from src.common.util.prompts import prompt_date_range
from src.indexers.kalshi.client import KalshiClient

DATA_DIR = Path("data/kalshi/historical/markets")
CURSOR_FILE = Path("data/kalshi/historical/.backfill_cursor")


class KalshiHistoricalMarketsIndexer(Indexer):
    """Fetches and stores Kalshi historical markets, optionally limited to a close-date range."""

    def __init__(self):
        super().__init__(
            name="kalshi_historical_markets",
            description="Backfills Historical Kalshi markets data to parquet files",
        )

    def run(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        client = KalshiClient()
        storage = ParquetStorage(data_dir=DATA_DIR)

        cursor = None
        if CURSOR_FILE.exists():
            cursor = CURSOR_FILE.read_text().strip() or None
            if cursor:
                print(f"Resuming from cursor: {cursor[:20]}...")

        # The historical endpoint has no close-date filter, so filter client-side
        print("Enter close-date range for markets")
        min_close, max_close = prompt_date_range(utc=True)
        # Make the end date inclusive of the whole day
        before_close = max_close + timedelta(days=1) if max_close else None

        total = 0
        for markets, next_cursor in client.iter_historical_markets(limit=1000, cursor=cursor):
            filtered_markets = [
                m
                for m in markets
                if (min_close is None or m.close_time >= min_close)
                and (before_close is None or m.close_time < before_close)
            ]

            if filtered_markets:
                total_stored = storage.append_markets(filtered_markets)
                total += len(filtered_markets)
                print(f"Fetched {len(filtered_markets)} markets (total: {total}, stored: {total_stored})")

            if next_cursor:
                CURSOR_FILE.write_text(next_cursor)
            else:
                if CURSOR_FILE.exists():
                    CURSOR_FILE.unlink()
                break

        print(f"\nBackfill complete: {total} markets fetched")
