"""Indexer for Kalshi markets data."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.common.indexer import Indexer
from src.common.storage import ParquetStorage
from src.indexers.kalshi.client import KalshiClient

DATA_DIR = Path("data/kalshi/historical/markets")
CURSOR_FILE = Path("data/kalshi/historical/.backfill_cursor")


class KalshiHistoricalMarketsIndexer(Indexer):
    """Fetches and stores Kalshi markets data."""

    def __init__(
        self,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
    ):
        super().__init__(
            name="kalshi_historical_markets",
            description="Backfills Historical Kalshi markets data to parquet files",
        )
        self._min_close_ts = min_close_ts
        self._max_close_ts = max_close_ts

    @staticmethod
    def _get_date_input(prompt: str) -> Optional[int]:
        while True:
            try:
                date_str = input(prompt)

                if not date_str:
                    return None

                return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD (e.g., 2026-03-01)")

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

        self._min_close = self._get_date_input("Enter minimum close date (YYYY-MM-DD) or leave blank for no minimum: ")
        self._max_close = self._get_date_input("Enter maximum close date (YYYY-MM-DD) or leave blank for no maximum: ")

        total = 0
        for markets, next_cursor in client.iter_historical_markets(limit=1000, cursor=cursor):
            filtered_markets = [
                m
                for m in markets
                if (self._min_close is None or m.close_time >= self._min_close)
                and (self._max_close is None or m.close_time <= self._max_close)
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
