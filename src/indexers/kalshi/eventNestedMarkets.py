"""Indexer for Kalshi markets data."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from src.common.indexer import Indexer
from src.common.storage import ParquetStorage
from src.indexers.kalshi.client import KalshiClient

DATA_DIR = Path("data/kalshi/markets")
CURSOR_FILE = Path("data/kalshi/.backfill_cursor")


class KalshiEventNestedMarketsIndexer(Indexer):
    """Fetches and stores Kalshi markets data."""

    def __init__(
        self,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
    ):
        super().__init__(
            name="kalshi_event_nested_markets",
            description="Backfills Kalshi event-nested markets data to parquet files",
        )

    @staticmethod
    def _get_date_input(prompt: str) -> Optional[int]:
        while True:
            try:
                date_str = input(prompt)

                if not date_str:
                    return None

                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                timestamp = int(date_obj.timestamp())
                # print(f"Parsed as: {date_obj} (timestamp: {timestamp})")
                return timestamp
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD (e.g., 2026-03-01)")

    def run(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        min_close_ts = self._get_date_input("Enter minimum close date (YYYY-MM-DD): ")
        max_close_ts = self._get_date_input("Enter maximum close date (YYYY-MM-DD): ")

        if min_close_ts and min_close_ts > max_close_ts:
            print("Error: minimum date cannot be after maximum date")
            return

        client = KalshiClient()
        storage = ParquetStorage(data_dir=DATA_DIR)

        cursor = None
        if CURSOR_FILE.exists():
            cursor = CURSOR_FILE.read_text().strip() or None
            if cursor:
                print(f"Resuming from cursor: {cursor[:20]}...")

        total = 0
        for markets, next_cursor in client.iter_event_nested_markets(
            limit=200,
            cursor=cursor,
            min_close_ts=min_close_ts,
            max_close_ts=max_close_ts,
            status="settled",  # Only fetch settled markets
        ):
            if markets:
                total_stored = storage.append_markets(markets)
                total += len(markets)
                print(f"Fetched {len(markets)} markets (total: {total}, stored: {total_stored})")

            if next_cursor:
                CURSOR_FILE.write_text(next_cursor)
            else:
                if CURSOR_FILE.exists():
                    CURSOR_FILE.unlink()
                break

        print(f"\nBackfill complete: {total} markets fetched")
