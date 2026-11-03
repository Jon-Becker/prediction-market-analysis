"""Indexer for Kalshi markets, fetched through the /events endpoint with nested markets."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb

from src.common.indexer import Indexer
from src.common.storage import ParquetStorage
from src.common.util.prompts import prompt_choice, prompt_date_range
from src.indexers.kalshi.client import KalshiClient

DATA_DIR = Path("data/kalshi/markets")
CURSOR_FILE = Path("data/kalshi/.backfill_event_nested_markets_cursor")

MODE_DATE_RANGE = "1"
MODE_NEW_ONLY = "2"


class KalshiEventNestedMarketsIndexer(Indexer):
    """Fetches settled Kalshi markets, either for a date range or only ones newer than what's saved."""

    def __init__(self):
        super().__init__(
            name="kalshi_event_nested_markets",
            description="Backfills settled Kalshi markets (via events) to parquet files",
        )

    @staticmethod
    def _latest_saved_close_time() -> datetime | None:
        """Close time of the newest finalized market already saved."""
        if not any(DATA_DIR.glob("markets_*.parquet")):
            return None
        latest = duckdb.sql(f"""
            SELECT MAX(close_time)
            FROM '{DATA_DIR}/markets_*.parquet'
            WHERE status = 'finalized'
            AND volume >= 100
        """).fetchone()[0]
        return latest

    def run(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        print("1) Fetch markets closing in a date range")
        print("2) Fetch only markets newer than the ones already saved")
        mode = prompt_choice("Choose 1 or 2: ", [MODE_DATE_RANGE, MODE_NEW_ONLY])

        min_close: datetime | None = None
        max_close: datetime | None = None
        if mode == MODE_DATE_RANGE:
            min_close, max_close = prompt_date_range()
        else:
            min_close = self._latest_saved_close_time()
            if min_close is None:
                print("No saved markets found, fetching everything")
            else:
                print(f"Fetching markets that closed after {min_close}")

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
            min_close_ts=int(min_close.timestamp()) if min_close else None,
            max_close_ts=int(max_close.timestamp()) if max_close else None,
            status="settled",
        ):
            if mode == MODE_NEW_ONLY and min_close is not None:
                # The API filter is inclusive and also returns settled-but-not-finalized markets
                markets = [m for m in markets if m.close_time > min_close and m.status == "finalized"]

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
