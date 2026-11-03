"""Indexer for Kalshi markets data."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb

from src.common.indexer import Indexer
from src.common.storage import ParquetStorage
from src.indexers.kalshi.client import KalshiClient

DATA_DIR = Path("data/kalshi/markets")
CURSOR_FILE = Path("data/kalshi/.backfill_cursor")


class KalshiNewEventNestedMarketsIndexer(Indexer):
    """Fetches and stores Kalshi markets data."""

    def __init__(
        self,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
    ):
        super().__init__(
            name="kalshi_new_event_nested_markets",
            description="Backfills new Kalshi event-nested markets data to parquet files",
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

        con = duckdb.connect()

        df_markets_min_max = con.execute(
            """
                WITH resolved_markets as(
                        SELECT close_time as date
                        FROM 'data/kalshi/markets/markets_*.parquet' m
                        WHERE 1=1
                        AND status = 'finalized'
                        AND volume >= 100
                        ORDER BY date ASC
                    )
                    SELECT MIN(date) as min,
                            MAX(date) as max
                    FROM resolved_markets
            """
        ).df()

        markets_min = df_markets_min_max["min"][0]
        markets_max = df_markets_min_max["max"][0]

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
            min_close_ts=int(markets_max.timestamp()),
            status="settled",  # Only fetch settled markets
        ):
            newMarkets = []

            if markets:
                for market in markets:
                    if market.close_time > markets_max and market.status == "finalized":
                        newMarkets.append(market)

                total_stored = storage.append_markets(newMarkets)
                total += len(newMarkets)
                print(
                    f"Fetched {len(markets)} markets, {len(newMarkets)} new. (total new: {total}, stored: {total_stored})"
                )

            if next_cursor:
                CURSOR_FILE.write_text(next_cursor)
            else:
                if CURSOR_FILE.exists():
                    CURSOR_FILE.unlink()
                break

        print(f"\nBackfill complete: {total} markets fetched")
