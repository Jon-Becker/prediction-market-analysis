"""Indexer for Polymarket markets data."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.common.indexer import Indexer
from src.indexers.polymarket.client import PolymarketClient

DATA_DIR = Path("data/polymarket/markets")
OFFSET_FILE = Path("data/polymarket/.backfill_offset")
CHUNK_SIZE = 10000


class PolymarketMarketsIndexer(Indexer):
    """Fetches and stores Polymarket markets data."""

    def __init__(self):
        super().__init__(
            name="polymarket_markets",
            description="Backfills Polymarket markets data to parquet files",
        )

    def run(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)

        client = PolymarketClient()

        offset = 0
        if OFFSET_FILE.exists():
            try:
                offset = int(OFFSET_FILE.read_text().strip())
                if offset > 0:
                    print(f"Resuming from offset: {offset}")
            except (ValueError, TypeError):
                offset = 0

        all_markets = []
        total = offset

        for markets, next_offset in client.iter_markets(offset=offset):
            if markets:
                fetched_at = datetime.utcnow()
                for market in markets:
                    record = asdict(market)
                    record["_fetched_at"] = fetched_at
                    all_markets.append(record)

                total += len(markets)
                print(f"Fetched {len(markets)} markets (total: {total})")

                # Save in chunks
                while len(all_markets) >= CHUNK_SIZE:
                    chunk = all_markets[:CHUNK_SIZE]
                    chunk_start = total - len(all_markets)
                    chunk_path = DATA_DIR / f"markets_{chunk_start}_{chunk_start + CHUNK_SIZE}.parquet"
                    pd.DataFrame(chunk).to_parquet(chunk_path)
                    all_markets = all_markets[CHUNK_SIZE:]

            if next_offset > 0:
                OFFSET_FILE.write_text(str(next_offset))
            else:
                break

        # Save remaining markets
        if all_markets:
            chunk_start = total - len(all_markets)
            chunk_path = DATA_DIR / f"markets_{chunk_start}_{chunk_start + len(all_markets)}.parquet"
            pd.DataFrame(all_markets).to_parquet(chunk_path)

        if OFFSET_FILE.exists():
            OFFSET_FILE.unlink()

        client.close()
        print(f"\nBackfill complete: {total} markets fetched")
