"""Indexer for block timestamps from the Polygon blockchain."""

import concurrent.futures
import os
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from tqdm import tqdm

from src.common.indexer import Indexer
from src.indexers.polymarket.blockchain import PolygonClient

POLYGON_RPC = os.getenv("POLYGON_RPC", "")
TRADES_DIR = Path("data/polymarket/trades")
BLOCKS_DIR = Path("data/polymarket/blocks")


BUCKET_SIZE = 10000
SAMPLE_INTERVAL = 10800  # roughly 4 samples per day. 86400s per day / 2s per block / 4samples per day
MAX_WORKERS = 50


class PolymarketBlocksIndexer(Indexer):
    """Builds a mapping from block number to timestamp, sampled 4 times daily."""

    def __init__(self):
        super().__init__(
            name="polymarket_blocks",
            description="Fetches block timestamps sampled every 4 times daily",
        )

    def _fetch_timestamp(
        self, client: PolygonClient, block_number: int
    ) -> Optional[dict]:
        """Fetch timestamp for a single block."""
        try:
            timestamp = client.get_block_timestamp(block_number)
            return {"block_number": block_number, "timestamp": timestamp}
        except Exception as e:
            tqdm.write(f"Error fetching block {block_number}: {e}")
            return None

    def _get_block_range(self) -> tuple[int, int]:
        """Get min and max block numbers from trades dataset using DuckDB."""
        if not TRADES_DIR.exists():
            raise FileNotFoundError(f"Trades directory not found: {TRADES_DIR}")

        parquet_pattern = str(TRADES_DIR / "*.parquet")
        query = f"""
            SELECT MIN(block_number), MAX(block_number)
            FROM read_parquet('{parquet_pattern}')
        """
        result = duckdb.execute(query).fetchone()
        return result[0], result[1]

    def _get_existing_blocks(self) -> set[int]:
        """Get block numbers already indexed."""
        if not BLOCKS_DIR.exists():
            return set()

        parquet_files = list(BLOCKS_DIR.glob("*.parquet"))
        if not parquet_files:
            return set()

        parquet_pattern = str(BLOCKS_DIR / "*.parquet")
        query = f"""
            SELECT DISTINCT block_number
            FROM read_parquet('{parquet_pattern}')
        """
        result = duckdb.execute(query).fetchall()
        return {row[0] for row in result}

    def run(self) -> None:
        """Fetch timestamps sampled across the trade range."""
        BLOCKS_DIR.mkdir(parents=True, exist_ok=True)

        print("Querying trades dataset for block range...")
        min_block, max_block = self._get_block_range()
        print(f"Block range: {min_block:,} to {max_block:,}")

        # Generate sampled block numbers at SAMPLE_INTERVAL intervals
        start = (min_block // SAMPLE_INTERVAL) * SAMPLE_INTERVAL
        sampled_blocks = list(range(start, max_block + 1, SAMPLE_INTERVAL))
        print(f"Sampled blocks (every {SAMPLE_INTERVAL}): {len(sampled_blocks):,}")

        existing_blocks = self._get_existing_blocks()
        blocks_to_fetch = [b for b in sampled_blocks if b not in existing_blocks]
        print(f"Blocks already indexed: {len(existing_blocks):,}")
        print(f"Blocks to fetch: {len(blocks_to_fetch):,}")

        if not blocks_to_fetch:
            print("All blocks already indexed")
            return

        client = PolygonClient()
        records = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._fetch_timestamp, client, block): block
                for block in blocks_to_fetch
            }

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Fetching timestamps",
            ):
                result = future.result()
                if result:
                    records.append(result)

        # Combine with existing data and save
        if records:
            self._save_all(records)

        print("\nIndexing complete")

    def _save_all(self, records: list[dict]) -> None:
        """Save all records to parquet files, 10k entries each."""
        df_new = pd.DataFrame(records)

        # Load existing data
        parquet_files = list(BLOCKS_DIR.glob("*.parquet"))
        if parquet_files:
            parquet_pattern = str(BLOCKS_DIR / "*.parquet")
            df_existing = duckdb.execute(
                f"SELECT * FROM read_parquet('{parquet_pattern}')"
            ).df()
            df_new = pd.concat([df_existing, df_new], ignore_index=True)
            df_new = df_new.drop_duplicates(subset=["block_number"])

        df_new = df_new.sort_values("block_number").reset_index(drop=True)

        # Clear existing files
        for f in BLOCKS_DIR.glob("*.parquet"):
            f.unlink()

        # Save in chunks of BUCKET_SIZE
        for i in range(0, len(df_new), BUCKET_SIZE):
            chunk = df_new.iloc[i : i + BUCKET_SIZE]
            output_path = BLOCKS_DIR / f"blocks_{i}_{i + BUCKET_SIZE}.parquet"
            chunk.to_parquet(output_path, index=False)
            print(f"Saved {len(chunk)} blocks to {output_path.name}")
