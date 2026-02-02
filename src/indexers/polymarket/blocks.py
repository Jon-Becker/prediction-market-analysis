"""Indexer for block timestamps from the Polygon blockchain."""

import concurrent.futures
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.common.indexer import Indexer
from src.indexers.polymarket.blockchain import PolygonClient

POLYGON_RPC = os.getenv("POLYGON_RPC", "")
BLOCKS_DIR = Path("data/polymarket/blocks")

BUCKET_SIZE = 100_000  # 100k blocks per file
SAMPLE_INTERVAL = 100  # Fetch every 100th block, interpolate the rest
MAX_WORKERS = 100


class PolymarketBlocksIndexer(Indexer):
    """Builds a mapping from block number to timestamp for every block."""

    def __init__(self):
        super().__init__(
            name="polymarket_blocks",
            description="Fetches block timestamps for every block",
        )

    def _fetch_timestamp(self, client: PolygonClient, block_number: int) -> Optional[tuple[int, int]]:
        """Fetch timestamp for a single block. Returns (block_number, unix_timestamp)."""
        try:
            unix_timestamp = client.get_block_timestamp(block_number)
            return (block_number, unix_timestamp)
        except Exception as e:
            tqdm.write(f"Error fetching block {block_number}: {e}")
            return None

    def _interpolate_timestamps(self, sampled: list[tuple[int, int]], start_block: int, end_block: int) -> list[dict]:
        """Interpolate timestamps for all blocks between sampled points."""
        sampled_sorted = sorted(sampled, key=lambda x: x[0])
        records = []

        for i in range(len(sampled_sorted) - 1):
            block_a, ts_a = sampled_sorted[i]
            block_b, ts_b = sampled_sorted[i + 1]

            block_diff = block_b - block_a
            ts_diff = ts_b - ts_a

            for block in range(block_a, block_b):
                offset = block - block_a
                interpolated_ts = ts_a + (ts_diff * offset) // block_diff
                timestamp_str = datetime.fromtimestamp(interpolated_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                records.append({"block_number": block, "timestamp": timestamp_str})

        # Add the last sampled block
        if sampled_sorted:
            last_block, last_ts = sampled_sorted[-1]
            timestamp_str = datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            records.append({"block_number": last_block, "timestamp": timestamp_str})

        return records

    def _get_last_indexed_block(self) -> int:
        """Get the highest block number from existing files based on filename."""
        if not BLOCKS_DIR.exists():
            return 0

        parquet_files = list(BLOCKS_DIR.glob("blocks_*.parquet"))
        if not parquet_files:
            return 0

        max_block = 0
        pattern = re.compile(r"blocks_(\d+)_(\d+)\.parquet")
        for f in parquet_files:
            match = pattern.match(f.name)
            if match:
                end_block = int(match.group(2))
                max_block = max(max_block, end_block)

        return max_block

    def _get_latest_block(self, client: PolygonClient) -> int:
        """Get the latest block number from the chain."""
        return client.get_block_number()

    def run(self) -> None:
        """Fetch timestamps for every block, continuing from where we left off."""
        BLOCKS_DIR.mkdir(parents=True, exist_ok=True)

        client = PolygonClient()

        last_indexed = self._get_last_indexed_block()
        latest_block = self._get_latest_block(client)

        print(f"Last indexed block: {last_indexed:,}")
        print(f"Latest chain block: {latest_block:,}")

        # Start from the next bucket boundary
        start_block = last_indexed
        if start_block == 0:
            start_block = (latest_block // BUCKET_SIZE) * BUCKET_SIZE - BUCKET_SIZE * 10

        blocks_remaining = latest_block - start_block
        print(f"Blocks to fetch: {blocks_remaining:,}")

        if blocks_remaining <= 0:
            print("Already up to date")
            return

        # Process in buckets of BUCKET_SIZE
        current_bucket_start = start_block
        while current_bucket_start < latest_block:
            bucket_end = min(current_bucket_start + BUCKET_SIZE, latest_block + 1)

            # Only fetch every SAMPLE_INTERVAL blocks, plus bucket boundaries
            sampled_blocks = list(range(current_bucket_start, bucket_end, SAMPLE_INTERVAL))
            if sampled_blocks[-1] != bucket_end - 1:
                sampled_blocks.append(bucket_end - 1)

            print(
                f"\nFetching {len(sampled_blocks):,} samples for blocks {current_bucket_start:,} to {bucket_end - 1:,}"
            )

            sampled_timestamps = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(self._fetch_timestamp, client, block): block for block in sampled_blocks}

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Fetching samples",
                ):
                    result = future.result()
                    if result:
                        sampled_timestamps.append(result)

            if sampled_timestamps:
                records = self._interpolate_timestamps(sampled_timestamps, current_bucket_start, bucket_end)
                self._save_bucket(records, current_bucket_start, bucket_end)

            current_bucket_start = bucket_end

        print("\nIndexing complete")

    def _save_bucket(self, records: list[dict], start_block: int, end_block: int) -> None:
        """Save a bucket of records to a parquet file."""
        df = pd.DataFrame(records)
        df = df.sort_values("block_number").reset_index(drop=True)

        output_path = BLOCKS_DIR / f"blocks_{start_block}_{end_block}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df)} blocks to {output_path.name}")
