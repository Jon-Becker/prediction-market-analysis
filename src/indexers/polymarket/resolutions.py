"""Indexer for Polymarket condition resolutions from the Polygon blockchain."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.common.indexer import Indexer
from src.indexers.polymarket.blockchain import (
    CONDITIONAL_TOKENS_START_BLOCK,
    PolygonClient,
)

DATA_DIR = Path("data/polymarket/resolutions")
CURSOR_FILE = Path("data/polymarket/.resolutions_block_cursor")


class PolymarketResolutionsIndexer(Indexer):
    """Fetches and stores Polymarket condition resolutions from the Polygon blockchain."""

    def __init__(
        self,
        from_block: Optional[int] = None,
        to_block: Optional[int] = None,
        chunk_size: int = 10000,
    ):
        super().__init__(
            name="polymarket_resolutions",
            description="Backfills Polymarket condition resolutions from Polygon blockchain to parquet files",
        )
        self._from_block = from_block
        self._to_block = to_block
        self._chunk_size = chunk_size

    def run(self) -> None:
        """Backfill all Polymarket condition resolutions from the Polygon blockchain.

        This fetches ConditionResolution events from the Gnosis ConditionalTokens
        contract and saves them to parquet files.
        """
        BATCH_SIZE = 10000
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        client = PolygonClient()
        current_block = client.get_block_number()

        # Determine starting block
        from_block = self._from_block
        if from_block is None:
            if CURSOR_FILE.exists():
                try:
                    from_block = int(CURSOR_FILE.read_text().strip())
                    print(f"Resuming from block {from_block}")
                except (ValueError, TypeError):
                    from_block = CONDITIONAL_TOKENS_START_BLOCK
            else:
                from_block = CONDITIONAL_TOKENS_START_BLOCK

        to_block = self._to_block
        if to_block is None:
            to_block = current_block

        print(f"Fetching resolutions from block {from_block} to {to_block}")
        print(f"Total blocks: {to_block - from_block:,}")

        all_resolutions = []
        total_saved = 0

        def get_next_chunk_idx():
            existing = list(DATA_DIR.glob("resolutions_*.parquet"))
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

        def save_batch(resolutions_batch):
            nonlocal total_saved
            if not resolutions_batch:
                return
            chunk_idx = get_next_chunk_idx()
            chunk_path = DATA_DIR / f"resolutions_{chunk_idx}_{chunk_idx + BATCH_SIZE}.parquet"
            df = pd.DataFrame(resolutions_batch)
            df.to_parquet(chunk_path)
            total_saved += len(resolutions_batch)
            tqdm.write(f"Saved {len(resolutions_batch)} resolutions to {chunk_path.name}")

        # Build list of chunk ranges
        ranges = []
        current = from_block
        while current <= to_block:
            end = min(current + self._chunk_size - 1, to_block)
            ranges.append((current, end))
            current = end + 1

        total_chunks = len(ranges)
        pbar = tqdm(total=total_chunks, desc="Backfilling", unit=" chunks")

        completed = False
        try:
            for chunk_start, chunk_end in ranges:
                fetched_at = datetime.utcnow()

                resolutions = client.get_condition_resolutions(
                    from_block=chunk_start,
                    to_block=chunk_end,
                )

                for resolution in resolutions:
                    record = asdict(resolution)
                    # Serialize to a JSON string to avoid parquet overflow
                    record["payout_numerators"] = json.dumps(record["payout_numerators"])
                    record["_fetched_at"] = fetched_at
                    all_resolutions.append(record)

                pbar.update(1)
                pbar.set_postfix(
                    block=chunk_end,
                    buffer=len(all_resolutions),
                    saved=total_saved,
                )

                # Save in batches
                while len(all_resolutions) >= BATCH_SIZE:
                    save_batch(all_resolutions[:BATCH_SIZE])
                    all_resolutions = all_resolutions[BATCH_SIZE:]

                # Save cursor after each completed range
                CURSOR_FILE.write_text(str(chunk_end))

            completed = True
        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved.")
        finally:
            pbar.close()
            # Flush buffered resolutions so no fetched data is lost on interrupt or error
            if all_resolutions:
                save_batch(all_resolutions)

        # Only clean up cursor on successful completion
        if completed and CURSOR_FILE.exists():
            CURSOR_FILE.unlink()

        print(f"\nBackfill complete: {total_saved} resolutions saved")
