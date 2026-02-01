"""Indexer for Polymarket trades from the Polygon blockchain."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.common.indexer import Indexer
from src.indexers.polymarket.blockchain import (
    CTF_EXCHANGE,
    NEGRISK_CTF_EXCHANGE,
    POLYMARKET_START_BLOCK,
    PolygonClient,
)

DATA_DIR = Path("data/polymarket/trades")
CURSOR_FILE = Path("data/polymarket/.backfill_block_cursor")


class PolymarketTradesIndexer(Indexer):
    """Fetches and stores Polymarket trades from the Polygon blockchain."""

    def __init__(
        self,
        from_block: Optional[int] = None,
        to_block: Optional[int] = None,
        chunk_size: int = 1000,
    ):
        super().__init__(
            name="polymarket_trades",
            description="Backfills Polymarket trades from Polygon blockchain to parquet files",
        )
        self._from_block = from_block
        self._to_block = to_block
        self._chunk_size = chunk_size

    def run(self) -> None:
        """Backfill all Polymarket trades from the Polygon blockchain.

        This fetches OrderFilled events from both CTF Exchange contracts
        (regular and NegRisk) and saves them to parquet files.
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
                    from_block = POLYMARKET_START_BLOCK
            else:
                from_block = POLYMARKET_START_BLOCK

        to_block = self._to_block
        if to_block is None:
            to_block = current_block

        print(f"Fetching trades from block {from_block} to {to_block}")
        print(f"Total blocks: {to_block - from_block:,}")

        all_trades = []
        total_saved = 0
        contracts = [
            ("CTF Exchange", CTF_EXCHANGE),
            ("NegRisk CTF Exchange", NEGRISK_CTF_EXCHANGE),
        ]

        def get_next_chunk_idx():
            existing = list(DATA_DIR.glob("trades_*.parquet"))
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

        def save_batch(trades_batch):
            nonlocal total_saved
            if not trades_batch:
                return
            chunk_idx = get_next_chunk_idx()
            chunk_path = DATA_DIR / f"trades_{chunk_idx}_{chunk_idx + BATCH_SIZE}.parquet"
            df = pd.DataFrame(trades_batch)
            df.to_parquet(chunk_path)
            total_saved += len(trades_batch)
            tqdm.write(f"Saved {len(trades_batch)} trades to {chunk_path.name}")

        # Build list of chunk ranges
        ranges = []
        current = from_block
        while current <= to_block:
            end = min(current + self._chunk_size - 1, to_block)
            ranges.append((current, end))
            current = end + 1

        # Process by block range, fetching from both contracts for each range
        total_chunks = len(ranges)
        pbar = tqdm(total=total_chunks, desc="Backfilling", unit=" chunks")

        try:
            for chunk_start, chunk_end in ranges:
                fetched_at = datetime.utcnow()

                # Fetch from both contracts for this block range
                for contract_name, contract_address in contracts:
                    trades = client.get_trades(
                        from_block=chunk_start,
                        to_block=chunk_end,
                        contract_address=contract_address,
                    )

                    for trade in trades:
                        trade_dict = asdict(trade)
                        # Convert large ints to strings to avoid parquet overflow
                        trade_dict["maker_asset_id"] = str(trade_dict["maker_asset_id"])
                        trade_dict["taker_asset_id"] = str(trade_dict["taker_asset_id"])
                        trade_dict["_fetched_at"] = fetched_at
                        trade_dict["_contract"] = contract_name
                        all_trades.append(trade_dict)

                # Update progress after both contracts processed for this range
                pbar.update(1)
                pbar.set_postfix(
                    block=chunk_end,
                    buffer=len(all_trades),
                    saved=total_saved,
                )

                # Save in batches
                while len(all_trades) >= BATCH_SIZE:
                    save_batch(all_trades[:BATCH_SIZE])
                    all_trades = all_trades[BATCH_SIZE:]

                # Save cursor after both contracts processed for this range
                CURSOR_FILE.write_text(str(chunk_end))

        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved.")
        finally:
            pbar.close()

        # Save remaining trades
        if all_trades:
            save_batch(all_trades)

        if CURSOR_FILE.exists():
            CURSOR_FILE.unlink()

        print(f"\nBackfill complete: {total_saved} trades saved")
