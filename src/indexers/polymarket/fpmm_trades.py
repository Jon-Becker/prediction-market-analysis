"""Indexer for Polymarket FPMM trades from the Polygon blockchain."""

import concurrent.futures
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm
from web3 import Web3

from src.common.indexer import Indexer
from src.indexers.polymarket.blockchain import PolygonClient

# FPMM Factory deployed at block 4023693 (around Sep 2020)
FPMM_FACTORY = "0x8b9805a2f595b6705e74f7310829f2d299d21522"
FPMM_START_BLOCK = 4023693


# Event signatures (keccak256 hashes)
# FPMMBuy(address indexed buyer, uint256 investmentAmount, uint256 feeAmount, uint256 indexed outcomeIndex, uint256 outcomeTokensBought)
FPMM_BUY_TOPIC = "0x" + Web3.keccak(text="FPMMBuy(address,uint256,uint256,uint256,uint256)").hex()
# FPMMSell(address indexed seller, uint256 returnAmount, uint256 feeAmount, uint256 indexed outcomeIndex, uint256 outcomeTokensSold)
FPMM_SELL_TOPIC = "0x" + Web3.keccak(text="FPMMSell(address,uint256,uint256,uint256,uint256)").hex()

DATA_DIR = Path("data/polymarket/legacy_trades")
CURSOR_FILE = Path("data/polymarket/.legacy_backfill_block_cursor")


@dataclass
class FPMMTrade:
    """A trade from a Fixed Product Market Maker (FPMM)."""

    block_number: int
    transaction_hash: str
    log_index: int
    fpmm_address: str  # The FPMM contract (market) address
    trader: str  # buyer or seller address
    amount: int  # investmentAmount (buy) or returnAmount (sell) in collateral units
    fee_amount: int
    outcome_index: int
    outcome_tokens: int  # outcomeTokensBought or outcomeTokensSold
    is_buy: bool
    timestamp: Optional[int] = None

    @property
    def price(self) -> float:
        """Calculate price in collateral per token."""
        if self.outcome_tokens > 0:
            return self.amount / self.outcome_tokens
        return 0.0

    @property
    def size(self) -> float:
        """Number of tokens traded (in token units, 18 decimals for FPMM)."""
        return self.outcome_tokens / 1e18

    @property
    def volume(self) -> float:
        """Volume in collateral units (typically 6 decimals for USDC)."""
        return self.amount / 1e6


class PolymarketLegacyTradesIndexer(Indexer):
    """Fetches and stores Polymarket legacy FPMM trades from the Polygon blockchain."""

    def __init__(
        self,
        from_block: Optional[int] = None,
        to_block: Optional[int] = None,
        chunk_size: int = 1000,
        max_workers: int = 50,
    ):
        super().__init__(
            name="polymarket_fpmm_trades",
            description="Backfills Polymarket FPMM (AMM) trades from Polygon blockchain",
        )
        self._from_block = from_block
        self._to_block = to_block
        self._chunk_size = chunk_size
        self._max_workers = max_workers

    def _decode_fpmm_buy(self, log: dict) -> FPMMTrade:
        """Decode an FPMMBuy event log."""
        # Indexed: buyer (topic1), outcomeIndex (topic2)
        # Non-indexed: investmentAmount, feeAmount, outcomeTokensBought
        buyer = Web3.to_checksum_address("0x" + log["topics"][1].hex()[-40:])
        outcome_index = int.from_bytes(log["topics"][2], "big")

        # Decode non-indexed data
        data = bytes(log["data"])
        investment_amount = int.from_bytes(data[0:32], "big")
        fee_amount = int.from_bytes(data[32:64], "big")
        outcome_tokens_bought = int.from_bytes(data[64:96], "big")

        return FPMMTrade(
            block_number=log["blockNumber"],
            transaction_hash=log["transactionHash"].hex(),
            log_index=log["logIndex"],
            fpmm_address=log["address"],
            trader=buyer,
            amount=investment_amount,
            fee_amount=fee_amount,
            outcome_index=outcome_index,
            outcome_tokens=outcome_tokens_bought,
            is_buy=True,
        )

    def _decode_fpmm_sell(self, log: dict) -> FPMMTrade:
        """Decode an FPMMSell event log."""
        # Indexed: seller (topic1), outcomeIndex (topic2)
        # Non-indexed: returnAmount, feeAmount, outcomeTokensSold
        seller = Web3.to_checksum_address("0x" + log["topics"][1].hex()[-40:])
        outcome_index = int.from_bytes(log["topics"][2], "big")

        # Decode non-indexed data
        data = bytes(log["data"])
        return_amount = int.from_bytes(data[0:32], "big")
        fee_amount = int.from_bytes(data[32:64], "big")
        outcome_tokens_sold = int.from_bytes(data[64:96], "big")

        return FPMMTrade(
            block_number=log["blockNumber"],
            transaction_hash=log["transactionHash"].hex(),
            log_index=log["logIndex"],
            fpmm_address=log["address"],
            trader=seller,
            amount=return_amount,
            fee_amount=fee_amount,
            outcome_index=outcome_index,
            outcome_tokens=outcome_tokens_sold,
            is_buy=False,
        )

    def _fetch_logs_with_retry(self, client: PolygonClient, topic: str, from_block: int, to_block: int) -> list[dict]:
        """Fetch logs for a topic, splitting range if too large."""
        try:
            return list(
                client.w3.eth.get_logs(
                    {
                        "topics": [topic],
                        "fromBlock": from_block,
                        "toBlock": to_block,
                    }
                )
            )
        except Exception as e:
            if "too large" in str(e).lower():
                mid = (from_block + to_block) // 2
                left = self._fetch_logs_with_retry(client, topic, from_block, mid)
                right = self._fetch_logs_with_retry(client, topic, mid + 1, to_block)
                return left + right
            raise

    def _fetch_chunk(self, client: PolygonClient, from_block: int, to_block: int) -> tuple[list[FPMMTrade], int, int]:
        """Fetch a single chunk of trades. Used by thread pool."""
        trades: list[FPMMTrade] = []

        try:
            # Fetch buy logs
            buy_logs = self._fetch_logs_with_retry(client, FPMM_BUY_TOPIC, from_block, to_block)
            for log in buy_logs:
                try:
                    trades.append(self._decode_fpmm_buy(log))
                except Exception as e:
                    tqdm.write(f"Error decoding FPMMBuy log: {e}")

            # Fetch sell logs
            sell_logs = self._fetch_logs_with_retry(client, FPMM_SELL_TOPIC, from_block, to_block)
            for log in sell_logs:
                try:
                    trades.append(self._decode_fpmm_sell(log))
                except Exception as e:
                    tqdm.write(f"Error decoding FPMMSell log: {e}")

        except Exception as e:
            tqdm.write(f"Error fetching blocks {from_block}-{to_block}: {e}")

        return trades, from_block, to_block

    def run(self) -> None:
        """Backfill all Polymarket FPMM trades from the Polygon blockchain.

        This fetches FPMMBuy and FPMMSell events from all FPMM contracts
        deployed via the Polymarket FPMM Factory and saves them to parquet files.
        """
        BATCH_SIZE = 10000
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        client = PolygonClient()

        # Determine starting block
        from_block = self._from_block
        if from_block is None:
            if CURSOR_FILE.exists():
                try:
                    from_block = int(CURSOR_FILE.read_text().strip())
                    print(f"Resuming from block {from_block}")
                except (ValueError, TypeError):
                    from_block = FPMM_START_BLOCK
            else:
                from_block = FPMM_START_BLOCK

        to_block = self._to_block
        if to_block is None:
            to_block = client.get_block_number()

        print(f"Fetching FPMM trades from block {from_block} to {to_block}")
        print(f"Total blocks: {to_block - from_block:,}")

        all_trades: list[dict] = []
        total_saved = 0

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

        # Process by block range with parallel fetching
        total_chunks = len(ranges)
        pbar = tqdm(total=total_chunks, desc="Backfilling Legacy", unit=" chunks")
        last_block_processed = from_block

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                for batch_start in range(0, len(ranges), self._max_workers):
                    batch = ranges[batch_start : batch_start + self._max_workers]
                    fetched_at = datetime.utcnow()

                    # Submit all chunks in this batch
                    futures = {
                        executor.submit(self._fetch_chunk, client, start, end): (start, end) for start, end in batch
                    }

                    # Collect results (order doesn't matter for trades)
                    results: dict[tuple[int, int], list[FPMMTrade]] = {}
                    for future in concurrent.futures.as_completed(futures):
                        trades, start, end = future.result()
                        results[(start, end)] = trades

                    # Process results in order for cursor tracking
                    for start, end in batch:
                        trades = results[(start, end)]
                        for trade in trades:
                            trade_dict = asdict(trade)
                            # Convert large ints to strings to avoid parquet overflow
                            trade_dict["amount"] = str(trade_dict["amount"])
                            trade_dict["fee_amount"] = str(trade_dict["fee_amount"])
                            trade_dict["outcome_tokens"] = str(trade_dict["outcome_tokens"])
                            trade_dict["_fetched_at"] = fetched_at
                            all_trades.append(trade_dict)

                        pbar.update(1)
                        last_block_processed = end

                    pbar.set_postfix(
                        block=last_block_processed,
                        buffer=len(all_trades),
                        saved=total_saved,
                    )

                    # Save in batches
                    while len(all_trades) >= BATCH_SIZE:
                        save_batch(all_trades[:BATCH_SIZE])
                        all_trades = all_trades[BATCH_SIZE:]

                    # Save cursor after each parallel batch completes
                    CURSOR_FILE.write_text(str(last_block_processed))

        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved.")
        finally:
            pbar.close()

        # Save remaining trades
        if all_trades:
            save_batch(all_trades)

        if CURSOR_FILE.exists():
            CURSOR_FILE.unlink()

        print(f"\nFPMM backfill complete: {total_saved} trades saved")
