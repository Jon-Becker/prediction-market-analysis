"""Fetch Polymarket trades directly from the Polygon blockchain."""

import concurrent.futures
import os
from collections.abc import Generator
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

load_dotenv()

# Contract addresses
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEGRISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

# OrderFilled event signature
ORDER_FILLED_TOPIC = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"

# ABI for OrderFilled event
ORDER_FILLED_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True, "name": "orderHash", "type": "bytes32"},
        {"indexed": True, "name": "maker", "type": "address"},
        {"indexed": True, "name": "taker", "type": "address"},
        {"indexed": False, "name": "makerAssetId", "type": "uint256"},
        {"indexed": False, "name": "takerAssetId", "type": "uint256"},
        {"indexed": False, "name": "makerAmountFilled", "type": "uint256"},
        {"indexed": False, "name": "takerAmountFilled", "type": "uint256"},
        {"indexed": False, "name": "fee", "type": "uint256"},
    ],
    "name": "OrderFilled",
    "type": "event",
}

# Gnosis ConditionalTokens contract used by Polymarket
CONDITIONAL_TOKENS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# ConditionResolution event signature
# ConditionResolution(bytes32 indexed conditionId, address indexed oracle, bytes32 indexed questionId,
#                     uint256 outcomeSlotCount, uint256[] payoutNumerators)
CONDITION_RESOLUTION_TOPIC = (
    "0x" + Web3.keccak(text="ConditionResolution(bytes32,address,bytes32,uint256,uint256[])").hex()
)

# ABI for ConditionResolution event
CONDITION_RESOLUTION_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True, "name": "conditionId", "type": "bytes32"},
        {"indexed": True, "name": "oracle", "type": "address"},
        {"indexed": True, "name": "questionId", "type": "bytes32"},
        {"indexed": False, "name": "outcomeSlotCount", "type": "uint256"},
        {"indexed": False, "name": "payoutNumerators", "type": "uint256[]"},
    ],
    "name": "ConditionResolution",
    "type": "event",
}

# Public Polygon RPC
POLYGON_RPC = os.getenv("POLYGON_RPC", "")


@dataclass
class BlockchainTrade:
    """A trade decoded from the blockchain."""

    block_number: int
    transaction_hash: str
    log_index: int
    order_hash: str
    maker: str
    taker: str
    maker_asset_id: int
    taker_asset_id: int
    maker_amount: int  # In smallest units (6 decimals for USDC)
    taker_amount: int
    fee: int
    timestamp: Optional[int] = None  # Block timestamp

    @property
    def is_buy(self) -> bool:
        """True if maker is providing USDC (buying outcome tokens)."""
        return self.maker_asset_id == 0

    @property
    def price(self) -> float:
        """Calculate price in USDC per token."""
        if self.is_buy:
            # Maker gives USDC, receives tokens
            if self.taker_amount > 0:
                return self.maker_amount / self.taker_amount
        else:
            # Maker gives tokens, receives USDC
            if self.maker_amount > 0:
                return self.taker_amount / self.maker_amount
        return 0.0

    @property
    def size(self) -> float:
        """Number of tokens traded (in token units, 6 decimals)."""
        if self.is_buy:
            return self.taker_amount / 1e6
        return self.maker_amount / 1e6

    @property
    def side(self) -> str:
        """BUY or SELL from taker's perspective."""
        return "BUY" if self.is_buy else "SELL"

    @property
    def condition_id(self) -> str:
        """Get condition ID from asset ID (first 32 bytes of position ID)."""
        asset_id = self.taker_asset_id if self.is_buy else self.maker_asset_id
        if asset_id == 0:
            return ""
        # The condition ID is embedded in the position ID
        # Position ID = keccak256(collateral, conditionId, partition)
        # We store the full asset ID as a hex string for now
        return hex(asset_id)


@dataclass
class ConditionResolution:
    """A ConditionResolution event decoded from the blockchain."""

    block_number: int
    transaction_hash: str
    log_index: int
    condition_id: str  # 0x-prefixed hex, joins to markets.condition_id
    oracle: str
    question_id: str  # 0x-prefixed hex
    outcome_slot_count: int
    payout_numerators: list[int]

    @property
    def winning_outcome(self) -> Optional[int]:
        """Index of the sole winning outcome, or None for split/invalid payouts."""
        nonzero = [i for i, numerator in enumerate(self.payout_numerators) if numerator != 0]
        if len(nonzero) == 1:
            return nonzero[0]
        return None


class PolygonClient:
    """Client for fetching Polymarket trades from Polygon blockchain."""

    def __init__(self, rpc_url: Optional[str] = None):
        self.rpc_url = rpc_url or POLYGON_RPC
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": 30}))
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        # Create contract instances for decoding
        self.ctf_exchange = self.w3.eth.contract(address=Web3.to_checksum_address(CTF_EXCHANGE), abi=[ORDER_FILLED_ABI])
        self.negrisk_exchange = self.w3.eth.contract(
            address=Web3.to_checksum_address(NEGRISK_CTF_EXCHANGE),
            abi=[ORDER_FILLED_ABI],
        )
        self.conditional_tokens = self.w3.eth.contract(
            address=Web3.to_checksum_address(CONDITIONAL_TOKENS),
            abi=[CONDITION_RESOLUTION_ABI],
        )

    def get_block_number(self) -> int:
        """Get current block number."""
        return self.w3.eth.block_number

    def get_block_timestamp(self, block_number: int) -> int:
        """Get timestamp for a block."""
        block = self.w3.eth.get_block(block_number)
        return block["timestamp"]

    def _decode_order_filled(self, log: dict, contract) -> BlockchainTrade:
        """Decode an OrderFilled event log."""
        decoded = contract.events.OrderFilled().process_log(log)
        args = decoded["args"]

        return BlockchainTrade(
            block_number=log["blockNumber"],
            transaction_hash=log["transactionHash"].hex(),
            log_index=log["logIndex"],
            order_hash=args["orderHash"].hex(),
            maker=args["maker"],
            taker=args["taker"],
            maker_asset_id=args["makerAssetId"],
            taker_asset_id=args["takerAssetId"],
            maker_amount=args["makerAmountFilled"],
            taker_amount=args["takerAmountFilled"],
            fee=args["fee"],
        )

    def get_trades(
        self,
        from_block: int,
        to_block: int,
        contract_address: str = CTF_EXCHANGE,
    ) -> list[BlockchainTrade]:
        """Fetch OrderFilled events from a block range."""
        contract = self.ctf_exchange if contract_address.lower() == CTF_EXCHANGE.lower() else self.negrisk_exchange

        logs = self.w3.eth.get_logs(
            {
                "address": Web3.to_checksum_address(contract_address),
                "topics": [ORDER_FILLED_TOPIC],
                "fromBlock": from_block,
                "toBlock": to_block,
            }
        )

        trades = []
        for log in logs:
            try:
                trade = self._decode_order_filled(log, contract)
                trades.append(trade)
            except Exception as e:
                print(f"Error decoding log: {e}")

        return trades

    def _decode_condition_resolution(self, log: dict) -> ConditionResolution:
        """Decode a ConditionResolution event log."""
        decoded = self.conditional_tokens.events.ConditionResolution().process_log(log)
        args = decoded["args"]

        return ConditionResolution(
            block_number=log["blockNumber"],
            transaction_hash=log["transactionHash"].hex(),
            log_index=log["logIndex"],
            condition_id="0x" + bytes(args["conditionId"]).hex(),
            oracle=args["oracle"],
            question_id="0x" + bytes(args["questionId"]).hex(),
            outcome_slot_count=args["outcomeSlotCount"],
            payout_numerators=list(args["payoutNumerators"]),
        )

    def get_condition_resolutions(self, from_block: int, to_block: int) -> list[ConditionResolution]:
        """Fetch ConditionResolution events from a block range."""
        logs = self.w3.eth.get_logs(
            {
                "address": Web3.to_checksum_address(CONDITIONAL_TOKENS),
                "topics": [CONDITION_RESOLUTION_TOPIC],
                "fromBlock": from_block,
                "toBlock": to_block,
            }
        )

        resolutions = []
        for log in logs:
            try:
                resolution = self._decode_condition_resolution(log)
                resolutions.append(resolution)
            except Exception as e:
                print(f"Error decoding log: {e}")

        return resolutions

    def _fetch_chunk(self, start: int, end: int, contract_address: str) -> tuple[list[BlockchainTrade], int, int]:
        """Fetch a single chunk of trades. Used by thread pool."""
        try:
            trades = self.get_trades(start, end, contract_address)
            return trades, start, end
        except Exception as e:
            if "too large" in str(e).lower():
                # Split into two halves and fetch sequentially
                mid = (start + end) // 2
                t1, _, _ = self._fetch_chunk(start, mid, contract_address)
                t2, _, _ = self._fetch_chunk(mid + 1, end, contract_address)
                return t1 + t2, start, end
            else:
                print(f"Error fetching blocks {start}-{end}: {e}")
                return [], start, end

    def iter_trades(
        self,
        from_block: int,
        to_block: Optional[int] = None,
        chunk_size: int = 1000,
        contract_address: str = CTF_EXCHANGE,
        max_workers: int = 5,
    ) -> Generator[tuple[list[BlockchainTrade], int, int], None, None]:
        """Iterate through trades in chunks of blocks using parallel fetching.

        Args:
            from_block: Starting block number
            to_block: Ending block number (default: latest)
            chunk_size: Number of blocks per query
            contract_address: CTF_EXCHANGE or NEGRISK_CTF_EXCHANGE
            max_workers: Number of parallel threads

        Yields:
            Tuples of (trades, chunk_start, chunk_end)
        """
        if to_block is None:
            to_block = self.get_block_number()

        # Build list of chunk ranges
        ranges = []
        current = from_block
        while current <= to_block:
            end = min(current + chunk_size - 1, to_block)
            ranges.append((current, end))
            current = end + 1

        # Process in batches of max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_start in range(0, len(ranges), max_workers):
                batch = ranges[batch_start : batch_start + max_workers]
                futures = {
                    executor.submit(self._fetch_chunk, start, end, contract_address): (start, end)
                    for start, end in batch
                }

                # Yield results in order
                results = {}
                for future in concurrent.futures.as_completed(futures):
                    trades, start, end = future.result()
                    results[(start, end)] = trades

                for start, end in batch:
                    yield results[(start, end)], start, end


# Polymarket CTF Exchange created at block 33605403
POLYMARKET_START_BLOCK = int(os.getenv("POLYMARKET_START_BLOCK", "33605403"))

# Gnosis ConditionalTokens deployed at block 4023686
CONDITIONAL_TOKENS_START_BLOCK = int(os.getenv("CONDITIONAL_TOKENS_START_BLOCK", "4023686"))


def get_deployment_block() -> int:
    """Get approximate block when Polymarket CTF Exchange was deployed."""
    return POLYMARKET_START_BLOCK
