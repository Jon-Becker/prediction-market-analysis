"""
Polymarket On-Chain Resolution Fetcher

Fetches resolution outcomes for Polymarket markets by querying:
1. UMA oracle on Polygon (primary resolution source)
2. CTF (Conditional Tokens Framework) settlement events
3. AMM prices at resolution block

Usage:
    from src.indexers.polymarket.onchain_resolution_fetcher import OnChainResolutionFetcher
    
    fetcher = OnChainResolutionFetcher(
        polygon_rpc="https://polygon-rpc.com/",
        uma_oracle_address="0x..."
    )
    
    resolutions = fetcher.fetch_resolutions(
        condition_ids=["0xabc...", "0xdef..."],
        start_block=50000000,
        end_block=51000000
    )
    
    for res in resolutions:
        print(f"Market: {res['condition_id']}")
        print(f"Outcome: {res['outcome_index']} (0=No, 1=Yes)")
        print(f"Payout: {res['payout_numerators']}")
        print(f"Resolution Block: {res['block_number']}")
        print(f"Resolution Time: {res['timestamp']}")
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class ResolutionEvent:
    """On-chain resolution event for a Polymarket market"""
    condition_id: str  # UniqueIdentifier for CTF condition
    outcome_index: int  # 0=No, 1=Yes (for binary markets)
    payout_numerators: List[int]  # Raw payout ratios [for_no, for_yes]
    resolution_block: int
    resolution_timestamp: datetime
    uma_answer: Optional[str] = None  # Raw UMA oracle answer (if available)
    ctf_settlement_tx: Optional[str] = None  # Settlement transaction hash


class OnChainResolutionFetcher:
    """
    Fetches Polymarket resolution data from on-chain sources.
    
    Polymarket uses:
    - UMA Optimistic Oracle for market resolution (on-chain assertion)
    - CTF (Conditional Tokens Framework) for settlement
    - Polygon as L2 settlement layer
    
    This class reconstructs the resolution timeline and outcome.
    """
    
    def __init__(
        self,
        polygon_rpc: str = "https://polygon-rpc.com/",
        uma_oracle_address: Optional[str] = None,
        ctf_exchange_address: Optional[str] = None,
        cache_file: Optional[str] = None,
    ):
        """
        Initialize the fetcher.
        
        Args:
            polygon_rpc: Polygon RPC endpoint URL
            uma_oracle_address: Address of UMA Optimistic Oracle on Polygon
            ctf_exchange_address: Address of CTF Exchange on Polygon
            cache_file: Optional path to cache resolution data (JSON)
        """
        self.polygon_rpc = polygon_rpc
        self.uma_oracle_address = uma_oracle_address or "0x..."  # Placeholder
        self.ctf_exchange_address = ctf_exchange_address or "0x..."  # Placeholder
        self.cache_file = cache_file
        self._cache: Dict[str, ResolutionEvent] = {}
        
        if cache_file:
            self._load_cache()
    
    def _load_cache(self):
        """Load cached resolutions from disk."""
        if not self.cache_file:
            return
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                for cond_id, event_dict in data.items():
                    self._cache[cond_id] = ResolutionEvent(
                        condition_id=cond_id,
                        outcome_index=event_dict['outcome_index'],
                        payout_numerators=event_dict['payout_numerators'],
                        resolution_block=event_dict['resolution_block'],
                        resolution_timestamp=datetime.fromisoformat(
                            event_dict['resolution_timestamp']
                        ),
                        uma_answer=event_dict.get('uma_answer'),
                        ctf_settlement_tx=event_dict.get('ctf_settlement_tx'),
                    )
            logger.info(f"Loaded {len(self._cache)} cached resolutions from {self.cache_file}")
        except FileNotFoundError:
            logger.info(f"No cache file found at {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save cached resolutions to disk."""
        if not self.cache_file:
            return
        try:
            cache_dict = {
                cond_id: {
                    'condition_id': cond_id,
                    'outcome_index': event.outcome_index,
                    'payout_numerators': event.payout_numerators,
                    'resolution_block': event.resolution_block,
                    'resolution_timestamp': event.resolution_timestamp.isoformat(),
                    'uma_answer': event.uma_answer,
                    'ctf_settlement_tx': event.ctf_settlement_tx,
                }
                for cond_id, event in self._cache.items()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_dict, f, indent=2)
            logger.info(f"Saved {len(self._cache)} resolutions to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    async def fetch_resolutions(
        self,
        condition_ids: List[str],
        start_block: Optional[int] = None,
        end_block: Optional[int] = None,
    ) -> List[ResolutionEvent]:
        """
        Fetch resolution events for given condition IDs.
        
        Args:
            condition_ids: List of condition IDs (hex strings, with '0x' prefix)
            start_block: Optional start block for filtering
            end_block: Optional end block for filtering
        
        Returns:
            List of ResolutionEvent objects
        
        Implementation strategy:
        1. Check cache first (fast path)
        2. Query UMA oracle for resolution assertions
        3. Extract outcome from payout structure
        4. Query CTF for settlement events (optional verification)
        5. Return resolution data
        """
        results = []
        uncached_ids = []
        
        # Check cache
        for cond_id in condition_ids:
            if cond_id in self._cache:
                results.append(self._cache[cond_id])
            else:
                uncached_ids.append(cond_id)
        
        if not uncached_ids:
            logger.info(f"All {len(condition_ids)} resolutions found in cache")
            return results
        
        logger.info(f"Fetching {len(uncached_ids)} uncached resolutions from Polygon")
        
        # TODO: Implement actual on-chain fetching
        # For now, this is a stub that returns the framework
        new_resolutions = await self._fetch_from_uma_oracle(
            uncached_ids, start_block, end_block
        )
        
        results.extend(new_resolutions)
        self._save_cache()
        
        return results
    
    async def _fetch_from_uma_oracle(
        self,
        condition_ids: List[str],
        start_block: Optional[int],
        end_block: Optional[int],
    ) -> List[ResolutionEvent]:
        """
        Fetch resolution assertions from UMA Optimistic Oracle.
        
        Query AssertionResolved events to find:
        - Assertion ID that maps to condition_id
        - Resolved answer
        - Block number and timestamp
        
        Implementation requires:
        - Web3 connection to Polygon RPC
        - Contract ABI for UMA oracle
        - Event filter for AssertionResolved
        """
        # Pseudocode:
        # filter = uma_contract.events.AssertionResolved.create_filter(
        #     from_block=start_block,
        #     to_block=end_block
        # )
        # events = filter.get_all_entries()
        # 
        # resolutions = []
        # for event in events:
        #     assertion_id = event['args']['assertionId']
        #     answer = event['args']['resolvedAnswer']
        #     block = event['blockNumber']
        #     timestamp = w3.eth.get_block(block)['timestamp']
        #     
        #     # Decode answer to outcome
        #     outcome_index = decode_uma_answer(answer)
        #     
        #     resolution = ResolutionEvent(
        #         condition_id=...,  # Map from assertion_id
        #         outcome_index=outcome_index,
        #         payout_numerators=[1 - outcome_index, outcome_index],
        #         resolution_block=block,
        #         resolution_timestamp=datetime.fromtimestamp(timestamp),
        #         uma_answer=hex(answer),
        #     )
        #     resolutions.append(resolution)
        #     self._cache[resolution.condition_id] = resolution
        # 
        # return resolutions
        
        logger.warning(
            "On-chain resolution fetcher not yet implemented. "
            "Requires web3.py + Polygon RPC connection + UMA oracle ABI. "
            "See ONCHAIN_IMPLEMENTATION.md for setup."
        )
        return []
    
    def get_final_price(
        self,
        resolution: ResolutionEvent,
        collateral_decimals: int = 6,
    ) -> float:
        """
        Convert resolution event to final market price.
        
        For binary markets:
        - No wins (outcome 0): price = 0.0
        - Yes wins (outcome 1): price = 1.0
        - Ambiguous (split payout): price = payout[1] / (payout[0] + payout[1])
        
        Args:
            resolution: ResolutionEvent from on-chain
            collateral_decimals: Decimals for collateral (USDC = 6)
        
        Returns:
            Final market price (0.0 to 1.0)
        """
        numerators = resolution.payout_numerators
        total = sum(numerators)
        
        if total == 0:
            logger.warning(f"Zero total payout for {resolution.condition_id}")
            return 0.5
        
        # YesToken final price = (yes_payout) / (no_payout + yes_payout)
        if len(numerators) >= 2:
            final_price = numerators[1] / total
        else:
            final_price = 0.0
        
        return final_price


class ResolutionDataPipeline:
    """
    End-to-end pipeline for enriching Polymarket data with on-chain resolutions.
    
    Usage:
        pipeline = ResolutionDataPipeline(
            polymarket_markets_df=df_markets,
            polymarket_trades_df=df_trades,
            polygon_rpc="https://polygon-rpc.com/"
        )
        
        # Fetch all resolutions
        enriched_df = pipeline.fetch_and_enrich()
        
        # Each market now has:
        # - outcome (0 or 1)
        # - resolution_block
        # - resolution_time
        # - final_price (from CTF settlement)
        # - abs_error = |final_price - outcome|
    """
    
    def __init__(
        self,
        polymarket_markets_df,
        polymarket_trades_df,
        polygon_rpc: str = "https://polygon-rpc.com/",
        cache_file: Optional[str] = None,
    ):
        self.markets_df = polymarket_markets_df
        self.trades_df = polymarket_trades_df
        self.fetcher = OnChainResolutionFetcher(
            polygon_rpc=polygon_rpc,
            cache_file=cache_file,
        )
    
    async def fetch_and_enrich(self):
        """Fetch resolutions and add to markets dataframe."""
        # Extract condition IDs from market data
        condition_ids = self.markets_df['condition_id'].unique().tolist()
        
        logger.info(f"Fetching resolutions for {len(condition_ids)} markets...")
        resolutions = await self.fetcher.fetch_resolutions(condition_ids)
        
        # Create lookup
        resolution_lookup = {r.condition_id: r for r in resolutions}
        
        # Enrich markets dataframe
        self.markets_df['outcome'] = self.markets_df['condition_id'].map(
            lambda cid: resolution_lookup[cid].outcome_index 
            if cid in resolution_lookup else None
        )
        self.markets_df['resolution_block'] = self.markets_df['condition_id'].map(
            lambda cid: resolution_lookup[cid].resolution_block 
            if cid in resolution_lookup else None
        )
        self.markets_df['resolution_timestamp'] = self.markets_df['condition_id'].map(
            lambda cid: resolution_lookup[cid].resolution_timestamp 
            if cid in resolution_lookup else None
        )
        
        # Compute final prices and errors
        self.markets_df['final_price'] = self.markets_df['condition_id'].map(
            lambda cid: self.fetcher.get_final_price(resolution_lookup[cid])
            if cid in resolution_lookup else None
        )
        
        # CRITICAL: final_price is the market's closing price at resolution
        # For binary markets: final_price is 0.0 or 1.0 (or split, e.g., 0.5)
        self.markets_df['abs_error'] = abs(
            self.markets_df['final_price'] - self.markets_df['outcome']
        )
        
        logger.info(
            f"Enriched {len(self.markets_df)} markets with on-chain resolutions"
        )
        
        return self.markets_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(__doc__)
    print("\nImplementation status: Framework complete, on-chain integration pending")
    print("See ONCHAIN_IMPLEMENTATION.md for setup instructions")
