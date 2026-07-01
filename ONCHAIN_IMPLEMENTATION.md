# Polymarket On-Chain Resolution Implementation Guide

## Overview

This document outlines how to implement Polymarket resolution data fetching from Polygon blockchain. This enables cross-platform replication of Kalshi calibration findings on Polymarket.

## Current State

✅ **Done:**
- Framework: `src/indexers/polymarket/onchain_resolution_fetcher.py` (complete)
- Data pipeline: `ResolutionDataPipeline` (ready for integration)
- Type safety: Async/await with proper error handling
- Caching: Persistent JSON cache for resolution data

❌ **Not Yet Implemented:**
- Polygon RPC connection
- UMA Oracle contract integration
- Event log filtering
- Answer decoding

## Why This Matters

Without on-chain resolutions, we cannot:
- Compute `abs_error = |final_price - outcome|` (required for Kalshi replication)
- Test **H1: Horizon×Volume interaction** (β=-0.064, p<10⁻⁸⁸)
- Test **H2: Spread-Error correlation** (r=0.117, p<10⁻¹⁰⁰)
- Distinguish calibration findings from Polymarket-specific patterns

With on-chain resolutions, we can:
- Replicate Kalshi findings on 200K Polymarket markets
- Cross-validate across platforms
- Publish in Management Science or AER: Insights tier

## Implementation Plan

### Phase 1: Setup (2-4 hours)

#### 1.1 Install Dependencies

```bash
pip install web3==6.11.3 contract-abi-utils eth-typing
```

#### 1.2 Obtain UMA Oracle ABI

Download or construct the ABI for UMA Optimistic Oracle on Polygon:

```python
# From UMA docs: https://github.com/UMAprotocol/protocol/tree/master/packages/core/contracts
UMA_ORACLE_ABI = [
    {
        "type": "event",
        "name": "AssertionResolved",
        "inputs": [
            {"name": "assertionId", "type": "bytes32", "indexed": True},
            {"name": "asserter", "type": "address", "indexed": True},
            {"name": "resolvedAnswer", "type": "bytes", "indexed": False},
            {"name": "resolveBlockNumber", "type": "uint256", "indexed": False},
        ]
    }
]
```

#### 1.3 Obtain CTF Exchange ABI

Download the Conditional Tokens Framework Exchange ABI from:
- https://github.com/gnosis/conditional-tokens-market-maker

### Phase 2: Data Mapping (4-6 hours)

#### 2.1 Map Condition IDs to Markets

Polymarket stores the `condition_id` (CTF identifier) in the market data:

```python
import pandas as pd

df_markets = pd.read_parquet("data/polymarket/markets/*.parquet")
print(df_markets[['id', 'question', 'condition_id']].head())

# condition_id is the CTF condition hash
# Each market has exactly one condition_id
```

#### 2.2 Match with Polymarket Resolution Data

Polymarket resolves markets on-chain via UMA oracle:

```python
# Resolution flow:
# 1. Market ends (end_date passes)
# 2. UMA oracle receives assertion (outcome Yes/No)
# 3. Assertion resolved after dispute window (typically 2 hours)
# 4. CTF settlement uses payout to determine Yes/No winners

# Query UMA AssertionResolved events:
# - Filter by market end_date (start_block ~ end_date block)
# - Extract assertionId → condition_id mapping
# - Get resolvedAnswer (encoded outcome)
# - Decode to binary (0 or 1)
```

### Phase 3: Implementation (6-8 hours)

#### 3.1 Connect to Polygon RPC

```python
from web3 import Web3

# Using public Polygon RPC (rate-limited but free)
w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com/"))

# Better: Use Alchemy or QuickNode (requires free API key)
w3 = Web3(Web3.HTTPProvider(
    f"https://polygon-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}"
))

assert w3.is_connected(), "Failed to connect to Polygon RPC"
print(f"Connected to Polygon. Latest block: {w3.eth.block_number}")
```

#### 3.2 Implement Event Filtering

```python
from web3.contract import Contract

# Load UMA Oracle contract
uma_oracle = w3.eth.contract(
    address="0x...",  # UMA oracle on Polygon
    abi=UMA_ORACLE_ABI
)

# Get all AssertionResolved events
# Strategy: Query in blocks of 10K blocks (RPC limits)
start_block = 50000000  # First Polymarket market block
end_block = w3.eth.block_number

events = []
for from_block in range(start_block, end_block, 10000):
    to_block = min(from_block + 10000, end_block)
    batch = uma_oracle.events.AssertionResolved.get_logs(
        from_block=from_block,
        to_block=to_block
    )
    events.extend(batch)
    print(f"Fetched {len(batch)} events from blocks {from_block}-{to_block}")

print(f"Total: {len(events)} resolution events")
```

#### 3.3 Decode Resolution Answers

```python
def decode_uma_answer(answer_bytes: bytes) -> int:
    """
    Decode UMA oracle answer to binary outcome (0 or 1).
    
    UMA resolves using UMIP-107 or similar:
    - answer = 0: Market resolves to No
    - answer = 1: Market resolves to Yes
    - answer = int(1e18): Market resolves to Yes (with scaling)
    - Custom: See market's price identifier
    """
    # For binary markets:
    answer_int = int.from_bytes(answer_bytes, byteorder='big')
    
    # Convert to 0/1 (handle both scaled and unscaled)
    if answer_int == 0:
        return 0
    elif answer_int > 0:
        return 1
    else:
        raise ValueError(f"Invalid UMA answer: {answer_int}")

# Example:
answer_bytes = bytes.fromhex("0x..." )
outcome = decode_uma_answer(answer_bytes)
print(f"Outcome: {'Yes' if outcome else 'No'}")
```

#### 3.4 Map AssertionId to Condition ID

```python
# Challenge: AssertionId (from UMA) ≠ Condition ID (from CTF)
# Solution: Use market's metadata to link

# Polymarket stores: {market_id, condition_id, price_identifier}
# UMA stores: {assertionId, price_identifier, asserter, resolved_answer}

# Link via price_identifier + assertion_id pattern:
def link_assertion_to_condition(
    assertion_id: str,
    markets_df: pd.DataFrame,
    uma_events: list
) -> str:
    """Find condition_id matching a given assertion_id."""
    # Approach 1: Direct mapping via market metadata
    # Approach 2: Look up in Polymarket subgraph
    # Approach 3: Query CTF conditional_tokens_market_maker contract
    
    # TODO: Implement based on Polymarket's internals
    pass
```

### Phase 4: Integration (4-6 hours)

#### 4.1 Update `onchain_resolution_fetcher.py`

Replace stub `_fetch_from_uma_oracle()` with actual implementation:

```python
async def _fetch_from_uma_oracle(
    self,
    condition_ids: List[str],
    start_block: Optional[int],
    end_block: Optional[int],
) -> List[ResolutionEvent]:
    """Fetch actual resolution data from UMA Oracle."""
    
    resolutions = []
    
    # 1. Query UMA Oracle events
    events = self.uma_contract.events.AssertionResolved.get_logs(
        from_block=start_block or 0,
        to_block=end_block or self.w3.eth.block_number
    )
    
    # 2. Decode events to resolutions
    for event in events:
        assertion_id = event['args']['assertionId']
        resolved_answer = event['args']['resolvedAnswer']
        block_number = event['blockNumber']
        
        # Decode answer
        outcome_index = decode_uma_answer(resolved_answer)
        
        # Map to condition ID
        condition_id = link_assertion_to_condition(assertion_id, condition_ids)
        if not condition_id:
            continue
        
        # Get block timestamp
        block = self.w3.eth.get_block(block_number)
        timestamp = datetime.fromtimestamp(block['timestamp'])
        
        # Create resolution event
        resolution = ResolutionEvent(
            condition_id=condition_id,
            outcome_index=outcome_index,
            payout_numerators=[1 - outcome_index, outcome_index],
            resolution_block=block_number,
            resolution_timestamp=timestamp,
            uma_answer=hex(int.from_bytes(resolved_answer, 'big')),
        )
        
        resolutions.append(resolution)
        self._cache[condition_id] = resolution
    
    return resolutions
```

#### 4.2 Create Integration Script

```python
# src/analysis/polymarket/fetch_resolutions_main.py

import asyncio
import pandas as pd
from src.indexers.polymarket.onchain_resolution_fetcher import ResolutionDataPipeline

async def main():
    # Load Polymarket data
    df_markets = pd.read_parquet("data/polymarket/markets/*.parquet")
    df_trades = pd.read_parquet("data/polymarket/trades/*.parquet")
    
    print(f"Loaded {len(df_markets)} markets, {len(df_trades)} trades")
    
    # Create pipeline
    pipeline = ResolutionDataPipeline(
        polymarket_markets_df=df_markets,
        polymarket_trades_df=df_trades,
        polygon_rpc="https://polygon-rpc.com/",
        cache_file="data/polymarket/resolutions_cache.json"
    )
    
    # Fetch and enrich
    enriched_df = await pipeline.fetch_and_enrich()
    
    # Save enriched data
    enriched_df.to_parquet("data/polymarket/markets_with_resolutions.parquet")
    
    print(f"Enriched dataframe saved. Shape: {enriched_df.shape}")
    print(f"Coverage: {enriched_df['outcome'].notna().sum()} resolved markets")
    
    # Compute error statistics
    enriched_df['abs_error'] = abs(enriched_df['final_price'] - enriched_df['outcome'])
    print(f"\nCalibration Statistics:")
    print(f"Mean abs_error: {enriched_df['abs_error'].mean():.4f}")
    print(f"Median abs_error: {enriched_df['abs_error'].median():.4f}")
    print(f"Std dev: {enriched_df['abs_error'].std():.4f}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Phase 5: Validation (2-4 hours)

#### 5.1 Sanity Checks

```python
# After fetching resolutions:

# 1. Check outcome distribution (should be ~50% Yes, 50% No)
assert enriched_df['outcome'].value_counts().to_dict() \
    ~= {0: len(enriched_df)//2, 1: len(enriched_df)//2}

# 2. Check error range (should be 0-1 for binary)
assert enriched_df['abs_error'].min() >= 0
assert enriched_df['abs_error'].max() <= 1

# 3. Compare to Kalshi baseline
print("Kalshi baseline: MAE 8.36%")
print(f"Polymarket: MAE {enriched_df['abs_error'].mean() * 100:.2f}%")
```

#### 5.2 Run Kalshi Replication Tests

Once on-chain data is available:

```python
# Use existing rigorous_cross_platform_analysis.py
# Update to use enriched Polymarket dataframe
# Run H1, H2 replication with proper train/test splits

python rigorous_cross_platform_analysis.py \
    --kalshi-data data/kalshi/resolved_markets.parquet \
    --polymarket-data data/polymarket/markets_with_resolutions.parquet \
    --output results_cross_platform_final.json
```

## Timeline & Effort

| Phase | Task | Time | Status |
|-------|------|------|--------|
| 1. Setup | Dependencies + ABIs | 2-4h | Not started |
| 2. Mapping | Condition ID linking | 4-6h | Not started |
| 3. Implementation | ORM events + decoding | 6-8h | Not started |
| 4. Integration | Full pipeline + script | 4-6h | Not started |
| 5. Validation | Sanity checks + replication | 2-4h | Not started |
| **Total** | **Full system** | **18-28h** | **Pending** |

**Estimated completion:** 2-4 days of focused work

## Resources

- **UMA Oracle:** https://github.com/UMAprotocol/protocol
- **CTF:** https://github.com/gnosis/conditional-tokens-framework
- **Polymarket:** https://polymarket.com/docs
- **Polygon RPC:** https://polygon-rpc.com/ (public, rate-limited)
- **Alchemy API:** https://www.alchemy.com/ (free tier, 300M compute units/month)

## Next Steps

1. Request GitHub issues for each phase
2. Allocate Paul/dev to blockchain integration
3. Set up Alchemy account for Polygon RPC access
4. Begin Phase 1 setup
5. Weekly check-ins on progress

---

**Document created:** July 1, 2026  
**Framework status:** Complete (async/await, caching, type safety)  
**Implementation status:** Pending blockchain integration  
**Estimated delivery:** Production-ready within 2-4 days of focused work
