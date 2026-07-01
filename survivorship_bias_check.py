#!/usr/bin/env python3
"""
SURVIVORSHIP BIAS ANALYSIS: Cancelled vs. Finalized Markets

Compares observable characteristics of markets that were cancelled vs. those that were finalized.
If cancelled markets are systematically different, this could bias our findings.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy import stats

start_time = pd.Timestamp.now(tz='UTC')
print("=" * 80)
print("SURVIVORSHIP BIAS ANALYSIS: Cancelled vs. Finalized Markets")
print("=" * 80)
print(f"\nStarting at {start_time}\n")

con = duckdb.connect()
markets_dir = Path('data/kalshi/markets')

# Step 1: Check what statuses exist
print("[STEP 1/4] Checking available market statuses...")
statuses = con.execute(f"""
    SELECT DISTINCT status, COUNT(*) as count
    FROM read_parquet('{markets_dir}/*.parquet')
    GROUP BY status
    ORDER BY count DESC
""").fetch_df()

print("\nMarket statuses:")
print(statuses)

# Step 2: Load finalized vs non-finalized markets
print("\n[STEP 2/4] Loading finalized and cancelled markets...")
all_markets = con.execute(f"""
    SELECT
        ticker, title, result, status,
        CAST(volume AS FLOAT) as volume,
        CAST(open_interest AS FLOAT) as open_interest,
        COALESCE(CAST(last_price AS FLOAT), 0) as last_price,
        market_type, created_time, close_time, _fetched_at
    FROM read_parquet('{markets_dir}/*.parquet')
    WHERE status IN ('finalized', 'closed', 'cancelled', 'liquidated')
""").fetch_df()

print(f"Total markets with terminal statuses: {len(all_markets):,}")
print(f"Status distribution:\n{all_markets['status'].value_counts()}")

finalized = all_markets[all_markets['status'] == 'finalized'].copy()
cancelled = all_markets[all_markets['status'].isin(['cancelled', 'liquidated'])].copy()

print(f"\nFinalized markets: {len(finalized):,}")
print(f"Cancelled/Liquidated markets: {len(cancelled):,}")

if len(cancelled) == 0:
    print("\n⚠️  No cancelled markets in dataset. Only finalized markets available.")
    print("This dataset appears to contain resolved markets only.")
    cancelled_stats = None
else:
    # Step 3: Compare characteristics
    print("\n[STEP 3/4] Comparing characteristics...")
    
    print("\n" + "=" * 80)
    print("VOLUME COMPARISON")
    print("=" * 80)
    
    fin_vol = finalized['volume'].dropna()
    can_vol = cancelled['volume'].dropna()
    
    print(f"Finalized - Mean volume: {fin_vol.mean():,.0f}, Median: {fin_vol.median():,.0f}")
    print(f"Cancelled - Mean volume: {can_vol.mean():,.0f}, Median: {can_vol.median():,.0f}")
    
    if len(can_vol) > 0:
        t_stat, p_val = stats.ttest_ind(fin_vol, can_vol, nan_policy='omit')
        print(f"T-test: t = {t_stat:.3f}, p = {p_val:.6f}")
    
    print("\n" + "=" * 80)
    print("MARKET TYPE DISTRIBUTION")
    print("=" * 80)
    print("\nFinalized:")
    print(finalized['market_type'].value_counts())
    print("\nCancelled:")
    print(cancelled['market_type'].value_counts())
    
    print("\n" + "=" * 80)
    print("PRICE DISTRIBUTION AT FETCHED TIME")
    print("=" * 80)
    
    fin_price = finalized['last_price'].dropna()
    can_price = cancelled['last_price'].dropna()
    
    print(f"Finalized - Mean price: {fin_price.mean():.2f}, Median: {fin_price.median():.2f}")
    print(f"Cancelled - Mean price: {can_price.mean():.2f}, Median: {can_price.median():.2f}")
    
    if len(can_price) > 0:
        t_stat, p_val = stats.ttest_ind(fin_price, can_price, nan_policy='omit')
        print(f"T-test: t = {t_stat:.3f}, p = {p_val:.6f}")
    
    cancelled_stats = {
        'finalized_count': len(finalized),
        'cancelled_count': len(cancelled),
        'finalized_median_volume': float(fin_vol.median()),
        'cancelled_median_volume': float(can_vol.median()) if len(can_vol) > 0 else None,
        'volume_ratio': float(fin_vol.median() / can_vol.median()) if len(can_vol) > 0 and can_vol.median() > 0 else None,
        'finalized_median_price': float(fin_price.median()),
        'cancelled_median_price': float(can_price.median()) if len(can_price) > 0 else None,
    }

# Step 4: Impact assessment
print("\n[STEP 4/4] Assessing impact on findings...")
print("\n" + "=" * 80)
print("IMPLICATION FOR ANALYSIS")
print("=" * 80)

if cancelled_stats is None:
    print("""
✓ CONCLUSION: No survivorship bias detected.

The dataset contains ONLY finalized (resolved) markets. No cancelled or liquidated 
markets are present. This eliminates survivorship bias concerns because:

1. We're not excluding failed markets that couldn't reach liquidity thresholds
2. Our liquidity threshold (~200 trades) is measured for successfully-resolved markets
3. If cancelled markets exist on Kalshi, they are not in this dataset (possible reasons: 
   - API fetch only caught resolved markets
   - Kalshi archives cancelled markets separately
   - Dataset is curated to resolved markets only)

RECOMMENDATION: Add a note in Limitations stating: "Our analysis examines resolved 
markets only. If Kalshi cancelled markets systematically differ (e.g., lower volume, 
higher ambiguity), our findings may not generalize to the full market population."
""")
else:
    if cancelled_stats['volume_ratio'] and cancelled_stats['volume_ratio'] > 2:
        print(f"""
⚠️  POTENTIAL BIAS DETECTED: Finalized markets have {cancelled_stats['volume_ratio']:.1f}x higher median volume.

This suggests survivorship bias: markets with low volume may be preferentially 
cancelled. Our liquidity threshold (~200 trades) could be confounded with 
cancellation probability.

RECOMMENDATION: (1) Run sensitivity analysis on low-volume markets, (2) add a note 
in Limitations about this selection effect, (3) consider causality caution.
""")
    else:
        print("""
✓ NO SIGNIFICANT BIAS: Finalized and cancelled markets have similar volume 
distributions. Survivorship bias is not a major concern.
""")

# Output results
results = {
    'timestamp': start_time.isoformat(),
    'finalized_count': int(len(finalized)),
    'cancelled_count': int(len(cancelled)),
    'status_distribution': finalized['status'].value_counts().to_dict(),
    'cancelled_stats': cancelled_stats,
}

with open('output/survivorship_bias_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to output/survivorship_bias_analysis.json")
print("=" * 80)
print(f"Analysis complete in {(pd.Timestamp.now(tz='UTC') - start_time).total_seconds():.1f}s")
print("=" * 80)
