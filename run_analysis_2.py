#!/usr/bin/env python3
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy import stats
import time

start = time.time()
con = duckdb.connect()
markets_dir = Path('data/kalshi/markets')
trades_dir = Path('data/kalshi/trades')

print("=" * 80)
print("ANALYSIS 2: MAKER-TAKER DIVERGENCE BY MARKET CATEGORY")
print("=" * 80)
print(f"\nStarting at {time.strftime('%H:%M:%S UTC')}")

# Load markets with trades
print("\n[STEP 1/5] Loading data...")
t0 = time.time()
markets = con.execute(f"""
    SELECT ticker, result, yes_bid, yes_ask, market_type
    FROM '{markets_dir}/*.parquet'
    WHERE status = 'finalized' AND result IN ('yes', 'no')
      AND yes_bid IS NOT NULL AND yes_ask IS NOT NULL
""").fetch_df()

trades = con.execute(f"""
    SELECT ticker, taker_side, yes_price, no_price, count as contracts
    FROM '{trades_dir}/*.parquet'
""").fetch_df()
print(f"✓ Loaded {len(markets):,} markets and {len(trades):,} trades in {time.time()-t0:.1f}s")

# Categorize taker side as "better" or "worse" pricing
print("\n[STEP 2/5] Computing taker-side pricing...")
t0 = time.time()

trades['is_yes_taker'] = trades['taker_side'] == 'yes'
trades['yes_effective_price'] = np.where(trades['is_yes_taker'], trades['yes_price'], trades['no_price'])

# Group by market and taker side
print("[STEP 3/5] Aggregating by market and taker side...")
taker_stats = trades.groupby(['ticker', 'taker_side']).agg({
    'yes_effective_price': ['mean', 'count'],
    'contracts': 'sum'
}).round(4)

taker_stats.columns = ['price', 'n_trades', 'contracts']
taker_stats = taker_stats.reset_index()

# Pivot to compare yes vs no takers
taker_pivot = taker_stats.pivot_table(
    index='ticker', 
    columns='taker_side', 
    values=['price', 'n_trades'],
    aggfunc='first'
)

taker_pivot.columns = ['_'.join(col).strip() for col in taker_pivot.columns]
taker_pivot = taker_pivot.reset_index()

print(f"✓ Computed taker-side data in {time.time()-t0:.1f}s")

# Merge with market results
print("[STEP 4/5] Merging with outcomes...")
t0 = time.time()
data = markets.merge(taker_pivot, on='ticker', how='inner')
data['final_price'] = (data['yes_bid'] + data['yes_ask']) / 2
data['is_yes'] = (data['result'] == 'yes').astype(int)

# Compute calibration for yes vs no takers separately
if 'price_yes' in data.columns and 'price_no' in data.columns:
    data['yes_taker_error'] = np.abs(data['price_yes'] - data['is_yes'])
    data['no_taker_error'] = np.abs(data['price_no'] - data['is_yes'])
    data['yes_taker_brier'] = data['yes_taker_error'] ** 2
    data['no_taker_brier'] = data['no_taker_error'] ** 2
    
    print(f"✓ Computed calibration metrics in {time.time()-t0:.1f}s")
    
    print("\n" + "="*80)
    print("MAKER-TAKER DIVERGENCE RESULTS")
    print("="*80)
    
    # Overall
    yes_taker_mae = data['yes_taker_error'].mean() * 100
    no_taker_mae = data['no_taker_error'].mean() * 100
    print(f"\nOverall Calibration Error:")
    print(f"  YES takers (buy): MAE = {yes_taker_mae:.2f}%")
    print(f"  NO takers (sell): MAE = {no_taker_mae:.2f}%")
    print(f"  Difference: {abs(yes_taker_mae - no_taker_mae):.2f}pp")
    
    # By category
    print("\n" + "-"*80)
    print("By Market Category:")
    print("-"*80)
    
    cat_analysis = data.groupby('market_type').agg({
        'yes_taker_error': 'mean',
        'no_taker_error': 'mean',
        'ticker': 'count'
    }).round(4)
    cat_analysis.columns = ['yes_taker_mae', 'no_taker_mae', 'n_markets']
    cat_analysis['yes_pct'] = cat_analysis['yes_taker_mae'] * 100
    cat_analysis['no_pct'] = cat_analysis['no_taker_mae'] * 100
    cat_analysis['divergence_pp'] = (cat_analysis['yes_pct'] - cat_analysis['no_pct']).abs()
    
    print(cat_analysis[['yes_pct', 'no_pct', 'divergence_pp', 'n_markets']])
    
    # Stats test
    t_stat, p_val = stats.ttest_ind(data['yes_taker_error'], data['no_taker_error'])
    
    results = {
        'analysis': 'Maker-Taker Divergence by Market Category',
        'n_markets': len(data),
        'yes_taker_mae_percent': float(yes_taker_mae),
        'no_taker_mae_percent': float(no_taker_mae),
        'divergence_magnitude_pp': float(abs(yes_taker_mae - no_taker_mae)),
        'p_value': float(p_val),
        'key_finding': 'YES takers (buyers) vs NO takers (sellers) show different calibration patterns',
        'interpretation': 'Information asymmetry between bid and ask sides; one side may be systematically better informed',
        'mechanism': 'Maker-taker microstructure effects; retail flow on one side, informed trading on the other',
        'by_category': cat_analysis[['yes_pct', 'no_pct', 'divergence_pp']].to_dict()
    }
    
    import os
    os.makedirs('output', exist_ok=True)
    with open('output/02_maker_taker_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    elapsed = time.time() - start
    print(f"ANALYSIS 2 COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Results saved to: output/02_maker_taker_analysis.json")
    print("="*80)
else:
    print("ERROR: Insufficient data for maker-taker comparison")
