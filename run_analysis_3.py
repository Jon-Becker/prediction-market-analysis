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
print("ANALYSIS 3: TIME-TO-RESOLUTION CALIBRATION DECAY")
print("=" * 80)
print(f"\nStarting at {time.strftime('%H:%M:%S UTC')}")

# Load markets  
print("\n[STEP 1/5] Loading finalized markets...")
t0 = time.time()
markets = con.execute(f"""
    SELECT 
        ticker, 
        result, 
        yes_bid, 
        yes_ask, 
        market_type,
        close_time,
        _fetched_at
    FROM '{markets_dir}/*.parquet'
    WHERE status = 'finalized' 
      AND result IN ('yes', 'no')
      AND yes_bid IS NOT NULL 
      AND yes_ask IS NOT NULL
      AND close_time IS NOT NULL
""").fetch_df()

# Convert to datetime and normalize timezone
markets['close_time'] = pd.to_datetime(markets['close_time'])
markets['_fetched_at'] = pd.to_datetime(markets['_fetched_at'])

# If already tz-aware, convert to UTC; if naive, assume UTC
markets['close_time'] = markets['close_time'].dt.tz_convert('UTC') if markets['close_time'].dt.tz is not None else markets['close_time'].dt.tz_localize('UTC')
markets['_fetched_at'] = markets['_fetched_at'].dt.tz_convert('UTC') if markets['_fetched_at'].dt.tz is not None else markets['_fetched_at'].dt.tz_localize('UTC')

# Time until resolution (days)
markets['days_to_close'] = (markets['close_time'] - markets['_fetched_at']).dt.total_seconds() / (24 * 3600)

print(f"✓ Loaded {len(markets):,} markets in {time.time()-t0:.1f}s")

# Only keep markets where snapshot is before close
markets = markets[markets['days_to_close'] >= 0]
print(f"✓ Filtered to {len(markets):,} markets with pre-close snapshots")

print("\n[STEP 2/5] Computing calibration metrics...")
t0 = time.time()

markets['final_price'] = (markets['yes_bid'] + markets['yes_ask']) / 2
markets['is_yes'] = (markets['result'] == 'yes').astype(int)
markets['mae'] = np.abs(markets['final_price'] - markets['is_yes'])
markets['brier'] = markets['mae'] ** 2

print(f"✓ Computed in {time.time()-t0:.1f}s")

print("\n[STEP 3/5] Binning by time to resolution...")
t0 = time.time()

# Create time bins: days to resolution
time_bins = [0, 1, 7, 14, 30, 60, 90, 180, 365, 10000]
markets['time_bin'] = pd.cut(markets['days_to_close'], bins=time_bins, right=True)

time_analysis = markets.groupby('time_bin', observed=True).agg({
    'mae': ['mean', 'std', 'count'],
    'brier': 'mean',
    'days_to_close': ['min', 'max']
})

time_analysis.columns = ['_'.join(col) for col in time_analysis.columns]
time_analysis['mae_pct'] = time_analysis['mae_mean'] * 100

print(f"✓ Binning complete in {time.time()-t0:.1f}s\n")

print("="*80)
print("TIME-TO-RESOLUTION CALIBRATION ANALYSIS")
print("="*80)
print(f"\n{'Days to Close':>20} | {'MAE %':>7} | {'Brier':>8} | {'n':>10}")
print("-"*80)

for idx, row in time_analysis.iterrows():
    dmin = int(row['days_to_close_min'])
    dmax = int(row['days_to_close_max'])
    print(f"{dmin:>6d}-{dmax:>10d} | {row['mae_pct']:>7.2f} | {row['brier_mean']:>8.4f} | {int(row['mae_count']):>10,d}")

# Early vs late markets
early = markets[markets['days_to_close'] >= 30]
late = markets[markets['days_to_close'] < 7]

print("\n" + "="*80)
print("EARLY VS LATE MARKET COMPARISON")
print("="*80)
print(f"Early markets (30+ days):  MAE = {early['mae'].mean()*100:6.2f}% | Brier = {early['brier'].mean():.4f} | n = {len(early):>10,d}")
print(f"Late markets (<7 days):    MAE = {late['mae'].mean()*100:6.2f}% | Brier = {late['brier'].mean():.4f} | n = {len(late):>10,d}")

improvement = early['mae'].mean() / late['mae'].mean()
print(f"Improvement factor (early vs late): {improvement:.2f}x")

# Stats
t_stat, p_val = stats.ttest_ind(early['mae'], late['mae'])
print(f"T-test p-value: {p_val:.2e}")

# By category
print("\n" + "="*80)
print("TIME-TO-RESOLUTION EFFECT BY CATEGORY")
print("="*80)

cat_time = markets.groupby('market_type').agg({
    'days_to_close': ['mean', 'median'],
    'mae': 'mean',
    'ticker': 'count'
}).round(3)

cat_time.columns = ['mean_days', 'median_days', 'mae', 'n']
cat_time['mae_pct'] = cat_time['mae'] * 100
cat_time = cat_time.sort_values('median_days')
print(cat_time[['median_days', 'mae_pct', 'n']])

# Save
results = {
    'analysis': 'Time-to-Resolution Calibration Decay',
    'n_markets': len(markets),
    'early_markets_30plus_days_mae_percent': float(early['mae'].mean() * 100),
    'late_markets_under_7_days_mae_percent': float(late['mae'].mean() * 100),
    'improvement_factor': float(improvement),
    'p_value': float(p_val),
    'key_finding': 'Calibration improves as markets approach resolution (MAE decreases over time)',
    'interpretation': 'Information gradually incorporates into prices; uncertainty resolves as outcome becomes clearer',
    'mechanism': 'Real-time information arrival + market learning + selection effects (hard-to-predict markets resolve slower)',
    'confound': 'Cannot disentangle information discovery from compositional effects (fast-resolving markets may be inherently more predictable)',
    'note': 'This analysis documents correlation; within-market panel analysis needed for causal inference'
}

import os
os.makedirs('output', exist_ok=True)
with open('output/03_time_resolution_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

elapsed = time.time() - start
print(f"\n{'='*80}")
print(f"ANALYSIS 3 COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
print(f"{'='*80}")
print(f"Results saved to: output/03_time_resolution_analysis.json")
