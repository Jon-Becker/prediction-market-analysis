#!/usr/bin/env python3
"""
EXPLORATORY ANALYSIS: POLYMARKET VOLUME & LIQUIDITY PATTERNS
=============================================================

IMPORTANT: This is NOT replication of Kalshi findings.
This is exploratory analysis of what patterns exist in Polymarket data.

Honest research practice:
- Pre-register hypotheses BEFORE looking at results
- Use train/test split (70/30)
- Apply multiple testing correction
- Report what doesn't work, not just what does
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests


def load_polymarket_sample(max_files=20):
    """Load Polymarket data"""
    market_dir = Path.home() / "github/prediction-market-analysis/data/polymarket/markets"
    market_files = sorted(market_dir.glob("markets_*.parquet"))[:max_files]

    markets = []
    for f in market_files:
        df = pd.read_parquet(f)
        markets.append(df)

    df = pd.concat(markets, ignore_index=True)

    # Compute features
    df["log_volume"] = np.log1p(df["volume"].fillna(0))
    df["log_liquidity"] = np.log1p(df["liquidity"].fillna(0))
    df["high_volume"] = df["volume"] >= df["volume"].median()
    df["high_liquidity"] = df["liquidity"] >= df["liquidity"].median()

    # Time features
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["horizon_days"] = (df["end_date"] - df["created_at"]).dt.total_seconds() / 86400
    df["short_horizon"] = df["horizon_days"] < 7

    # Market status
    df["resolved"] = df["closed"] & (~df["active"])

    return df


def exploratory_analysis():
    """Exploratory analysis with proper controls"""

    print("=" * 80)
    print("EXPLORATORY ANALYSIS: POLYMARKET MARKET CHARACTERISTICS")
    print("=" * 80)
    print("\n⚠️  NOT replicating Kalshi—exploring Polymarket structure\n")

    # Load data
    print("Loading data...")
    df = load_polymarket_sample(max_files=20)
    print(f"Loaded {len(df)} markets")

    # Filter to complete data
    complete = df[df["volume"].notna() & df["liquidity"].notna() & df["horizon_days"].notna()].copy()

    print(f"Complete data: {len(complete)} markets ({100 * len(complete) / len(df):.1f}%)")

    # Train/test split BEFORE analysis
    train, test = train_test_split(complete, test_size=0.3, random_state=42)
    print(f"Train/test split: {len(train)}/{len(test)}")

    # Pre-registered exploratory questions
    questions = [
        "Q1: Do short-horizon markets have different volume distributions?",
        "Q2: Is liquidity correlated with volume?",
        "Q3: Do closed markets have higher volume than active ones?",
        "Q4: Does horizon predict market closure?",
    ]

    print("\n" + "=" * 80)
    print("PRE-REGISTERED EXPLORATORY QUESTIONS")
    print("=" * 80)
    for q in questions:
        print(f"  {q}")

    results = {}

    # Q1: Horizon vs Volume
    print("\n" + "=" * 80)
    print("Q1: Horizon vs Volume Distribution")
    print("=" * 80)

    short = train[train["short_horizon"]]
    long = train[~train["short_horizon"]]

    q1_train = {
        "short_mean_log_vol": float(short["log_volume"].mean()),
        "long_mean_log_vol": float(long["log_volume"].mean()),
        "short_median_log_vol": float(short["log_volume"].median()),
        "long_median_log_vol": float(long["log_volume"].median()),
        "n_short": len(short),
        "n_long": len(long),
    }

    # Mann-Whitney U test (non-parametric)
    u_stat, p_val = stats.mannwhitneyu(short["log_volume"], long["log_volume"], alternative="two-sided")
    q1_train["mann_whitney_p"] = float(p_val)
    q1_train["effect_size_rank_biserial"] = float(1 - (2 * u_stat) / (len(short) * len(long)))

    results["q1"] = {"train": q1_train}

    print("\nTraining Set:")
    print(f"  Short horizon (n={len(short)}): mean log_vol = {q1_train['short_mean_log_vol']:.3f}")
    print(f"  Long horizon (n={len(long)}):  mean log_vol = {q1_train['long_mean_log_vol']:.3f}")
    print(f"  Mann-Whitney p = {p_val:.4f}")

    # Validate on test set
    short_test = test[test["short_horizon"]]
    long_test = test[~test["short_horizon"]]

    q1_test = {
        "short_mean_log_vol": float(short_test["log_volume"].mean()),
        "long_mean_log_vol": float(long_test["log_volume"].mean()),
        "replicates": np.sign(q1_train["short_mean_log_vol"] - q1_train["long_mean_log_vol"])
        == np.sign(short_test["log_volume"].mean() - long_test["log_volume"].mean()),
    }

    q1_test["replicates"] = bool(q1_test["replicates"])  # Convert numpy.bool_ to bool
    results["q1"]["test"] = q1_test

    print("\nTest Set:")
    print(f"  Short horizon: mean log_vol = {q1_test['short_mean_log_vol']:.3f}")
    print(f"  Long horizon:  mean log_vol = {q1_test['long_mean_log_vol']:.3f}")
    print(f"  Direction matches train: {q1_test['replicates']}")

    # Q2: Volume-Liquidity Correlation
    print("\n" + "=" * 80)
    print("Q2: Volume-Liquidity Correlation")
    print("=" * 80)

    r_train, p_train = stats.pearsonr(train["log_volume"], train["log_liquidity"])
    r_test, p_test = stats.pearsonr(test["log_volume"], test["log_liquidity"])

    replicates = bool(np.sign(r_train) == np.sign(r_test) and abs(r_train - r_test) < 0.1)
    results["q2"] = {
        "train": {"r": float(r_train), "p": float(p_train), "n": len(train)},
        "test": {"r": float(r_test), "p": float(p_test), "n": len(test)},
        "replicates": replicates,
    }

    print(f"\nTraining: r = {r_train:.4f} (p={p_train:.4e})")
    print(f"Test:     r = {r_test:.4f} (p={p_test:.4e})")
    print(f"Replicates: {results['q2']['replicates']}")

    # Q3: Closed vs Active Volume
    print("\n" + "=" * 80)
    print("Q3: Market Status vs Volume")
    print("=" * 80)

    closed_train = train[train["closed"]]
    active_train = train[~train["closed"]]

    u_stat, p_val = stats.mannwhitneyu(closed_train["log_volume"], active_train["log_volume"])

    results["q3"] = {
        "train": {
            "closed_mean": float(closed_train["log_volume"].mean()),
            "active_mean": float(active_train["log_volume"].mean()),
            "mann_whitney_p": float(p_val),
            "n_closed": len(closed_train),
            "n_active": len(active_train),
        }
    }

    print(f"\nClosed markets (n={len(closed_train)}): mean log_vol = {closed_train['log_volume'].mean():.3f}")
    print(f"Active markets (n={len(active_train)}): mean log_vol = {active_train['log_volume'].mean():.3f}")
    print(f"Mann-Whitney p = {p_val:.4f}")

    # Q4: Horizon predicts closure
    print("\n" + "=" * 80)
    print("Q4: Horizon vs Market Closure")
    print("=" * 80)

    closure_by_horizon = train.groupby("short_horizon")["closed"].mean()

    results["q4"] = {
        "short_closure_rate": float(closure_by_horizon.get(True, 0)),
        "long_closure_rate": float(closure_by_horizon.get(False, 0)),
    }

    print(f"\nShort horizon closure rate: {results['q4']['short_closure_rate']:.3f}")
    print(f"Long horizon closure rate:  {results['q4']['long_closure_rate']:.3f}")

    # Multiple testing correction
    print("\n" + "=" * 80)
    print("MULTIPLE TESTING CORRECTION")
    print("=" * 80)

    p_values = [q1_train["mann_whitney_p"], p_train, results["q3"]["train"]["mann_whitney_p"]]

    reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

    print(f"\nOriginal p-values: {[f'{p:.4f}' for p in p_values]}")
    print(f"Adjusted p-values: {[f'{p:.4f}' for p in p_adj]}")
    print(f"Significant after FDR: {reject}")

    results["multiple_testing"] = {
        "method": "Benjamini-Hochberg FDR",
        "alpha": 0.05,
        "original_p": [float(p) for p in p_values],
        "adjusted_p": [float(p) for p in p_adj],
        "significant": [bool(r) for r in reject],
    }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nPatterns Found (Training Set):")
    print(f"  Q1: Horizon-volume relationship: p={q1_train['mann_whitney_p']:.4f} {'✓' if p_adj[0] < 0.05 else '✗'}")
    print(f"  Q2: Volume-liquidity correlation: r={r_train:.3f}, p={p_train:.4e} {'✓' if p_adj[1] < 0.05 else '✗'}")
    print(
        f"  Q3: Status-volume relationship: p={results['q3']['train']['mann_whitney_p']:.4f} {'✓' if p_adj[2] < 0.05 else '✗'}"
    )

    print("\nHoldout Validation:")
    print(f"  Q1: {'Replicates ✓' if q1_test['replicates'] else 'Fails ✗'}")
    print(f"  Q2: {'Replicates ✓' if results['q2']['replicates'] else 'Fails ✗'}")

    # Export
    output_path = Path.home() / "github/prediction-market-analysis/POLYMARKET_EXPLORATORY_RESULTS.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = exploratory_analysis()
