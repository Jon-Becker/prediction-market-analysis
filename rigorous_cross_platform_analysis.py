#!/usr/bin/env python3
"""
RIGOROUS CROSS-PLATFORM VALIDATION: KALSHI → POLYMARKET
========================================================

Scientific Methodology:
1. Load Kalshi findings that survived adversarial review
2. Pre-register replication hypotheses for Polymarket
3. Use 70/30 train/test split on Polymarket data
4. Test on training set with robustness checks
5. Validate on holdout test set
6. Document what replicates vs. what's platform-specific

Salvaged Kalshi Findings:
- H1: Horizon×Volume interaction (β=-0.064, p<10⁻⁸⁸, but selection confounded)
- H2: Spread correlation with error (r=0.117, real, but low R²=0.015)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ============================================================================
# LOAD KALSHI ANALYSIS RESULTS
# ============================================================================


def load_kalshi_baseline():
    """Load Kalshi analysis results to establish baseline for comparison"""

    print("=" * 80)
    print("KALSHI BASELINE FINDINGS (from adversarial review)")
    print("=" * 80)

    kalshi_findings = {
        "H1_horizon_volume": {
            "description": "Short-horizon markets benefit more from volume",
            "coefficient": -0.064,
            "p_value": 1e-88,
            "effect_size_cohens_d": 0.210,
            "concerns": [
                "Confounded by event type (sports vs non-sports)",
                "Baseline incomparability (12% vs 1% MAE across horizons)",
                "7-30 day reversal unexplained",
                "No category controls in original regression",
            ],
            "verdict": "Real pattern but selection bias makes causal interpretation suspect",
        },
        "H2_spread_error": {
            "description": "Bid-ask spread correlates with market error",
            "correlation": 0.117,
            "p_value": 1e-100,
            "r_squared": 0.0151,
            "concerns": [
                "Explains only 1.5% of variance",
                "91% mediation claim was mathematical error",
                "Actual mediation likely 40-60%",
                "Practical significance unclear",
            ],
            "verdict": "Real correlation but weak predictive power",
        },
    }

    for key, finding in kalshi_findings.items():
        print(f"\n{key}: {finding['description']}")
        print(f"  Status: {finding['verdict']}")
        print("  Concerns:")
        for concern in finding["concerns"]:
            print(f"    - {concern}")

    return kalshi_findings


# ============================================================================
# POLYMARKET DATA LOADER WITH PROPER STRUCTURE
# ============================================================================


def load_polymarket_markets(max_files=20):
    """Load Polymarket market data"""

    print(f"\nLoading Polymarket markets (max {max_files} files)...")
    market_dir = Path.home() / "github/prediction-market-analysis/data/polymarket/markets"
    market_files = sorted(market_dir.glob("markets_*.parquet"))[:max_files]

    markets = []
    for f in market_files:
        df = pd.read_parquet(f)
        markets.append(df)

    df = pd.concat(markets, ignore_index=True)
    print(f"  Loaded {len(df)} markets")
    print(f"  Columns: {df.columns.tolist()}")

    return df


def compute_polymarket_features(markets):
    """Compute analysis features from raw Polymarket market data"""

    print("\nComputing features...")
    df = markets.copy()

    # Parse outcomes and prices
    if "outcomes" in df.columns and "outcome_prices" in df.columns:
        # outcomes: ['Yes', 'No']
        # outcome_prices: ['0.52', '0.48']
        df["last_price_yes"] = df["outcome_prices"].apply(
            lambda x: float(eval(x)[0]) if isinstance(x, str) and x.startswith("[") else np.nan
        )
    else:
        df["last_price_yes"] = np.nan

    # Volume and liquidity
    df["log_volume"] = np.log1p(df["volume"].fillna(0))
    df["high_volume"] = df["volume"] >= df["volume"].median()

    # Time horizon
    if "end_date" in df.columns and "created_at" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"])
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["horizon_days"] = (df["end_date"] - df["created_at"]).dt.total_seconds() / 86400
        df["short_horizon"] = df["horizon_days"] < 7
    else:
        df["horizon_days"] = np.nan
        df["short_horizon"] = False

    # Market status
    df["resolved"] = df["closed"] & (~df["active"])

    # Spread proxy (from liquidity)
    # Lower liquidity → higher spread
    df["spread_proxy"] = 1 / (1 + df["liquidity"].fillna(0))

    print(f"  Resolved markets: {df['resolved'].sum()}")
    print(f"  With volume data: {df['volume'].notna().sum()}")
    print(f"  With horizon data: {df['horizon_days'].notna().sum()}")

    return df


# ============================================================================
# DISCOVERY LOG
# ============================================================================


class CrossPlatformLog:
    """Track cross-platform replication attempts"""

    def __init__(self):
        self.replications = []

    def register(self, hypothesis_id, kalshi_finding, polymarket_prediction):
        """Register replication attempt"""
        self.replications.append(
            {
                "id": hypothesis_id,
                "kalshi_finding": kalshi_finding,
                "polymarket_prediction": polymarket_prediction,
                "train_result": None,
                "test_result": None,
                "robustness": [],
                "verdict": None,
            }
        )

    def add_train_result(self, idx, result):
        self.replications[idx]["train_result"] = result

    def add_test_result(self, idx, result):
        self.replications[idx]["test_result"] = result

    def add_robustness(self, idx, check):
        self.replications[idx]["robustness"].append(check)

    def set_verdict(self, idx, verdict, reason):
        self.replications[idx]["verdict"] = {"status": verdict, "reason": reason}

    def to_markdown(self):
        """Export cross-platform comparison"""
        md = "# CROSS-PLATFORM REPLICATION RESULTS\n\n"
        md += "## Methodology\n\n"
        md += "1. Loaded salvaged Kalshi findings from adversarial review\n"
        md += "2. Pre-registered replication predictions for Polymarket\n"
        md += "3. Split Polymarket data 70/30 train/test\n"
        md += "4. Tested on training set with robustness checks\n"
        md += "5. Validated on holdout test set\n"
        md += "6. Reported replication success/failure honestly\n\n"

        md += "## Findings\n\n"

        for rep in self.replications:
            md += f"### {rep['id']}\n\n"
            md += f"**Kalshi Finding:** {rep['kalshi_finding']}\n\n"
            md += f"**Polymarket Prediction:** {rep['polymarket_prediction']}\n\n"

            if rep["train_result"]:
                md += "**Training Set Result:**\n"
                for k, v in rep["train_result"].items():
                    md += f"- {k}: {v}\n"
                md += "\n"

            if rep["robustness"]:
                md += "**Robustness Checks:**\n"
                for check in rep["robustness"]:
                    md += f"- {check}\n"
                md += "\n"

            if rep["test_result"]:
                md += "**Holdout Test Result:**\n"
                for k, v in rep["test_result"].items():
                    md += f"- {k}: {v}\n"
                md += "\n"

            if rep["verdict"]:
                md += f"**VERDICT: {rep['verdict']['status']}**\n\n"
                md += f"{rep['verdict']['reason']}\n\n"

            md += "---\n\n"

        # Summary table
        md += "## Summary Table: Kalshi vs Polymarket\n\n"
        md += "| Finding | Kalshi | Polymarket (Train) | Polymarket (Test) | Replicates? |\n"
        md += "|---------|--------|-------------------|-------------------|-------------|\n"

        for rep in self.replications:
            kalshi = rep["kalshi_finding"][:40]
            train = str(rep["train_result"]) if rep["train_result"] else "-"
            test = str(rep["test_result"]) if rep["test_result"] else "-"
            verdict = rep["verdict"]["status"] if rep["verdict"] else "?"
            md += f"| {rep['id']} | {kalshi} | {train[:40]} | {test[:40]} | {verdict} |\n"

        return md


# ============================================================================
# REPLICATION TESTS
# ============================================================================


def test_h1_replication(train_df, test_df, log, idx=0):
    """Replicate H1: Horizon×Volume interaction"""

    print("\n" + "=" * 80)
    print("REPLICATION TEST: H1 Horizon×Volume Interaction")
    print("=" * 80)

    # Filter to usable data
    train_valid = train_df[train_df["volume"].notna() & train_df["horizon_days"].notna()].copy()

    test_valid = test_df[test_df["volume"].notna() & test_df["horizon_days"].notna()].copy()

    print(f"\nTraining set: {len(train_valid)} markets")
    print(f"Test set: {len(test_valid)} markets")

    if len(train_valid) < 100:
        log.add_train_result(idx, {"error": "Insufficient data"})
        log.set_verdict(idx, "UNCERTAIN", "Insufficient Polymarket data for testing")
        return

    # Compute group means
    train_results = train_valid.groupby(["short_horizon", "high_volume"])["volume"].agg(["mean", "count"])
    print("\nTraining set volume by groups:")
    print(train_results)

    # Test interaction via correlation
    # Kalshi: Short horizon + high volume → lower error
    # Proxy: Check if volume distribution differs by horizon

    short_vol = train_valid[train_valid["short_horizon"]]["log_volume"].mean()
    long_vol = train_valid[~train_valid["short_horizon"]]["log_volume"].mean()

    train_result = {
        "short_horizon_mean_log_vol": f"{short_vol:.3f}",
        "long_horizon_mean_log_vol": f"{long_vol:.3f}",
        "difference": f"{short_vol - long_vol:.3f}",
        "n_short": int(train_valid["short_horizon"].sum()),
        "n_long": int((~train_valid["short_horizon"]).sum()),
    }

    log.add_train_result(idx, train_result)

    # Test set validation
    if len(test_valid) >= 50:
        short_vol_test = test_valid[test_valid["short_horizon"]]["log_volume"].mean()
        long_vol_test = test_valid[~test_valid["short_horizon"]]["log_volume"].mean()

        test_result = {
            "short_horizon_mean_log_vol": f"{short_vol_test:.3f}",
            "long_horizon_mean_log_vol": f"{long_vol_test:.3f}",
            "difference": f"{short_vol_test - long_vol_test:.3f}",
            "direction_matches_train": np.sign(short_vol - long_vol) == np.sign(short_vol_test - long_vol_test),
        }

        log.add_test_result(idx, test_result)

    # Robustness
    log.add_robustness(idx, "Volume data available but lacking error/outcome data for full replication")
    log.add_robustness(idx, "Cannot compute MAE without resolved outcomes with known results")

    log.set_verdict(
        idx,
        "PARTIAL",
        "Can observe volume×horizon patterns but lack outcome data for error calculation. "
        + "Polymarket market structure differs significantly from Kalshi (CTF tokens vs direct markets).",
    )


def test_h2_replication(train_df, test_df, log, idx=1):
    """Replicate H2: Spread correlation"""

    print("\n" + "=" * 80)
    print("REPLICATION TEST: H2 Spread-Error Correlation")
    print("=" * 80)

    # Check if we have both spread and outcome data
    if "spread_proxy" not in train_df.columns:
        log.add_train_result(idx, {"error": "No spread data"})
        log.set_verdict(idx, "UNCERTAIN", "Polymarket data lacks explicit spread measurement")
        return

    # Use liquidity as inverse of spread
    train_valid = train_df[train_df["liquidity"].notna()].copy()
    test_valid = test_df[test_df["liquidity"].notna()].copy()

    print(f"\nTraining set: {len(train_valid)} markets with liquidity data")
    print(f"Test set: {len(test_valid)} markets with liquidity data")

    # Summary stats
    train_result = {
        "mean_liquidity": f"{train_valid['liquidity'].mean():.2f}",
        "median_liquidity": f"{train_valid['liquidity'].median():.2f}",
        "std_liquidity": f"{train_valid['liquidity'].std():.2f}",
        "n": len(train_valid),
    }

    log.add_train_result(idx, train_result)

    if len(test_valid) >= 50:
        test_result = {
            "mean_liquidity": f"{test_valid['liquidity'].mean():.2f}",
            "median_liquidity": f"{test_valid['liquidity'].median():.2f}",
            "similar_to_train": abs(train_valid["liquidity"].mean() - test_valid["liquidity"].mean())
            < train_valid["liquidity"].std(),
        }
        log.add_test_result(idx, test_result)

    log.add_robustness(idx, "Liquidity is inverse proxy for spread, not direct measurement")
    log.add_robustness(idx, "Cannot compute error correlation without outcome/resolution data")

    log.set_verdict(
        idx,
        "UNCERTAIN",
        "Polymarket provides liquidity but not bid-ask spread directly. "
        + "Lack of outcome data prevents error correlation analysis.",
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    print("=" * 80)
    print("RIGOROUS CROSS-PLATFORM VALIDATION: KALSHI → POLYMARKET")
    print("=" * 80)

    # Load Kalshi baseline
    load_kalshi_baseline()

    # Initialize log
    log = CrossPlatformLog()

    # Pre-register replication hypotheses
    print("\n" + "=" * 80)
    print("PRE-REGISTERING REPLICATION HYPOTHESES")
    print("=" * 80)

    log.register(
        "H1_HORIZON_VOLUME",
        "Kalshi: β=-0.064 for horizon×volume interaction",
        "Polymarket should show similar pattern if effect is real and platform-independent",
    )

    log.register(
        "H2_SPREAD_ERROR",
        "Kalshi: r=0.117 for spread-error correlation",
        "Polymarket should show positive correlation between liquidity and accuracy",
    )

    print("✓ Registered 2 replication hypotheses")

    # Load Polymarket data
    print("\n" + "=" * 80)
    print("LOADING POLYMARKET DATA")
    print("=" * 80)

    markets = load_polymarket_markets(max_files=20)
    df = compute_polymarket_features(markets)

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    print(f"\n✓ Train/test split: {len(train_df)}/{len(test_df)} markets")

    # Run replication tests
    print("\n" + "=" * 80)
    print("PHASE 1: REPLICATION TESTS (TRAINING SET)")
    print("=" * 80)

    test_h1_replication(train_df, test_df, log, idx=0)
    test_h2_replication(train_df, test_df, log, idx=1)

    # Export results
    output = log.to_markdown()
    output_path = Path.home() / "github/prediction-market-analysis/CROSS_PLATFORM_ANALYSIS.md"
    output_path.write_text(output)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print("\n✓ Analysis complete")
    print(f"✓ Results saved to: {output_path}")

    # Print key limitation
    print("\n" + "=" * 80)
    print("KEY LIMITATION")
    print("=" * 80)
    print("""
Polymarket data structure fundamentally differs from Kalshi:
- Polymarket: CTF (Conditional Token Framework) with market/outcome separation
- Kalshi: Direct binary markets with integrated pricing
- Polymarket markets lack resolved outcomes in loaded data
- Cannot compute error (abs_error = |price - outcome|) without outcomes
- Volume and liquidity data available but insufficient for full replication

To complete validation, need:
1. Polymarket resolved outcomes (from API or separate datasource)
2. Link trades to specific outcome tokens (Yes/No)
3. Compute final prices per outcome at resolution
4. Then run full regression: error ~ horizon * volume

Current analysis demonstrates METHODOLOGY but lacks data for conclusive test.
    """)

    return log


if __name__ == "__main__":
    log = main()
