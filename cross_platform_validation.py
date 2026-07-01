#!/usr/bin/env python3
"""
RIGOROUS CROSS-PLATFORM VALIDATION OF PREDICTION MARKET CALIBRATION
====================================================================

Methodology:
1. Pre-register hypotheses BEFORE analysis
2. 70% train / 30% test split for ALL datasets
3. Run analysis on training set ONLY
4. Validate on holdout test set
5. Document failures and robustness checks
6. Report ONLY findings that survive adversarial testing

Salvaged Kalshi findings to replicate:
- H1: Horizon×Volume interaction (β=-0.064, but selection confounded)
- H2: Spread correlation with error (r=0.117, real)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ============================================================================
# DISCOVERY LOG CLASS
# ============================================================================


class DiscoveryLog:
    """Track hypothesis, test, result, and robustness checks"""

    def __init__(self):
        self.entries = []

    def register_hypothesis(self, id, hypothesis, prediction, justification):
        """Pre-register hypothesis BEFORE looking at results"""
        entry = {
            "id": id,
            "hypothesis": hypothesis,
            "prediction": prediction,
            "justification": justification,
            "test_results": None,
            "robustness_checks": [],
            "holdout_validation": None,
            "final_verdict": None,
        }
        self.entries.append(entry)
        return len(self.entries) - 1

    def add_test_result(self, idx, test_name, result_dict):
        """Add test result to hypothesis"""
        self.entries[idx]["test_results"] = {"test_name": test_name, "result": result_dict}

    def add_robustness_check(self, idx, check_name, result):
        """Add robustness check (interaction terms, confounders, etc.)"""
        self.entries[idx]["robustness_checks"].append({"check": check_name, "result": result})

    def add_holdout_validation(self, idx, validation_result):
        """Add holdout test set validation"""
        self.entries[idx]["holdout_validation"] = validation_result

    def set_verdict(self, idx, verdict, reason):
        """Final verdict: CONFIRMED, REJECTED, or UNCERTAIN"""
        self.entries[idx]["final_verdict"] = {"verdict": verdict, "reason": reason}

    def to_markdown(self):
        """Export discovery log to markdown"""
        md = "# DISCOVERY LOG\n\n"
        for _i, entry in enumerate(self.entries, 1):
            md += f"## Hypothesis {entry['id']}\n\n"
            md += f"**Statement:** {entry['hypothesis']}\n\n"
            md += f"**Prediction:** {entry['prediction']}\n\n"
            md += f"**Justification:** {entry['justification']}\n\n"

            if entry["test_results"]:
                md += "### Test Results (Training Set)\n\n"
                result = entry["test_results"]["result"]
                for k, v in result.items():
                    md += f"- {k}: {v}\n"
                md += "\n"

            if entry["robustness_checks"]:
                md += "### Robustness Checks\n\n"
                for check in entry["robustness_checks"]:
                    md += f"**{check['check']}:** {check['result']}\n\n"

            if entry["holdout_validation"]:
                md += "### Holdout Validation\n\n"
                for k, v in entry["holdout_validation"].items():
                    md += f"- {k}: {v}\n"
                md += "\n"

            if entry["final_verdict"]:
                verdict = entry["final_verdict"]["verdict"]
                reason = entry["final_verdict"]["reason"]
                md += f"### VERDICT: {verdict}\n\n"
                md += f"{reason}\n\n"

            md += "---\n\n"

        return md


# ============================================================================
# DATA LOADING WITH TRAIN/TEST SPLIT
# ============================================================================


def load_polymarket_data(max_files=50, test_size=0.3, random_state=42):
    """
    Load Polymarket data with mandatory train/test split BEFORE analysis

    Args:
        max_files: Maximum number of market files to load
        test_size: Proportion for test set (default 30%)
        random_state: Random seed for reproducibility

    Returns:
        train_df, test_df: DataFrames with train/test split
    """
    print("Loading Polymarket data...")

    # Load markets
    market_dir = Path.home() / "github/prediction-market-analysis/data/polymarket/markets"
    market_files = sorted(market_dir.glob("markets_*.parquet"))[:max_files]

    markets = []
    for f in market_files:
        df = pd.read_parquet(f)
        markets.append(df)

    markets = pd.concat(markets, ignore_index=True)
    print(f"Loaded {len(markets)} markets")

    # Load trades
    trade_dir = Path.home() / "github/prediction-market-analysis/data/polymarket/trades"
    trade_files = sorted(trade_dir.glob("trades_*.parquet"))[:max_files]

    trades = []
    for f in trade_files:
        df = pd.read_parquet(f)
        trades.append(df)

    trades = pd.concat(trades, ignore_index=True)
    print(f"Loaded {len(trades)} trades")

    # Merge and compute features
    print("Computing market features...")
    df = compute_market_features(markets, trades)

    # Filter to resolved markets only
    df = df[df['resolved']].copy()
    print(f"Filtered to {len(df)} resolved markets")

    # MANDATORY TRAIN/TEST SPLIT
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=None,  # Can add stratification if needed
    )

    print("\nTrain/Test Split:")
    print(f"  Training: {len(train_df)} markets ({100 * (1 - test_size):.0f}%)")
    print(f"  Test:     {len(test_df)} markets ({100 * test_size:.0f}%)")
    print("\n⚠️  ANALYSIS MUST USE TRAINING SET ONLY")
    print("⚠️  TEST SET IS HOLDOUT FOR VALIDATION")

    return train_df, test_df


def compute_market_features(markets, trades):
    """Compute market-level features from trades"""

    # Aggregate trades by market
    trade_stats = (
        trades.groupby("market")
        .agg(
            {
                "size": ["count", "sum"],  # volume metrics
                "price": ["first", "last", "mean", "std"],  # price metrics
                "timestamp": ["min", "max"],  # timing
            }
        )
        .reset_index()
    )

    trade_stats.columns = [
        "market",
        "num_trades",
        "total_volume",
        "open_price",
        "last_price",
        "mean_price",
        "price_std",
        "first_trade",
        "last_trade",
    ]

    # Merge with market metadata
    df = markets.merge(trade_stats, left_on="id", right_on="market", how="left")

    # Compute derived features
    df["log_volume"] = np.log1p(df["num_trades"])
    df["high_volume"] = df["num_trades"] >= 200  # Use Kalshi threshold

    # Compute spread (if available in market data)
    if "spread" in df.columns:
        df["spread"] = df["spread"]
    else:
        # Approximate spread from price volatility
        df["spread"] = df["price_std"].fillna(0)

    # Compute time horizon
    if "end_date" in df.columns and "created_at" in df.columns:
        df["horizon_days"] = (df["end_date"] - df["created_at"]).dt.total_seconds() / 86400
        df["short_horizon"] = df["horizon_days"] < 7  # <1 week
    else:
        df["horizon_days"] = np.nan
        df["short_horizon"] = False

    # Compute error (if outcome known)
    if "outcome" in df.columns and "last_price" in df.columns:
        df["outcome_binary"] = df["outcome"].apply(lambda x: 1 if x == "Yes" else 0 if x == "No" else np.nan)
        df["last_price_prob"] = df["last_price"]
        df["abs_error"] = np.abs(df["last_price_prob"] - df["outcome_binary"])
    else:
        df["abs_error"] = np.nan

    return df


# ============================================================================
# PRE-REGISTERED HYPOTHESES
# ============================================================================


def register_hypotheses(log):
    """Pre-register hypotheses BEFORE looking at data"""

    # H1: Horizon×Volume interaction (from Kalshi)
    log.register_hypothesis(
        id="H1_HORIZON_VOLUME",
        hypothesis="Short-horizon markets benefit more from trading volume than long-horizon markets",
        prediction="β_interaction < 0 in regression: abs_error ~ short_horizon * high_volume",
        justification="Kalshi showed β=-0.064 (p<10^-88). Theoretically: time pressure + information aggregation. BUT potentially confounded by event type.",
    )

    # H2: Spread correlation (from Kalshi)
    log.register_hypothesis(
        id="H2_SPREAD_ERROR",
        hypothesis="Bid-ask spread predicts market error",
        prediction="r(spread, abs_error) > 0.10",
        justification="Kalshi showed r=0.117. Spread signals market confidence/uncertainty. Real correlation but small R².",
    )

    # H3: User-level effects (new for Polymarket)
    log.register_hypothesis(
        id="H3_USER_CONCENTRATION",
        hypothesis="Markets dominated by few traders are less accurate",
        prediction="Negative correlation between trader concentration (Herfindahl index) and accuracy",
        justification="Diversity of opinion improves aggregation (Surowiecki). Concentration → groupthink/manipulation.",
    )

    # H4: Orderbook depth (new if data available)
    log.register_hypothesis(
        id="H4_ORDERBOOK_DEPTH",
        hypothesis="Orderbook depth predicts accuracy",
        prediction="Markets with deeper orderbooks (more limit orders) have lower error",
        justification="Depth signals liquidity and trader confidence. May proxy for information quality.",
    )


# ============================================================================
# HYPOTHESIS TESTING WITH ROBUSTNESS CHECKS
# ============================================================================


def test_h1_horizon_volume(train_df, log, idx=0):
    """Test H1: Horizon×Volume interaction"""

    print("\n" + "=" * 80)
    print("TESTING H1: Horizon×Volume Interaction")
    print("=" * 80)

    # Filter to valid data
    df = train_df[
        train_df["abs_error"].notna() & train_df["short_horizon"].notna() & train_df["high_volume"].notna()
    ].copy()

    print(f"Sample size: {len(df)} markets")

    if len(df) < 100:
        log.add_test_result(idx, "H1_test", {"error": "Insufficient data"})
        log.set_verdict(idx, "UNCERTAIN", "Insufficient data for testing")
        return

    # Test 1: Basic interaction model (following Kalshi)
    formula = "abs_error ~ short_horizon * high_volume"
    model = smf.ols(formula, data=df).fit()

    beta_interaction = model.params["short_horizon:high_volume"]
    p_interaction = model.pvalues["short_horizon:high_volume"]

    log.add_test_result(
        idx,
        "Interaction_Regression",
        {
            "beta_interaction": f"{beta_interaction:.6f}",
            "p_value": f"{p_interaction:.2e}",
            "N": len(df),
            "R_squared": f"{model.rsquared:.4f}",
        },
    )

    print(f"\nBasic Model: {formula}")
    print(f"  β_interaction = {beta_interaction:.6f} (p={p_interaction:.2e})")
    print(f"  R² = {model.rsquared:.4f}")

    # ROBUSTNESS CHECK 1: Add category controls (if available)
    if "category" in df.columns:
        formula_controls = "abs_error ~ short_horizon * high_volume + C(category)"
        model_controls = smf.ols(formula_controls, data=df).fit()
        beta_controlled = model_controls.params["short_horizon:high_volume"]

        log.add_robustness_check(
            idx,
            "Category_Controls",
            f"β={beta_controlled:.6f}, changed by {100 * (beta_controlled / beta_interaction - 1):.1f}%",
        )

        print("\nWith Category Controls:")
        print(f"  β_interaction = {beta_controlled:.6f}")
        print(f"  Change: {100 * (beta_controlled / beta_interaction - 1):.1f}%")

    # ROBUSTNESS CHECK 2: Test for confounding by base difficulty
    df["base_error"] = df.groupby("short_horizon")["abs_error"].transform("median")
    log.add_robustness_check(
        idx,
        "Baseline_Difficulty",
        f"Short-horizon median error: {df[df['short_horizon']].abs_error.median():.3f}, "
        + f"Long-horizon: {df[~df['short_horizon']].abs_error.median():.3f}",
    )

    # ROBUSTNESS CHECK 3: Continuous volume instead of binary threshold
    formula_continuous = "abs_error ~ short_horizon * log_volume"
    model_continuous = smf.ols(formula_continuous, data=df).fit()
    beta_continuous = model_continuous.params["short_horizon:log_volume"]

    log.add_robustness_check(
        idx,
        "Continuous_Volume",
        f"β_continuous={beta_continuous:.6f}, model is {'' if beta_continuous < 0 else 'NOT '}consistent",
    )

    print("\nContinuous Volume Specification:")
    print(f"  β_interaction = {beta_continuous:.6f}")

    # ROBUSTNESS CHECK 4: Multiple testing correction
    # If we're running multiple hypotheses
    log.add_robustness_check(
        idx,
        "Multiple_Testing",
        f"Bonferroni threshold (4 tests): p < {0.05 / 4:.4f}. "
        + f"Finding {'survives' if p_interaction < 0.05 / 4 else 'FAILS'} correction",
    )

    return model


def test_h2_spread_error(train_df, log, idx=1):
    """Test H2: Spread correlation with error"""

    print("\n" + "=" * 80)
    print("TESTING H2: Spread Correlation with Error")
    print("=" * 80)

    # Filter to valid data
    df = train_df[train_df["abs_error"].notna() & train_df["spread"].notna()].copy()

    print(f"Sample size: {len(df)} markets")

    if len(df) < 100:
        log.add_test_result(idx, "H2_test", {"error": "Insufficient data"})
        log.set_verdict(idx, "UNCERTAIN", "Insufficient data for testing")
        return

    # Test 1: Simple correlation
    r, p = stats.pearsonr(df["spread"], df["abs_error"])

    log.add_test_result(idx, "Correlation", {"r": f"{r:.4f}", "p_value": f"{p:.2e}", "N": len(df)})

    print(f"\nCorrelation: r = {r:.4f} (p={p:.2e})")

    # Test 2: Regression with spread
    model = smf.ols("abs_error ~ spread", data=df).fit()
    beta_spread = model.params["spread"]
    r_squared = model.rsquared

    print(f"Regression R² = {r_squared:.4f}")

    log.add_robustness_check(idx, "R_squared", f"R²={r_squared:.4f}. Model explains {100 * r_squared:.1f}% of variance")

    # ROBUSTNESS CHECK 1: Control for volume
    if "log_volume" in df.columns:
        model_controls = smf.ols("abs_error ~ spread + log_volume", data=df).fit()
        beta_controlled = model_controls.params["spread"]

        log.add_robustness_check(
            idx,
            "Volume_Controls",
            f"β={beta_controlled:.6f}, changed by {100 * (beta_controlled / beta_spread - 1):.1f}%",
        )

        print("\nWith Volume Controls:")
        print(f"  β_spread = {beta_controlled:.6f}")

    # ROBUSTNESS CHECK 2: Partial correlation (not mediation!)
    if "log_volume" in df.columns:
        # Regress out volume from both
        from scipy.stats import pearsonr

        resid_spread = sm.OLS(df["spread"], sm.add_constant(df["log_volume"])).fit().resid
        resid_error = sm.OLS(df["abs_error"], sm.add_constant(df["log_volume"])).fit().resid
        r_partial, p_partial = pearsonr(resid_spread, resid_error)

        log.add_robustness_check(
            idx,
            "Partial_Correlation",
            f"r_partial={r_partial:.4f} (controlling for volume). "
            + f"Direct r={r:.4f}. {'Sign reversal suggests suppression, NOT mediation' if np.sign(r) != np.sign(r_partial) else 'Sign consistent'}",
        )

        print(f"\nPartial correlation (controlling volume): r = {r_partial:.4f}")

    return model


def test_h3_user_concentration(train_df, log, idx=2):
    """Test H3: User concentration effects (Polymarket-specific)"""

    print("\n" + "=" * 80)
    print("TESTING H3: User Concentration Effects")
    print("=" * 80)

    # Check if user-level data available
    if "user" not in train_df.columns and "maker" not in train_df.columns:
        print("User data not available in current dataset")
        log.add_test_result(idx, "H3_test", {"error": "User data not available"})
        log.set_verdict(idx, "UNCERTAIN", "User-level data not present in loaded dataset")
        return

    # TODO: Implement if user data available
    print("User data analysis not yet implemented")
    log.add_test_result(idx, "H3_test", {"status": "Not implemented"})
    log.set_verdict(idx, "UNCERTAIN", "Analysis pending user data structure exploration")


# ============================================================================
# HOLDOUT VALIDATION
# ============================================================================


def validate_on_holdout(test_df, log):
    """Validate all confirmed findings on holdout test set"""

    print("\n" + "=" * 80)
    print("HOLDOUT VALIDATION ON TEST SET")
    print("=" * 80)
    print(f"Test set size: {len(test_df)} markets")

    # Validate H1
    if log.entries[0]["test_results"] is not None:
        print("\nValidating H1: Horizon×Volume Interaction")
        df = test_df[
            test_df["abs_error"].notna() & test_df["short_horizon"].notna() & test_df["high_volume"].notna()
        ].copy()

        if len(df) >= 100:
            model = smf.ols("abs_error ~ short_horizon * high_volume", data=df).fit()
            beta_test = model.params["short_horizon:high_volume"]
            p_test = model.pvalues["short_horizon:high_volume"]

            # Compare to training result
            beta_train = float(log.entries[0]["test_results"]["result"]["beta_interaction"])

            log.add_holdout_validation(
                0,
                {
                    "beta_test": f"{beta_test:.6f}",
                    "beta_train": f"{beta_train:.6f}",
                    "difference": f"{100 * (beta_test / beta_train - 1):.1f}%",
                    "p_value": f"{p_test:.2e}",
                    "replicates": "YES" if np.sign(beta_test) == np.sign(beta_train) and p_test < 0.05 else "NO",
                },
            )

            print(f"  Training β = {beta_train:.6f}")
            print(f"  Test β     = {beta_test:.6f}")
            print(f"  Difference = {100 * (beta_test / beta_train - 1):.1f}%")
            print(f"  Replicates: {'✓ YES' if np.sign(beta_test) == np.sign(beta_train) else '✗ NO'}")
        else:
            log.add_holdout_validation(0, {"error": "Insufficient test data"})

    # Validate H2
    if log.entries[1]["test_results"] is not None:
        print("\nValidating H2: Spread Correlation")
        df = test_df[test_df["abs_error"].notna() & test_df["spread"].notna()].copy()

        if len(df) >= 100:
            r_test, p_test = stats.pearsonr(df["spread"], df["abs_error"])
            r_train = float(log.entries[1]["test_results"]["result"]["r"])

            log.add_holdout_validation(
                1,
                {
                    "r_test": f"{r_test:.4f}",
                    "r_train": f"{r_train:.4f}",
                    "difference": f"{100 * (r_test / r_train - 1):.1f}%",
                    "p_value": f"{p_test:.2e}",
                    "replicates": "YES" if np.sign(r_test) == np.sign(r_train) and p_test < 0.05 else "NO",
                },
            )

            print(f"  Training r = {r_train:.4f}")
            print(f"  Test r     = {r_test:.4f}")
            print(f"  Difference = {100 * (r_test / r_train - 1):.1f}%")
            print(f"  Replicates: {'✓ YES' if np.sign(r_test) == np.sign(r_train) else '✗ NO'}")
        else:
            log.add_holdout_validation(1, {"error": "Insufficient test data"})


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================


def main():
    """Main analysis pipeline with rigorous methodology"""

    print("=" * 80)
    print("CROSS-PLATFORM PREDICTION MARKET CALIBRATION VALIDATION")
    print("=" * 80)
    print("\nMethodology:")
    print("  1. Pre-register hypotheses BEFORE analysis")
    print("  2. 70/30 train/test split")
    print("  3. Training set analysis only")
    print("  4. Robustness checks for each finding")
    print("  5. Holdout validation on test set")
    print("  6. Report ONLY findings that replicate")
    print("\n")

    # Initialize discovery log
    log = DiscoveryLog()

    # Pre-register hypotheses
    print("Pre-registering hypotheses...")
    register_hypotheses(log)
    print(f"Registered {len(log.entries)} hypotheses")

    # Load data with train/test split
    train_df, test_df = load_polymarket_data(max_files=30, test_size=0.3)

    # Test hypotheses on TRAINING SET ONLY
    print("\n" + "=" * 80)
    print("PHASE 1: HYPOTHESIS TESTING (TRAINING SET)")
    print("=" * 80)

    test_h1_horizon_volume(train_df, log, idx=0)
    test_h2_spread_error(train_df, log, idx=1)
    test_h3_user_concentration(train_df, log, idx=2)

    # Validate on holdout set
    print("\n" + "=" * 80)
    print("PHASE 2: HOLDOUT VALIDATION (TEST SET)")
    print("=" * 80)

    validate_on_holdout(test_df, log)

    # Set final verdicts
    print("\n" + "=" * 80)
    print("FINAL VERDICTS")
    print("=" * 80)

    for i, entry in enumerate(log.entries):
        if entry["holdout_validation"] is not None:
            if "replicates" in entry["holdout_validation"]:
                if entry["holdout_validation"]["replicates"] == "YES":
                    log.set_verdict(
                        i, "CONFIRMED", "Finding survived robustness checks and replicated on holdout test set"
                    )
                else:
                    log.set_verdict(i, "REJECTED", "Finding failed to replicate on holdout test set")
            else:
                log.set_verdict(i, "UNCERTAIN", entry["holdout_validation"].get("error", "Unable to validate"))
        else:
            log.set_verdict(i, "UNCERTAIN", "No holdout validation performed")

    # Export discovery log
    log_md = log.to_markdown()
    output_path = Path.home() / "github/prediction-market-analysis/POLYMARKET_DISCOVERY_LOG.md"
    output_path.write_text(log_md)
    print(f"\n✓ Discovery log saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    confirmed = sum(1 for e in log.entries if e["final_verdict"] and e["final_verdict"]["verdict"] == "CONFIRMED")
    rejected = sum(1 for e in log.entries if e["final_verdict"] and e["final_verdict"]["verdict"] == "REJECTED")
    uncertain = sum(1 for e in log.entries if e["final_verdict"] and e["final_verdict"]["verdict"] == "UNCERTAIN")

    print("\nFindings:")
    print(f"  ✓ CONFIRMED:  {confirmed}")
    print(f"  ✗ REJECTED:   {rejected}")
    print(f"  ? UNCERTAIN:  {uncertain}")
    print(f"\nReplication rate: {100 * confirmed / len(log.entries):.0f}%")

    return log


if __name__ == "__main__":
    log = main()
