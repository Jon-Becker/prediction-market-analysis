#!/usr/bin/env python3
"""Analyze win rate by trade size with confidence intervals.

Examines whether larger trades are more informed (higher excess win rate),
controlling for contract price. Computes 95% confidence intervals for
each trade size bucket.
"""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def main():
    base_dir = Path(__file__).parent.parent.parent
    trades_dir = base_dir / "data" / "trades"
    markets_dir = base_dir / "data" / "markets"
    fig_dir = base_dir / "research" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Compute excess win rate by trade size bin with variance for CI
    # Uses log-scale bins and controls for price by computing excess win rate
    df = con.execute(
        f"""
        WITH trade_data AS (
            SELECT
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS trade_size_usd,
                CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS expected_win_rate
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN '{markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE m.status = 'finalized'
              AND m.result IN ('yes', 'no')
        ),
        binned AS (
            SELECT
                POWER(10, FLOOR(LOG10(GREATEST(trade_size_usd, 0.01)) * 4) / 4.0) AS bin_lower,
                AVG(won) AS win_rate,
                AVG(expected_win_rate) AS expected_win_rate,
                AVG(won - expected_win_rate) AS excess_win_rate,
                VAR_SAMP(won - expected_win_rate) AS var_excess,
                COUNT(*) AS n_trades,
                SUM(trade_size_usd) AS total_volume
            FROM trade_data
            GROUP BY bin_lower
            HAVING COUNT(*) >= 10
        )
        SELECT
            bin_lower AS trade_size_bin,
            win_rate,
            expected_win_rate,
            excess_win_rate,
            var_excess,
            n_trades,
            total_volume
        FROM binned
        ORDER BY bin_lower
        """
    ).df()

    # Calculate standard error and 95% CI
    df["se"] = np.sqrt(df["var_excess"] / df["n_trades"])
    df["ci_lower"] = df["excess_win_rate"] - 1.96 * df["se"]
    df["ci_upper"] = df["excess_win_rate"] + 1.96 * df["se"]

    df.to_csv(fig_dir / "win_rate_by_trade_size.csv", index=False)

    # Create line chart with confidence band
    fig, ax = plt.subplots(figsize=(10, 6))

    x = df["trade_size_bin"].values
    y = df["excess_win_rate"].values * 100
    ci_lower = df["ci_lower"].values * 100
    ci_upper = df["ci_upper"].values * 100

    # Plot confidence band
    ax.fill_between(x, ci_lower, ci_upper, alpha=0.2, color="#4C72B0", label="95% CI")

    # Plot line
    ax.plot(x, y, color="#4C72B0", linewidth=2, marker="o", markersize=4, label="Excess Win Rate")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel("Trade Size (USD)")
    ax.set_ylabel("Excess Win Rate (pp)")
    ax.set_title("Win Rate by Trade Size (price-adjusted)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(fig_dir / "win_rate_by_trade_size.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "win_rate_by_trade_size.pdf", bbox_inches="tight")
    plt.close(fig)

    # JSON output for paper - multi-line chart with CI as separate lines
    json_data = {
        "type": "line",
        "title": "Excess Win Rate by Trade Size",
        "data": [
            {
                "trade_size": round(row["trade_size_bin"], 2),
                "Excess Win Rate": round(row["excess_win_rate"] * 100, 2),
                "95% CI Lower": round(row["ci_lower"] * 100, 2),
                "95% CI Upper": round(row["ci_upper"] * 100, 2),
            }
            for _, row in df.iterrows()
        ],
        "xKey": "trade_size",
        "yKeys": ["Excess Win Rate", "95% CI Lower", "95% CI Upper"],
        "xScale": "log",
        "yUnit": "percent",
        "colors": {
            "Excess Win Rate": "#4C72B0",
            "95% CI Lower": "#4C72B0",
            "95% CI Upper": "#4C72B0",
        },
        "styles": {
            "Excess Win Rate": "solid",
            "95% CI Lower": "dashed",
            "95% CI Upper": "dashed",
        },
    }
    with open(fig_dir / "win_rate_by_trade_size.json", "w") as f:
        json.dump(json_data, f)

    # Print summary statistics
    print(f"Outputs saved to {fig_dir}")
    print("\n=== Win Rate by Trade Size Statistics ===")

    # Compute weighted trend
    log_x = np.log10(x)
    weights = df["n_trades"].values
    coeffs = np.polyfit(log_x, y, 1, w=np.sqrt(weights))
    print(f"\nTrend: {coeffs[0]:+.3f} pp per 10x increase in trade size")
    print(f"Intercept: {coeffs[1]:+.3f} pp at $1")

    # Summary by size buckets
    print("\n=== By Trade Size Bucket ===")
    small = df[df["trade_size_bin"] < 10]
    medium = df[(df["trade_size_bin"] >= 10) & (df["trade_size_bin"] < 100)]
    large = df[(df["trade_size_bin"] >= 100) & (df["trade_size_bin"] < 1000)]
    xlarge = df[df["trade_size_bin"] >= 1000]

    for name, subset in [
        ("Small (<$10)", small),
        ("Medium ($10-$100)", medium),
        ("Large ($100-$1,000)", large),
        ("Very Large (>$1,000)", xlarge),
    ]:
        if len(subset) > 0:
            weighted_excess = (subset["excess_win_rate"] * subset["n_trades"]).sum() / subset["n_trades"].sum() * 100
            total_n = subset["n_trades"].sum()
            total_vol = subset["total_volume"].sum()
            print(f"\n{name}:")
            print(f"  Excess win rate: {weighted_excess:+.2f} pp")
            print(f"  N trades: {total_n:,.0f}")
            print(f"  Total volume: ${total_vol/1e6:,.1f}M")


if __name__ == "__main__":
    main()
