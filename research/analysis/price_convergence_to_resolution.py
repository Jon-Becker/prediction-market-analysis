#!/usr/bin/env python3
"""Analyze market calibration by time to resolution.

Measures excess win rate and mean absolute error by hours remaining until market
close. Note: Long-running markets (>4 weeks) show strong calibration, suggesting
selection effects or that well-calibrated markets tend to run longer.
Includes z-tests for statistical significance.
"""

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

    df = con.execute(
        f"""
        WITH trade_data AS (
            SELECT
                EXTRACT(EPOCH FROM (m.close_time - t.created_time)) / 3600.0 AS hours_to_close,
                (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS price_frac,
                CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won
            FROM '{trades_dir}/*.parquet' t
            JOIN '{markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE m.result IN ('yes', 'no')
              AND m.close_time IS NOT NULL
              AND t.created_time < m.close_time
        ),
        binned AS (
            SELECT
                CASE
                    WHEN hours_to_close < 1 THEN 0.5
                    WHEN hours_to_close < 2 THEN 1.5
                    WHEN hours_to_close < 4 THEN 3
                    WHEN hours_to_close < 8 THEN 6
                    WHEN hours_to_close < 24 THEN 16
                    WHEN hours_to_close < 72 THEN 48
                    WHEN hours_to_close < 168 THEN 120
                    WHEN hours_to_close < 720 THEN 444
                    ELSE 1000
                END AS hours_bin,
                CASE
                    WHEN hours_to_close < 1 THEN '<1h'
                    WHEN hours_to_close < 2 THEN '1-2h'
                    WHEN hours_to_close < 4 THEN '2-4h'
                    WHEN hours_to_close < 8 THEN '4-8h'
                    WHEN hours_to_close < 24 THEN '8-24h'
                    WHEN hours_to_close < 72 THEN '1-3d'
                    WHEN hours_to_close < 168 THEN '3-7d'
                    WHEN hours_to_close < 720 THEN '1-4w'
                    ELSE '>4w'
                END AS hours_label,
                AVG(won) AS win_rate,
                AVG(price_frac) AS expected_win_rate,
                AVG(won - price_frac) AS excess_win_rate,
                VAR_POP(won - price_frac) AS var_excess,
                AVG(ABS(won - price_frac)) AS mean_abs_error,
                COUNT(*) AS n_trades
            FROM trade_data
            WHERE hours_to_close >= 0
            GROUP BY hours_bin, hours_label
        )
        SELECT hours_bin, hours_label, win_rate, expected_win_rate, excess_win_rate, var_excess, mean_abs_error, n_trades
        FROM binned
        ORDER BY hours_bin
        """
    ).df()

    # Z-test: H0: excess_win_rate = 0 (market is perfectly calibrated)
    df["se_excess"] = np.sqrt(df["var_excess"] / df["n_trades"])
    df["z_stat"] = df["excess_win_rate"] / df["se_excess"]
    df["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(df["z_stat"])))
    df["significant"] = df["p_value"] < 0.05

    df.to_csv(fig_dir / "price_convergence_to_resolution.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(df))
    labels = df["hours_label"].tolist()

    ax1.bar(x, df["mean_abs_error"] * 100, color="#4C72B0", alpha=0.7, edgecolor="none")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_xlabel("Time to Resolution")
    ax1.set_ylabel("Mean Absolute Error (pp)")
    ax1.set_title("Price Accuracy vs Time to Resolution\n(note: >4w markets show strong calibration)")
    ax1.invert_xaxis()

    sizes = np.clip(df["n_trades"] / df["n_trades"].max() * 300, 30, 300)
    colors = ["#D65F5F" if sig else "#4C72B0" for sig in df["significant"]]
    ax2.scatter(x, df["excess_win_rate"] * 100, s=sizes, c=colors, alpha=0.7, edgecolor="none")
    ax2.errorbar(x, df["excess_win_rate"] * 100, yerr=1.96 * df["se_excess"] * 100,
                 fmt="none", color="gray", alpha=0.5, capsize=3)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_xlabel("Time to Resolution")
    ax2.set_ylabel("Excess Win Rate (pp)")
    ax2.set_title("Calibration Bias vs Time to Resolution\n(red = p < 0.05, error bars = 95% CI)")
    ax2.invert_xaxis()

    plt.tight_layout()
    fig.savefig(fig_dir / "price_convergence_to_resolution.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "price_convergence_to_resolution.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")
    print(f"\nStatistical Summary:")
    print(f"  H0: Excess win rate = 0 (perfect calibration)")
    for _, row in df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"  {row['hours_label']:>6}: excess={row['excess_win_rate']*100:+.3f}pp, z={row['z_stat']:.2f}, p={row['p_value']:.2e} {sig}")


if __name__ == "__main__":
    main()
