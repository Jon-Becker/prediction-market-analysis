#!/usr/bin/env python3
"""Analyze market duration effects on trading behavior and accuracy.

Compares short-lived markets vs long-running ones to identify differences
in trader behavior, accuracy, and market characteristics.
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
        WITH market_stats AS (
            SELECT
                m.ticker,
                m.result,
                EXTRACT(EPOCH FROM (m.close_time - m.open_time)) / 3600.0 AS duration_hours,
                m.volume AS market_volume
            FROM '{markets_dir}/*.parquet' m
            WHERE m.result IN ('yes', 'no')
              AND m.open_time IS NOT NULL
              AND m.close_time IS NOT NULL
              AND m.close_time > m.open_time
        ),
        trade_data AS (
            SELECT
                ms.duration_hours,
                (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS price_frac,
                CASE WHEN t.taker_side = ms.result THEN 1.0 ELSE 0.0 END AS won,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet' t
            JOIN market_stats ms ON t.ticker = ms.ticker
        )
        SELECT
            CASE
                WHEN duration_hours < 1 THEN 0.5
                WHEN duration_hours < 4 THEN 2.5
                WHEN duration_hours < 12 THEN 8
                WHEN duration_hours < 24 THEN 18
                WHEN duration_hours < 72 THEN 48
                WHEN duration_hours < 168 THEN 120
                WHEN duration_hours < 720 THEN 444
                ELSE 1000
            END AS duration_bin,
            CASE
                WHEN duration_hours < 1 THEN '<1h'
                WHEN duration_hours < 4 THEN '1-4h'
                WHEN duration_hours < 12 THEN '4-12h'
                WHEN duration_hours < 24 THEN '12-24h'
                WHEN duration_hours < 72 THEN '1-3d'
                WHEN duration_hours < 168 THEN '3-7d'
                WHEN duration_hours < 720 THEN '1-4w'
                ELSE '>4w'
            END AS duration_label,
            AVG(won) AS win_rate,
            AVG(price_frac) AS expected_win_rate,
            AVG(won - price_frac) AS excess_win_rate,
            VAR_POP(won - price_frac) AS var_excess,
            AVG(ABS(won - price_frac)) AS mean_abs_error,
            AVG(volume_usd) AS avg_trade_size,
            SUM(volume_usd) AS total_volume,
            COUNT(*) AS n_trades
        FROM trade_data
        GROUP BY duration_bin, duration_label
        HAVING COUNT(*) >= 1000
        ORDER BY duration_bin
        """
    ).df()

    # Z-test: H0: excess_win_rate = 0
    df["se_excess"] = np.sqrt(df["var_excess"] / df["n_trades"])
    df["z_stat"] = df["excess_win_rate"] / df["se_excess"]
    df["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(df["z_stat"])))
    df["significant"] = df["p_value"] < 0.05

    df_market_counts = con.execute(
        f"""
        SELECT
            CASE
                WHEN EXTRACT(EPOCH FROM (close_time - open_time)) / 3600.0 < 1 THEN '<1h'
                WHEN EXTRACT(EPOCH FROM (close_time - open_time)) / 3600.0 < 4 THEN '1-4h'
                WHEN EXTRACT(EPOCH FROM (close_time - open_time)) / 3600.0 < 12 THEN '4-12h'
                WHEN EXTRACT(EPOCH FROM (close_time - open_time)) / 3600.0 < 24 THEN '12-24h'
                WHEN EXTRACT(EPOCH FROM (close_time - open_time)) / 3600.0 < 72 THEN '1-3d'
                WHEN EXTRACT(EPOCH FROM (close_time - open_time)) / 3600.0 < 168 THEN '3-7d'
                WHEN EXTRACT(EPOCH FROM (close_time - open_time)) / 3600.0 < 720 THEN '1-4w'
                ELSE '>4w'
            END AS duration_label,
            COUNT(*) AS n_markets,
            AVG(volume) AS avg_market_volume
        FROM '{markets_dir}/*.parquet'
        WHERE result IN ('yes', 'no')
          AND open_time IS NOT NULL
          AND close_time IS NOT NULL
          AND close_time > open_time
        GROUP BY duration_label
        """
    ).df()

    df.to_csv(fig_dir / "market_duration_effects.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(len(df))
    labels = df["duration_label"].tolist()

    ax1 = axes[0, 0]
    ax1.bar(x, df["mean_abs_error"] * 100, color="#4C72B0", alpha=0.7, edgecolor="none")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_xlabel("Market Duration")
    ax1.set_ylabel("Mean Absolute Error (pp)")
    ax1.set_title("Price Accuracy by Market Duration")

    ax2 = axes[0, 1]
    colors = ["#D65F5F" if sig else "#888888" for sig in df["significant"]]
    ax2.bar(x, df["excess_win_rate"] * 100, color=colors, alpha=0.7, edgecolor="none")
    ax2.errorbar(x, df["excess_win_rate"] * 100, yerr=1.96 * df["se_excess"] * 100,
                 fmt="none", color="gray", alpha=0.5, capsize=3)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_xlabel("Market Duration")
    ax2.set_ylabel("Excess Win Rate (pp)")
    ax2.set_title("Calibration Bias by Market Duration\n(red = p < 0.05)")

    ax3 = axes[1, 0]
    ax3.bar(x, df["avg_trade_size"], color="#55A868", alpha=0.7, edgecolor="none")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha="right")
    ax3.set_xlabel("Market Duration")
    ax3.set_ylabel("Avg Trade Size (USD)")
    ax3.set_title("Trade Size by Market Duration")

    ax4 = axes[1, 1]
    ax4.bar(x, df["n_trades"] / 1e6, color="#8172B3", alpha=0.7, edgecolor="none")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=45, ha="right")
    ax4.set_xlabel("Market Duration")
    ax4.set_ylabel("Number of Trades (millions)")
    ax4.set_title("Trading Activity by Market Duration")

    plt.tight_layout()
    fig.savefig(fig_dir / "market_duration_effects.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "market_duration_effects.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")
    print(f"\nStatistical Summary:")
    print(f"  H0: Excess win rate = 0")
    for _, row in df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"  {row['duration_label']:>6}: excess={row['excess_win_rate']*100:+.3f}pp, z={row['z_stat']:.2f}, p={row['p_value']:.2e} {sig}")


if __name__ == "__main__":
    main()
