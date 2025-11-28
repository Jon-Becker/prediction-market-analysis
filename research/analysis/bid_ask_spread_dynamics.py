#!/usr/bin/env python3
"""Analyze bid-ask spread dynamics in prediction markets.

Examines how spreads relate to price level, volume, and market characteristics.
Spread may proxy for uncertainty or liquidity.
Includes correlation tests for statistical significance.
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def main():
    base_dir = Path(__file__).parent.parent.parent
    markets_dir = base_dir / "data" / "markets"
    fig_dir = base_dir / "research" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    df_by_price = con.execute(
        f"""
        WITH market_data AS (
            SELECT
                (yes_bid + yes_ask) / 2.0 AS mid_price,
                yes_ask - yes_bid AS yes_spread,
                no_ask - no_bid AS no_spread,
                (yes_ask - yes_bid + no_ask - no_bid) / 2.0 AS avg_spread,
                volume,
                open_interest,
                result
            FROM '{markets_dir}/*.parquet'
            WHERE yes_bid IS NOT NULL
              AND yes_ask IS NOT NULL
              AND yes_bid > 0
              AND yes_ask > 0
              AND yes_ask > yes_bid
              AND result IN ('yes', 'no')
        )
        SELECT
            FLOOR(mid_price / 5) * 5 + 2.5 AS price_bin,
            AVG(avg_spread) AS avg_spread,
            AVG(yes_spread) AS avg_yes_spread,
            MEDIAN(avg_spread) AS median_spread,
            COUNT(*) AS n_markets
        FROM market_data
        WHERE mid_price BETWEEN 1 AND 99
        GROUP BY price_bin
        HAVING COUNT(*) >= 100
        ORDER BY price_bin
        """
    ).df()

    df_by_volume = con.execute(
        f"""
        WITH market_data AS (
            SELECT
                yes_ask - yes_bid AS yes_spread,
                volume,
                result
            FROM '{markets_dir}/*.parquet'
            WHERE yes_bid IS NOT NULL
              AND yes_ask IS NOT NULL
              AND yes_bid > 0
              AND yes_ask > 0
              AND yes_ask > yes_bid
              AND volume > 0
              AND result IN ('yes', 'no')
        )
        SELECT
            POWER(10, FLOOR(LOG10(GREATEST(volume, 1)) * 2) / 2.0) AS volume_bin,
            AVG(yes_spread) AS avg_spread,
            MEDIAN(yes_spread) AS median_spread,
            COUNT(*) AS n_markets
        FROM market_data
        GROUP BY volume_bin
        HAVING COUNT(*) >= 100
        ORDER BY volume_bin
        """
    ).df()

    df_spread_vs_accuracy = con.execute(
        f"""
        WITH market_data AS (
            SELECT
                yes_ask - yes_bid AS spread,
                (yes_bid + yes_ask) / 2.0 / 100.0 AS price_frac,
                CASE WHEN result = 'yes' THEN 1.0 ELSE 0.0 END AS yes_won
            FROM '{markets_dir}/*.parquet'
            WHERE yes_bid IS NOT NULL
              AND yes_ask IS NOT NULL
              AND yes_bid > 0
              AND yes_ask > 0
              AND yes_ask > yes_bid
              AND result IN ('yes', 'no')
        )
        SELECT
            CASE
                WHEN spread <= 2 THEN 1
                WHEN spread <= 5 THEN 3.5
                WHEN spread <= 10 THEN 7.5
                WHEN spread <= 20 THEN 15
                WHEN spread <= 40 THEN 30
                ELSE 60
            END AS spread_bin,
            CASE
                WHEN spread <= 2 THEN '0-2'
                WHEN spread <= 5 THEN '2-5'
                WHEN spread <= 10 THEN '5-10'
                WHEN spread <= 20 THEN '10-20'
                WHEN spread <= 40 THEN '20-40'
                ELSE '>40'
            END AS spread_label,
            AVG(ABS(yes_won - price_frac)) AS mean_abs_error,
            AVG(yes_won - price_frac) AS bias,
            COUNT(*) AS n_markets
        FROM market_data
        GROUP BY spread_bin, spread_label
        HAVING COUNT(*) >= 100
        ORDER BY spread_bin
        """
    ).df()

    # Spearman correlation: spread vs volume (expect negative - higher volume = tighter spread)
    spearman_vol, spearman_vol_p = stats.spearmanr(
        df_by_volume["volume_bin"], df_by_volume["avg_spread"]
    )

    # Spearman correlation: spread vs accuracy (expect positive - wider spread = higher error)
    spearman_acc, spearman_acc_p = stats.spearmanr(
        df_spread_vs_accuracy["spread_bin"], df_spread_vs_accuracy["mean_abs_error"]
    )

    df_by_price.to_csv(fig_dir / "bid_ask_spread_dynamics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax1 = axes[0]
    ax1.plot(df_by_price["price_bin"], df_by_price["avg_spread"], marker="o", color="#4C72B0", linewidth=2, markersize=6)
    ax1.fill_between(df_by_price["price_bin"], df_by_price["median_spread"], df_by_price["avg_spread"], alpha=0.3, color="#4C72B0")
    ax1.set_xlabel("Mid Price (cents)")
    ax1.set_ylabel("Spread (cents)")
    ax1.set_title("Bid-Ask Spread vs Price Level")
    ax1.set_xlim(0, 100)

    ax2 = axes[1]
    ax2.scatter(df_by_volume["volume_bin"], df_by_volume["avg_spread"], s=80, color="#4C72B0", alpha=0.7, edgecolor="none")
    ax2.set_xscale("log")
    ax2.set_xlabel("Volume (contracts)")
    ax2.set_ylabel("Avg Spread (cents)")
    ax2.set_title(f"Spread vs Volume\n(ρ={spearman_vol:.3f}, p={spearman_vol_p:.2e})")

    ax3 = axes[2]
    x = np.arange(len(df_spread_vs_accuracy))
    ax3.bar(x, df_spread_vs_accuracy["mean_abs_error"] * 100, color="#4C72B0", alpha=0.7, edgecolor="none")
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_spread_vs_accuracy["spread_label"])
    ax3.set_xlabel("Spread (cents)")
    ax3.set_ylabel("Mean Absolute Error (pp)")
    ax3.set_title(f"Price Accuracy vs Spread\n(ρ={spearman_acc:.3f}, p={spearman_acc_p:.2e})")

    plt.tight_layout()
    fig.savefig(fig_dir / "bid_ask_spread_dynamics.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "bid_ask_spread_dynamics.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")
    print(f"\nStatistical Summary:")
    print(f"  Spread vs Volume: Spearman ρ={spearman_vol:.3f}, p={spearman_vol_p:.2e}")
    print(f"  Spread vs Error:  Spearman ρ={spearman_acc:.3f}, p={spearman_acc_p:.2e}")


if __name__ == "__main__":
    main()
