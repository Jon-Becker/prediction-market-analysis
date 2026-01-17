#!/usr/bin/env python3
"""Analyze trade size distribution by role (maker vs taker).

Tests whether makers place larger trades than takers on average,
which would be consistent with sophisticated, well-capitalized participants.
Section 7.3 of the paper.
"""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def main():
    base_dir = Path(__file__).parent.parent.parent
    trades_dir = base_dir / "data" / "trades"
    markets_dir = base_dir / "data" / "markets"
    fig_dir = base_dir / "research" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Get aggregate trade size stats for takers and makers
    aggregate_stats = con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
        ),
        taker_trades AS (
            SELECT
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS trade_size_usd,
                t.count AS contracts
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
        ),
        maker_trades AS (
            SELECT
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END) / 100.0 AS trade_size_usd,
                t.count AS contracts
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
        )
        SELECT
            'taker' AS role,
            AVG(trade_size_usd) AS mean_trade_size,
            MEDIAN(trade_size_usd) AS median_trade_size,
            STDDEV_POP(trade_size_usd) AS std_trade_size,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY trade_size_usd) AS p25_trade_size,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY trade_size_usd) AS p75_trade_size,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY trade_size_usd) AS p90_trade_size,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY trade_size_usd) AS p95_trade_size,
            AVG(contracts) AS mean_contracts,
            MEDIAN(contracts) AS median_contracts,
            COUNT(*) AS n_trades,
            SUM(trade_size_usd) AS total_volume
        FROM taker_trades

        UNION ALL

        SELECT
            'maker' AS role,
            AVG(trade_size_usd) AS mean_trade_size,
            MEDIAN(trade_size_usd) AS median_trade_size,
            STDDEV_POP(trade_size_usd) AS std_trade_size,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY trade_size_usd) AS p25_trade_size,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY trade_size_usd) AS p75_trade_size,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY trade_size_usd) AS p90_trade_size,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY trade_size_usd) AS p95_trade_size,
            AVG(contracts) AS mean_contracts,
            MEDIAN(contracts) AS median_contracts,
            COUNT(*) AS n_trades,
            SUM(trade_size_usd) AS total_volume
        FROM maker_trades
        """
    ).df()

    # Get trade size distribution by buckets
    bucket_df = con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
        ),
        all_trades AS (
            SELECT
                'taker' AS role,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS trade_size_usd,
                t.count AS contracts
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker

            UNION ALL

            SELECT
                'maker' AS role,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END) / 100.0 AS trade_size_usd,
                t.count AS contracts
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
        )
        SELECT
            role,
            CASE
                WHEN trade_size_usd < 1 THEN '<$1'
                WHEN trade_size_usd < 10 THEN '$1-$10'
                WHEN trade_size_usd < 100 THEN '$10-$100'
                WHEN trade_size_usd < 1000 THEN '$100-$1K'
                WHEN trade_size_usd < 10000 THEN '$1K-$10K'
                ELSE '>$10K'
            END AS bucket,
            CASE
                WHEN trade_size_usd < 1 THEN 1
                WHEN trade_size_usd < 10 THEN 2
                WHEN trade_size_usd < 100 THEN 3
                WHEN trade_size_usd < 1000 THEN 4
                WHEN trade_size_usd < 10000 THEN 5
                ELSE 6
            END AS bucket_order,
            COUNT(*) AS n_trades,
            SUM(trade_size_usd) AS volume_usd
        FROM all_trades
        GROUP BY role, bucket, bucket_order
        ORDER BY role, bucket_order
        """
    ).df()

    aggregate_stats.to_csv(fig_dir / "trade_size_by_role.csv", index=False)
    bucket_df.to_csv(fig_dir / "trade_size_distribution_by_role.csv", index=False)

    # Figure 1: Comparison bar chart
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(2)
    width = 0.35

    mean_sizes = aggregate_stats.set_index("role")["mean_trade_size"]
    median_sizes = aggregate_stats.set_index("role")["median_trade_size"]

    ax1.bar(x - width / 2, [mean_sizes["taker"], mean_sizes["maker"]], width, label="Mean", color="#3498db", alpha=0.8)
    ax1.bar(x + width / 2, [median_sizes["taker"], median_sizes["maker"]], width, label="Median", color="#e74c3c", alpha=0.8)
    ax1.set_ylabel("Trade Size (USD)")
    ax1.set_title("Trade Size by Role: Mean vs Median")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Taker", "Maker"])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (mean, median) in enumerate(zip([mean_sizes["taker"], mean_sizes["maker"]],
                                            [median_sizes["taker"], median_sizes["maker"]])):
        ax1.annotate(f"${mean:.0f}", (i - width / 2, mean), ha="center", va="bottom", fontsize=9)
        ax1.annotate(f"${median:.0f}", (i + width / 2, median), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig1.savefig(fig_dir / "trade_size_by_role.png", dpi=300, bbox_inches="tight")
    fig1.savefig(fig_dir / "trade_size_by_role.pdf", bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Distribution comparison
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    taker_buckets = bucket_df[bucket_df["role"] == "taker"].sort_values("bucket_order")
    maker_buckets = bucket_df[bucket_df["role"] == "maker"].sort_values("bucket_order")

    buckets = taker_buckets["bucket"].tolist()
    x = np.arange(len(buckets))
    width = 0.35

    taker_pcts = (taker_buckets["n_trades"] / taker_buckets["n_trades"].sum() * 100).tolist()
    maker_pcts = (maker_buckets["n_trades"] / maker_buckets["n_trades"].sum() * 100).tolist()

    ax2.bar(x - width / 2, taker_pcts, width, label="Taker", color="#e74c3c", alpha=0.8)
    ax2.bar(x + width / 2, maker_pcts, width, label="Maker", color="#2ecc71", alpha=0.8)
    ax2.set_ylabel("% of Trades")
    ax2.set_xlabel("Trade Size Bucket")
    ax2.set_title("Trade Size Distribution by Role")
    ax2.set_xticks(x)
    ax2.set_xticklabels(buckets, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig2.savefig(fig_dir / "trade_size_distribution_by_role.png", dpi=300, bbox_inches="tight")
    fig2.savefig(fig_dir / "trade_size_distribution_by_role.pdf", bbox_inches="tight")
    plt.close(fig2)

    # JSON output
    json_data = {
        "type": "table",
        "title": "Trade Size Statistics by Role",
        "data": [
            {
                "Role": row["role"].title(),
                "Mean Trade Size": f"${row['mean_trade_size']:.2f}",
                "Median Trade Size": f"${row['median_trade_size']:.2f}",
                "N Trades": f"{int(row['n_trades']):,}",
            }
            for _, row in aggregate_stats.iterrows()
        ],
    }
    with open(fig_dir / "trade_size_by_role.json", "w") as f:
        json.dump(json_data, f)

    # Print summary table for Section 7.3
    print(f"Outputs saved to {fig_dir}")
    print("\n" + "=" * 70)
    print("TRADE SIZE STATISTICS BY ROLE")
    print("=" * 70)

    print("\nTable for Section 7.3:")
    print("-" * 70)
    taker_row = aggregate_stats[aggregate_stats["role"] == "taker"].iloc[0]
    maker_row = aggregate_stats[aggregate_stats["role"] == "maker"].iloc[0]

    print(f"{'Metric':<25} {'Taker':>20} {'Maker':>20}")
    print("-" * 70)
    print(f"{'Mean Trade Size':<25} ${taker_row['mean_trade_size']:>19,.2f} ${maker_row['mean_trade_size']:>19,.2f}")
    print(f"{'Median Trade Size':<25} ${taker_row['median_trade_size']:>19,.2f} ${maker_row['median_trade_size']:>19,.2f}")
    print(f"{'Std Dev':<25} ${taker_row['std_trade_size']:>19,.2f} ${maker_row['std_trade_size']:>19,.2f}")
    print(f"{'25th Percentile':<25} ${taker_row['p25_trade_size']:>19,.2f} ${maker_row['p25_trade_size']:>19,.2f}")
    print(f"{'75th Percentile':<25} ${taker_row['p75_trade_size']:>19,.2f} ${maker_row['p75_trade_size']:>19,.2f}")
    print(f"{'90th Percentile':<25} ${taker_row['p90_trade_size']:>19,.2f} ${maker_row['p90_trade_size']:>19,.2f}")
    print(f"{'95th Percentile':<25} ${taker_row['p95_trade_size']:>19,.2f} ${maker_row['p95_trade_size']:>19,.2f}")
    print(f"{'Mean Contracts':<25} {taker_row['mean_contracts']:>20,.1f} {maker_row['mean_contracts']:>20,.1f}")
    print(f"{'Median Contracts':<25} {taker_row['median_contracts']:>20,.0f} {maker_row['median_contracts']:>20,.0f}")
    print(f"{'N Trades':<25} {int(taker_row['n_trades']):>20,} {int(maker_row['n_trades']):>20,}")
    print(f"{'Total Volume':<25} ${taker_row['total_volume']/1e9:>18,.2f}B ${maker_row['total_volume']/1e9:>18,.2f}B")

    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    ratio = maker_row["mean_trade_size"] / taker_row["mean_trade_size"]
    print(f"\nMaker mean trade size / Taker mean trade size = {ratio:.2f}x")
    if ratio > 1:
        print("Makers place larger trades on average, consistent with sophisticated, well-capitalized participants.")
    else:
        print("Takers place larger trades on average, contrary to the sophistication hypothesis.")

    print("\n" + "=" * 70)
    print("TRADE SIZE DISTRIBUTION")
    print("=" * 70)
    print("\nShare of trades by size bucket:")
    print("-" * 50)
    print(f"{'Bucket':<15} {'Taker %':>15} {'Maker %':>15}")
    print("-" * 50)
    for bucket in buckets:
        taker_pct = taker_buckets[taker_buckets["bucket"] == bucket]["n_trades"].values[0] / taker_buckets["n_trades"].sum() * 100
        maker_pct = maker_buckets[maker_buckets["bucket"] == bucket]["n_trades"].values[0] / maker_buckets["n_trades"].sum() * 100
        print(f"{bucket:<15} {taker_pct:>14.1f}% {maker_pct:>14.1f}%")


if __name__ == "__main__":
    main()
