#!/usr/bin/env python3
"""Analyze maker vs taker returns by market category.

Tests whether the maker/taker gap varies by market category. Hypothesis:
the gap should be larger in categories with more retail participation (sports)
and smaller in categories with more sophisticated participants (finance).
"""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from util.categories import CATEGORY_SQL, GROUP_COLORS, get_group


def main():
    base_dir = Path(__file__).parent.parent.parent
    trades_dir = base_dir / "data" / "trades"
    markets_dir = base_dir / "data" / "markets"
    fig_dir = base_dir / "research" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Get taker and maker returns by category
    df = con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker, event_ticker, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
        ),
        taker_positions AS (
            SELECT
                {CATEGORY_SQL.replace('event_ticker', 'm.event_ticker')} AS category,
                CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                t.count AS contracts,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
        ),
        maker_positions AS (
            SELECT
                {CATEGORY_SQL.replace('event_ticker', 'm.event_ticker')} AS category,
                CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                CASE WHEN t.taker_side != m.result THEN 1.0 ELSE 0.0 END AS won,
                t.count AS contracts,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
        ),
        taker_stats AS (
            SELECT
                category,
                AVG(won) AS win_rate,
                AVG(price / 100.0) AS avg_price,
                AVG(won - price / 100.0) AS excess_return,
                VAR_POP(won - price / 100.0) AS var_excess,
                COUNT(*) AS n_trades,
                SUM(contracts) AS contracts,
                SUM(volume_usd) AS volume_usd,
                SUM(contracts * (won - price / 100.0)) AS pnl
            FROM taker_positions
            GROUP BY category
        ),
        maker_stats AS (
            SELECT
                category,
                AVG(won) AS win_rate,
                AVG(price / 100.0) AS avg_price,
                AVG(won - price / 100.0) AS excess_return,
                VAR_POP(won - price / 100.0) AS var_excess,
                COUNT(*) AS n_trades,
                SUM(contracts) AS contracts,
                SUM(volume_usd) AS volume_usd,
                SUM(contracts * (won - price / 100.0)) AS pnl
            FROM maker_positions
            GROUP BY category
        )
        SELECT
            t.category,
            t.win_rate AS taker_win_rate,
            t.avg_price AS taker_avg_price,
            t.excess_return AS taker_excess,
            t.var_excess AS taker_var,
            t.n_trades AS taker_n,
            t.contracts AS taker_contracts,
            t.volume_usd AS taker_volume,
            t.pnl AS taker_pnl,
            m.win_rate AS maker_win_rate,
            m.avg_price AS maker_avg_price,
            m.excess_return AS maker_excess,
            m.var_excess AS maker_var,
            m.n_trades AS maker_n,
            m.contracts AS maker_contracts,
            m.volume_usd AS maker_volume,
            m.pnl AS maker_pnl
        FROM taker_stats t
        JOIN maker_stats m ON t.category = m.category
        ORDER BY t.volume_usd DESC
        """
    ).df()

    # Apply group mapping
    df["group"] = df["category"].apply(get_group)

    # Aggregate by group
    group_stats = []
    for group in df["group"].unique():
        group_df = df[df["group"] == group]

        # Volume-weighted excess returns
        taker_vol_weighted = (group_df["taker_excess"] * group_df["taker_contracts"]).sum() / group_df["taker_contracts"].sum()
        maker_vol_weighted = (group_df["maker_excess"] * group_df["maker_contracts"]).sum() / group_df["maker_contracts"].sum()

        group_stats.append({
            "group": group,
            "taker_excess": taker_vol_weighted * 100,
            "maker_excess": maker_vol_weighted * 100,
            "gap": (maker_vol_weighted - taker_vol_weighted) * 100,
            "taker_n": int(group_df["taker_n"].sum()),
            "maker_n": int(group_df["maker_n"].sum()),
            "taker_volume": group_df["taker_volume"].sum(),
            "maker_volume": group_df["maker_volume"].sum(),
            "taker_pnl": group_df["taker_pnl"].sum(),
            "maker_pnl": group_df["maker_pnl"].sum(),
        })

    group_df = pd.DataFrame(group_stats)
    group_df = group_df.sort_values("taker_volume", ascending=False)

    df.to_csv(fig_dir / "maker_taker_returns_by_category.csv", index=False)
    group_df.to_csv(fig_dir / "maker_taker_returns_by_group.csv", index=False)

    # Figure 1: Bar chart by group
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    top_groups = group_df.head(8)
    x = np.arange(len(top_groups))
    width = 0.35
    ax1.bar(x - width / 2, top_groups["taker_excess"], width, label="Taker", color="#e74c3c", alpha=0.8)
    ax1.bar(x + width / 2, top_groups["maker_excess"], width, label="Maker", color="#2ecc71", alpha=0.8)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Category")
    ax1.set_ylabel("Volume-Weighted Excess Return (pp)")
    ax1.set_title("Maker vs Taker Returns by Category")
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_groups["group"], rotation=45, ha="right")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig1.savefig(fig_dir / "maker_taker_returns_by_group.png", dpi=300, bbox_inches="tight")
    fig1.savefig(fig_dir / "maker_taker_returns_by_group.pdf", bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Scatter plot of taker vs maker returns
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    colors = [GROUP_COLORS.get(g, "#888888") for g in top_groups["group"]]
    sizes = np.log10(top_groups["taker_volume"] + 1) * 50
    scatter = ax2.scatter(top_groups["taker_excess"], top_groups["maker_excess"], c=colors, s=sizes, alpha=0.7, edgecolors="black", linewidth=0.5)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax2.plot([-5, 5], [5, -5], "k:", alpha=0.3, label="Zero-sum line")
    for i, row in top_groups.iterrows():
        ax2.annotate(row["group"], (row["taker_excess"], row["maker_excess"]), fontsize=8, ha="center", va="bottom")
    ax2.set_xlabel("Taker Excess Return (pp)")
    ax2.set_ylabel("Maker Excess Return (pp)")
    ax2.set_title("Maker vs Taker Returns by Category\n(size = log volume)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(fig_dir / "maker_taker_scatter_by_group.png", dpi=300, bbox_inches="tight")
    fig2.savefig(fig_dir / "maker_taker_scatter_by_group.pdf", bbox_inches="tight")
    plt.close(fig2)

    # JSON output
    json_data = {
        "type": "bar",
        "title": "Maker vs Taker Returns by Category",
        "data": [
            {
                "category": row["group"],
                "Taker Return": round(row["taker_excess"], 2),
                "Maker Return": round(row["maker_excess"], 2),
                "Gap": round(row["gap"], 2),
                "N trades": int(row["taker_n"]),
            }
            for _, row in top_groups.iterrows()
        ],
        "xKey": "category",
        "yKeys": ["Taker Return", "Maker Return"],
        "yUnit": "percent",
    }
    with open(fig_dir / "maker_taker_returns_by_group.json", "w") as f:
        json.dump(json_data, f)

    # Print summary
    print(f"Outputs saved to {fig_dir}")
    print("\n" + "=" * 70)
    print("MAKER VS TAKER RETURNS BY CATEGORY")
    print("=" * 70)

    print("\nBy category group (sorted by volume):")
    print("-" * 70)
    print(f"{'Category':<15} {'Taker (pp)':>12} {'Maker (pp)':>12} {'Gap (pp)':>10} {'N trades':>12}")
    print("-" * 70)
    for _, row in group_df.iterrows():
        print(f"{row['group']:<15} {row['taker_excess']:>+12.2f} {row['maker_excess']:>+12.2f} {row['gap']:>10.2f} {row['taker_n']:>12,}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Categories with largest maker-taker gap:")
    top_gap = group_df.nlargest(3, "gap")
    for _, row in top_gap.iterrows():
        print(f"  {row['group']}: {row['gap']:.2f} pp")

    print(f"\nCategories with smallest maker-taker gap:")
    bottom_gap = group_df.nsmallest(3, "gap")
    for _, row in bottom_gap.iterrows():
        print(f"  {row['group']}: {row['gap']:.2f} pp")


if __name__ == "__main__":
    main()
