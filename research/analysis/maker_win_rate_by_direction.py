#!/usr/bin/env python3
"""Analyze maker win rate by position direction (YES vs NO).

Focused analysis for Section 6.4 of the paper. Tests whether makers who
buy NO outperform makers who buy YES at equivalent prices, confirming
selective positioning rather than passive accommodation.
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

    # Get maker win rates by direction at each price
    df = con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
        ),
        maker_yes_positions AS (
            -- Maker bought YES (taker sold YES = taker bought NO)
            SELECT
                t.yes_price AS price,
                CASE WHEN m.result = 'yes' THEN 1.0 ELSE 0.0 END AS won,
                t.count AS contracts,
                'YES' AS maker_side
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
            WHERE t.taker_side = 'no'
        ),
        maker_no_positions AS (
            -- Maker bought NO (taker sold NO = taker bought YES)
            SELECT
                t.no_price AS price,
                CASE WHEN m.result = 'no' THEN 1.0 ELSE 0.0 END AS won,
                t.count AS contracts,
                'NO' AS maker_side
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
            WHERE t.taker_side = 'yes'
        ),
        all_maker_positions AS (
            SELECT * FROM maker_yes_positions
            UNION ALL
            SELECT * FROM maker_no_positions
        )
        SELECT
            maker_side,
            price,
            SUM(won * contracts) / SUM(contracts) AS win_rate,
            price / 100.0 AS implied_prob,
            SUM(won * contracts) / SUM(contracts) - price / 100.0 AS mispricing,
            COUNT(*) AS n_trades,
            SUM(contracts) AS contracts
        FROM all_maker_positions
        WHERE price BETWEEN 1 AND 99
        GROUP BY maker_side, price
        ORDER BY maker_side, price
        """
    ).df()

    # Pivot for comparison
    df_yes = df[df["maker_side"] == "YES"][["price", "win_rate", "mispricing", "n_trades", "contracts"]].copy()
    df_yes = df_yes.rename(columns={
        "win_rate": "yes_win_rate",
        "mispricing": "yes_mispricing",
        "n_trades": "yes_n",
        "contracts": "yes_contracts",
    })

    df_no = df[df["maker_side"] == "NO"][["price", "win_rate", "mispricing", "n_trades", "contracts"]].copy()
    df_no = df_no.rename(columns={
        "win_rate": "no_win_rate",
        "mispricing": "no_mispricing",
        "n_trades": "no_n",
        "contracts": "no_contracts",
    })

    comparison = pd.merge(df_yes, df_no, on="price", how="outer")
    comparison["implied_prob"] = comparison["price"] / 100.0
    comparison = comparison.sort_values("price")

    comparison.to_csv(fig_dir / "maker_win_rate_by_direction.csv", index=False)

    # Table for specific price ranges (Section 6.4)
    ranges = [
        (1, 10, "1-10c (longshots)"),
        (91, 99, "91-99c (favorites)"),
    ]

    range_stats = []
    for low, high, label in ranges:
        subset = comparison[(comparison["price"] >= low) & (comparison["price"] <= high)].copy()
        if len(subset) == 0:
            continue

        # Contract-weighted win rates
        yes_win_rate = (subset["yes_win_rate"] * subset["yes_contracts"]).sum() / subset["yes_contracts"].sum()
        no_win_rate = (subset["no_win_rate"] * subset["no_contracts"]).sum() / subset["no_contracts"].sum()
        implied = (subset["implied_prob"] * subset["yes_contracts"]).sum() / subset["yes_contracts"].sum()

        range_stats.append({
            "range": label,
            "implied_prob": implied * 100,
            "maker_yes_win_rate": yes_win_rate * 100,
            "maker_no_win_rate": no_win_rate * 100,
            "yes_mispricing": (yes_win_rate - implied) * 100,
            "no_mispricing": (no_win_rate - implied) * 100,
            "yes_n": int(subset["yes_n"].sum()),
            "no_n": int(subset["no_n"].sum()),
        })

    range_df = pd.DataFrame(range_stats)
    range_df.to_csv(fig_dir / "maker_win_rate_by_direction_ranges.csv", index=False)

    # Figure 1: Win rate comparison at price extremes
    fig1, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Longshots (1-10c)
    ax1 = axes[0]
    longshot_data = comparison[(comparison["price"] >= 1) & (comparison["price"] <= 10)]
    ax1.plot(longshot_data["price"], longshot_data["yes_win_rate"] * 100, "o-", color="#2ecc71", linewidth=2, markersize=6, label="Maker bought YES")
    ax1.plot(longshot_data["price"], longshot_data["no_win_rate"] * 100, "s-", color="#e74c3c", linewidth=2, markersize=6, label="Maker bought NO")
    ax1.plot(longshot_data["price"], longshot_data["implied_prob"] * 100, "k--", linewidth=1.5, alpha=0.7, label="Implied probability")
    ax1.set_xlabel("Price (cents)")
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_title("Maker Win Rate at Longshot Prices (1-10c)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 11)

    # Favorites (91-99c)
    ax2 = axes[1]
    fav_data = comparison[(comparison["price"] >= 91) & (comparison["price"] <= 99)]
    ax2.plot(fav_data["price"], fav_data["yes_win_rate"] * 100, "o-", color="#2ecc71", linewidth=2, markersize=6, label="Maker bought YES")
    ax2.plot(fav_data["price"], fav_data["no_win_rate"] * 100, "s-", color="#e74c3c", linewidth=2, markersize=6, label="Maker bought NO")
    ax2.plot(fav_data["price"], fav_data["implied_prob"] * 100, "k--", linewidth=1.5, alpha=0.7, label="Implied probability")
    ax2.set_xlabel("Price (cents)")
    ax2.set_ylabel("Win Rate (%)")
    ax2.set_title("Maker Win Rate at Favorite Prices (91-99c)")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(90, 100)

    plt.tight_layout()
    fig1.savefig(fig_dir / "maker_win_rate_by_direction.png", dpi=300, bbox_inches="tight")
    fig1.savefig(fig_dir / "maker_win_rate_by_direction.pdf", bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Full price range comparison
    fig2, ax = plt.subplots(figsize=(12, 7))
    ax.plot(comparison["price"], comparison["yes_win_rate"] * 100, color="#2ecc71", linewidth=1.5, label="Maker bought YES", alpha=0.8)
    ax.plot(comparison["price"], comparison["no_win_rate"] * 100, color="#e74c3c", linewidth=1.5, label="Maker bought NO", alpha=0.8)
    ax.plot(comparison["price"], comparison["implied_prob"] * 100, "k--", linewidth=1.5, alpha=0.7, label="Implied probability")
    ax.set_xlabel("Maker's Purchase Price (cents)")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Maker Win Rate by Position Direction")
    ax.set_xlim(1, 99)
    ax.set_xticks(range(0, 101, 10))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(fig_dir / "maker_win_rate_by_direction_full.png", dpi=300, bbox_inches="tight")
    fig2.savefig(fig_dir / "maker_win_rate_by_direction_full.pdf", bbox_inches="tight")
    plt.close(fig2)

    # JSON output
    json_data = {
        "type": "line",
        "title": "Maker Win Rate by Position Direction",
        "data": [
            {
                "price": int(row["price"]),
                "Maker bought YES": round(row["yes_win_rate"] * 100, 2) if pd.notna(row["yes_win_rate"]) else None,
                "Maker bought NO": round(row["no_win_rate"] * 100, 2) if pd.notna(row["no_win_rate"]) else None,
                "Implied probability": round(row["implied_prob"] * 100, 2),
            }
            for _, row in comparison.iterrows()
        ],
        "xKey": "price",
        "yKeys": ["Maker bought YES", "Maker bought NO", "Implied probability"],
        "yUnit": "percent",
    }
    with open(fig_dir / "maker_win_rate_by_direction.json", "w") as f:
        json.dump(json_data, f)

    # Print summary for paper tables
    print(f"Outputs saved to {fig_dir}")
    print("\n" + "=" * 70)
    print("MAKER WIN RATE BY POSITION DIRECTION")
    print("=" * 70)

    print("\nTable for Section 6.4:")
    print("-" * 80)
    print(f"{'Price Range':<20} {'Implied (%)':<12} {'Maker YES (%)':<14} {'Maker NO (%)':<14} {'N (YES)':<12} {'N (NO)':<12}")
    print("-" * 80)
    for _, row in range_df.iterrows():
        print(f"{row['range']:<20} {row['implied_prob']:<12.1f} {row['maker_yes_win_rate']:<14.2f} {row['maker_no_win_rate']:<14.2f} {int(row['yes_n']):<12,} {int(row['no_n']):<12,}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # At 1-10c: makers who buy NO (when YES is cheap) should outperform
    # At 91-99c: makers who buy YES (when NO is cheap) should outperform
    longshot_yes = range_df[range_df["range"].str.contains("1-10c")]["maker_yes_win_rate"].values[0]
    longshot_no = range_df[range_df["range"].str.contains("1-10c")]["maker_no_win_rate"].values[0]
    fav_yes = range_df[range_df["range"].str.contains("91-99c")]["maker_yes_win_rate"].values[0]
    fav_no = range_df[range_df["range"].str.contains("91-99c")]["maker_no_win_rate"].values[0]

    print(f"\nAt longshot prices (1-10c):")
    print(f"  Maker buying YES wins: {longshot_yes:.2f}%")
    print(f"  Maker buying NO wins: {longshot_no:.2f}%")
    print(f"  Difference: {longshot_no - longshot_yes:+.2f} pp (NO {'outperforms' if longshot_no > longshot_yes else 'underperforms'})")

    print(f"\nAt favorite prices (91-99c):")
    print(f"  Maker buying YES wins: {fav_yes:.2f}%")
    print(f"  Maker buying NO wins: {fav_no:.2f}%")
    print(f"  Difference: {fav_yes - fav_no:+.2f} pp (YES {'outperforms' if fav_yes > fav_no else 'underperforms'})")

    # Count price levels where NO outperforms at tails
    low_prices = comparison[(comparison["price"] >= 1) & (comparison["price"] <= 10)]
    high_prices = comparison[(comparison["price"] >= 91) & (comparison["price"] <= 99)]
    no_better_low = (low_prices["no_win_rate"] > low_prices["yes_win_rate"]).sum()
    yes_better_high = (high_prices["yes_win_rate"] > high_prices["no_win_rate"]).sum()

    print(f"\nAt 1-10c: Maker NO outperforms Maker YES at {no_better_low}/{len(low_prices)} price points")
    print(f"At 91-99c: Maker YES outperforms Maker NO at {yes_better_high}/{len(high_prices)} price points")


if __name__ == "__main__":
    main()
