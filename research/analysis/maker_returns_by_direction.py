#!/usr/bin/env python3
"""Analyze maker returns by position direction (YES vs NO).

Tests whether maker profits are purely spread compensation or reflect
directional alpha by comparing maker performance when buying YES vs NO.
If makers systematically outperform on NO positions, this suggests selective
positioning rather than passive accommodation.
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

    # Maker bought YES when taker_side = 'no' (maker is counterparty)
    # Maker bought NO when taker_side = 'yes'
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
            AVG(won) AS win_rate,
            price / 100.0 AS expected_win_rate,
            AVG(won) - price / 100.0 AS excess_return,
            VAR_POP(won - price / 100.0) AS var_excess,
            COUNT(*) AS n_trades,
            SUM(contracts) AS contracts,
            SUM(contracts * price / 100.0) AS volume_usd
        FROM all_maker_positions
        WHERE price BETWEEN 1 AND 99
        GROUP BY maker_side, price
        ORDER BY maker_side, price
        """
    ).df()

    # Pivot to compare YES vs NO at each price
    df_yes = df[df["maker_side"] == "YES"].copy()
    df_no = df[df["maker_side"] == "NO"].copy()

    # Merge on price
    comparison = pd.merge(
        df_yes[["price", "win_rate", "excess_return", "n_trades", "contracts", "volume_usd"]].rename(
            columns={
                "win_rate": "yes_win_rate",
                "excess_return": "yes_excess",
                "n_trades": "yes_n",
                "contracts": "yes_contracts",
                "volume_usd": "yes_volume",
            }
        ),
        df_no[["price", "win_rate", "excess_return", "n_trades", "contracts", "volume_usd"]].rename(
            columns={
                "win_rate": "no_win_rate",
                "excess_return": "no_excess",
                "n_trades": "no_n",
                "contracts": "no_contracts",
                "volume_usd": "no_volume",
            }
        ),
        on="price",
        how="outer",
    )
    comparison = comparison.sort_values("price")
    comparison["diff"] = comparison["no_excess"] - comparison["yes_excess"]

    comparison.to_csv(fig_dir / "maker_returns_by_direction.csv", index=False)

    # Aggregate by price range
    ranges = [
        (1, 10, "1-10c"),
        (11, 25, "11-25c"),
        (26, 50, "26-50c"),
        (51, 75, "51-75c"),
        (76, 90, "76-90c"),
        (91, 99, "91-99c"),
    ]

    range_stats = []
    for low, high, label in ranges:
        subset = comparison[(comparison["price"] >= low) & (comparison["price"] <= high)]
        if len(subset) == 0:
            continue

        # Volume-weighted excess return
        yes_vol_weighted = (subset["yes_excess"] * subset["yes_contracts"]).sum() / subset["yes_contracts"].sum()
        no_vol_weighted = (subset["no_excess"] * subset["no_contracts"]).sum() / subset["no_contracts"].sum()

        range_stats.append({
            "range": label,
            "yes_excess": yes_vol_weighted * 100,  # Convert to percentage points
            "no_excess": no_vol_weighted * 100,
            "diff": (no_vol_weighted - yes_vol_weighted) * 100,
            "yes_n": int(subset["yes_n"].sum()),
            "no_n": int(subset["no_n"].sum()),
        })

    range_df = pd.DataFrame(range_stats)
    range_df.to_csv(fig_dir / "maker_returns_by_direction_ranges.csv", index=False)

    # Figure 1: Maker excess returns by direction and price
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(comparison["price"], comparison["yes_excess"] * 100, color="#2ecc71", linewidth=1.5, label="Maker bought YES", alpha=0.8)
    ax1.plot(comparison["price"], comparison["no_excess"] * 100, color="#e74c3c", linewidth=1.5, label="Maker bought NO", alpha=0.8)
    ax1.fill_between(comparison["price"], comparison["yes_excess"] * 100, alpha=0.2, color="#2ecc71")
    ax1.fill_between(comparison["price"], comparison["no_excess"] * 100, alpha=0.2, color="#e74c3c")
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Maker's Purchase Price (cents)")
    ax1.set_ylabel("Excess Return (pp)")
    ax1.set_title("Maker Excess Returns by Position Direction")
    ax1.set_xlim(1, 99)
    ax1.set_xticks(range(0, 101, 10))
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(fig_dir / "maker_returns_by_direction.png", dpi=300, bbox_inches="tight")
    fig1.savefig(fig_dir / "maker_returns_by_direction.pdf", bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Bar chart by price range
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(range_df))
    width = 0.35
    ax2.bar(x - width / 2, range_df["yes_excess"], width, label="Maker bought YES", color="#2ecc71", alpha=0.8)
    ax2.bar(x + width / 2, range_df["no_excess"], width, label="Maker bought NO", color="#e74c3c", alpha=0.8)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Price Range")
    ax2.set_ylabel("Volume-Weighted Excess Return (pp)")
    ax2.set_title("Maker Excess Returns by Direction and Price Range")
    ax2.set_xticks(x)
    ax2.set_xticklabels(range_df["range"])
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig2.savefig(fig_dir / "maker_returns_by_direction_ranges.png", dpi=300, bbox_inches="tight")
    fig2.savefig(fig_dir / "maker_returns_by_direction_ranges.pdf", bbox_inches="tight")
    plt.close(fig2)

    # JSON output for paper
    json_data = {
        "type": "line",
        "title": "Maker Excess Returns by Position Direction",
        "data": [
            {
                "price": int(row["price"]),
                "Maker bought YES": round(row["yes_excess"] * 100, 2) if pd.notna(row["yes_excess"]) else None,
                "Maker bought NO": round(row["no_excess"] * 100, 2) if pd.notna(row["no_excess"]) else None,
            }
            for _, row in comparison.iterrows()
        ],
        "xKey": "price",
        "yKeys": ["Maker bought YES", "Maker bought NO"],
        "yUnit": "percent",
    }
    with open(fig_dir / "maker_returns_by_direction.json", "w") as f:
        json.dump(json_data, f)

    # Print summary
    print(f"Outputs saved to {fig_dir}")
    print("\n" + "=" * 70)
    print("MAKER RETURNS BY POSITION DIRECTION")
    print("=" * 70)

    print("\nVolume-weighted excess returns by price range (pp):")
    print(range_df.to_string(index=False))

    # Overall comparison
    total_yes_excess = (comparison["yes_excess"] * comparison["yes_contracts"]).sum() / comparison["yes_contracts"].sum()
    total_no_excess = (comparison["no_excess"] * comparison["no_contracts"]).sum() / comparison["no_contracts"].sum()

    print(f"\nOverall volume-weighted excess return:")
    print(f"  Maker bought YES: {total_yes_excess * 100:+.2f} pp")
    print(f"  Maker bought NO: {total_no_excess * 100:+.2f} pp")
    print(f"  Difference (NO - YES): {(total_no_excess - total_yes_excess) * 100:+.2f} pp")

    # Count price levels where NO outperforms YES
    no_better = (comparison["no_excess"] > comparison["yes_excess"]).sum()
    total_prices = len(comparison)
    print(f"\nPrice levels where Maker NO outperforms Maker YES: {no_better}/{total_prices} ({no_better / total_prices * 100:.1f}%)")


if __name__ == "__main__":
    main()
