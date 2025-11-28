#!/usr/bin/env python3
"""Analyze price clustering in prediction market prices.

Examines trade concentration by price level, including round numbers
(10, 25, 50, 75, 90) and extreme prices (1, 99). Analysis reveals that
extreme boundary prices cluster more than round numbers.
Includes chi-square tests for statistical significance of clustering.
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

    df_by_price = con.execute(
        f"""
        SELECT
            CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END AS price,
            COUNT(*) AS n_trades,
            SUM(count) AS total_contracts,
            SUM(count * (CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) / 100.0) AS volume_usd
        FROM '{trades_dir}/*.parquet'
        WHERE (CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) BETWEEN 1 AND 99
        GROUP BY price
        ORDER BY price
        """
    ).df()

    round_numbers = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    key_round = [10, 25, 50, 75, 90]
    extreme_prices = [1, 2, 3, 97, 98, 99]

    df_by_price["is_round_5"] = df_by_price["price"].isin(round_numbers)
    df_by_price["is_key_round"] = df_by_price["price"].isin(key_round)
    df_by_price["is_extreme"] = df_by_price["price"].isin(extreme_prices)

    total_trades = df_by_price["n_trades"].sum()
    expected_per_price = total_trades / 99

    df_by_price["trade_ratio"] = df_by_price["n_trades"] / expected_per_price

    df_clustering = con.execute(
        f"""
        WITH trade_prices AS (
            SELECT
                CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END AS price,
                count AS contracts,
                count * (CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet'
        )
        SELECT
            CASE
                WHEN price IN (1, 2, 3, 97, 98, 99) THEN 'Extreme (1-3, 97-99)'
                WHEN price IN (10, 25, 50, 75, 90) THEN 'Key Round'
                WHEN price % 5 = 0 THEN 'Round (÷5)'
                ELSE 'Non-Round'
            END AS price_type,
            COUNT(*) AS n_trades,
            SUM(contracts) AS total_contracts,
            SUM(volume_usd) AS volume_usd,
            AVG(volume_usd) AS avg_trade_size
        FROM trade_prices
        WHERE price BETWEEN 1 AND 99
        GROUP BY price_type
        """
    ).df()

    n_extreme = 6
    n_key = 5
    n_round = 19 - 5
    n_non = 99 - 19 - 6

    df_clustering["n_prices"] = df_clustering["price_type"].map({
        "Extreme (1-3, 97-99)": n_extreme,
        "Key Round": n_key,
        "Round (÷5)": n_round,
        "Non-Round": n_non
    })
    df_clustering["trades_per_price"] = df_clustering["n_trades"] / df_clustering["n_prices"]
    df_clustering["volume_per_price"] = df_clustering["volume_usd"] / df_clustering["n_prices"]

    overall_trades_per_price = df_clustering["n_trades"].sum() / 99
    df_clustering["clustering_ratio"] = df_clustering["trades_per_price"] / overall_trades_per_price

    # Chi-square test: H0: trades are uniformly distributed across price types
    # Expected count proportional to number of prices in each category
    observed = df_clustering["n_trades"].values
    total_n = observed.sum()
    n_prices_per_type = df_clustering["n_prices"].values
    expected = total_n * n_prices_per_type / n_prices_per_type.sum()
    chi2_stat, chi2_p = stats.chisquare(observed, expected)

    # Z-test for each price type vs uniform expectation
    df_clustering["expected_trades"] = expected
    df_clustering["z_stat"] = (df_clustering["n_trades"] - df_clustering["expected_trades"]) / np.sqrt(df_clustering["expected_trades"])
    df_clustering["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(df_clustering["z_stat"])))
    df_clustering["significant"] = df_clustering["p_value"] < 0.05

    df_by_price.to_csv(fig_dir / "round_number_clustering.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    colors = np.where(df_by_price["is_extreme"], "#8B008B",
                      np.where(df_by_price["is_key_round"], "#D65F5F",
                               np.where(df_by_price["is_round_5"], "#F0A000", "#4C72B0")))
    ax1.bar(df_by_price["price"], df_by_price["n_trades"] / 1e6, color=colors, alpha=0.7, edgecolor="none", width=1)
    ax1.set_xlabel("Price (cents)")
    ax1.set_ylabel("Number of Trades (millions)")
    ax1.set_title("Trade Count by Price")
    ax1.set_xlim(0, 100)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#8B008B", alpha=0.7, label="Extreme (1-3, 97-99)"),
        Patch(facecolor="#D65F5F", alpha=0.7, label="Key Round (10,25,50,75,90)"),
        Patch(facecolor="#F0A000", alpha=0.7, label="Round (÷5)"),
        Patch(facecolor="#4C72B0", alpha=0.7, label="Non-Round")
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax2 = axes[0, 1]
    ax2.bar(df_by_price["price"], df_by_price["trade_ratio"], color=colors, alpha=0.7, edgecolor="none", width=1)
    ax2.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Price (cents)")
    ax2.set_ylabel("Trade Ratio (vs uniform)")
    ax2.set_title("Trade Concentration Ratio")
    ax2.set_xlim(0, 100)

    ax3 = axes[1, 0]
    price_types = df_clustering["price_type"].tolist()
    x = np.arange(len(price_types))
    bar_colors = ["#D65F5F" if sig else "#888888" for sig in df_clustering["significant"]]
    ax3.bar(x, df_clustering["clustering_ratio"], color=bar_colors, alpha=0.7, edgecolor="none")
    ax3.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(price_types)
    ax3.set_ylabel("Clustering Ratio")
    ax3.set_title(f"Trade Clustering by Price Type\n(χ²={chi2_stat:.0f}, p={chi2_p:.2e})")

    ax4 = axes[1, 1]
    ax4.bar(x, df_clustering["avg_trade_size"], color=["#D65F5F", "#4C72B0", "#F0A000"], alpha=0.7, edgecolor="none")
    ax4.set_xticks(x)
    ax4.set_xticklabels(price_types)
    ax4.set_ylabel("Avg Trade Size (USD)")
    ax4.set_title("Trade Size by Price Type")

    plt.tight_layout()
    fig.savefig(fig_dir / "round_number_clustering.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "round_number_clustering.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")
    print(f"\nStatistical Summary:")
    print(f"  Chi-square test (H0: uniform distribution): χ²={chi2_stat:.2f}, p={chi2_p:.2e}")
    print(f"\n  By price type:")
    for _, row in df_clustering.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"    {row['price_type']:>12}: ratio={row['clustering_ratio']:.3f}, z={row['z_stat']:.2f}, p={row['p_value']:.2e} {sig}")


if __name__ == "__main__":
    main()
