#!/usr/bin/env python3
"""Analyze upset frequency - how often low/high probability events resolve unexpectedly.

Examines whether prices at extremes (near 0 or 100) are well-calibrated,
or if tails are systematically mispriced. Includes z-tests for statistical significance.
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
                CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet' t
            JOIN '{markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE m.result IN ('yes', 'no')
        ),
        binned AS (
            SELECT
                CASE
                    WHEN price <= 5 THEN 2.5
                    WHEN price <= 10 THEN 7.5
                    WHEN price <= 15 THEN 12.5
                    WHEN price <= 20 THEN 17.5
                    WHEN price <= 25 THEN 22.5
                    WHEN price <= 30 THEN 27.5
                    WHEN price <= 35 THEN 32.5
                    WHEN price <= 40 THEN 37.5
                    WHEN price <= 45 THEN 42.5
                    WHEN price <= 50 THEN 47.5
                    WHEN price <= 55 THEN 52.5
                    WHEN price <= 60 THEN 57.5
                    WHEN price <= 65 THEN 62.5
                    WHEN price <= 70 THEN 67.5
                    WHEN price <= 75 THEN 72.5
                    WHEN price <= 80 THEN 77.5
                    WHEN price <= 85 THEN 82.5
                    WHEN price <= 90 THEN 87.5
                    WHEN price <= 95 THEN 92.5
                    ELSE 97.5
                END AS price_bin,
                AVG(won) * 100 AS actual_win_rate,
                AVG(price) AS avg_price_in_bin,
                COUNT(*) AS n_trades,
                SUM(volume_usd) AS total_volume_usd
            FROM trade_data
            GROUP BY price_bin
        )
        SELECT
            price_bin,
            actual_win_rate,
            avg_price_in_bin AS expected_win_rate,
            actual_win_rate - avg_price_in_bin AS excess_win_rate,
            n_trades,
            total_volume_usd
        FROM binned
        ORDER BY price_bin
        """
    ).df()

    df["upset_rate"] = np.where(
        df["price_bin"] <= 50,
        df["actual_win_rate"],
        100 - df["actual_win_rate"]
    )

    # Z-test for each bin: H0: actual_win_rate = expected_win_rate (bin midpoint)
    # For binomial proportion: SE = sqrt(p*(1-p)/n)
    df["se_win_rate"] = np.sqrt(df["actual_win_rate"] / 100 * (1 - df["actual_win_rate"] / 100) / df["n_trades"]) * 100
    df["z_stat"] = df["excess_win_rate"] / df["se_win_rate"]
    df["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(df["z_stat"])))
    df["significant"] = df["p_value"] < 0.05

    df.to_csv(fig_dir / "upset_frequency.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax1 = axes[0]
    ax1.scatter(df["price_bin"], df["actual_win_rate"], s=80, color="#4C72B0", alpha=0.7, edgecolor="none", label="Actual")
    ax1.plot([0, 100], [0, 100], linestyle="--", color="#D65F5F", linewidth=1.5, label="Perfect calibration")
    ax1.set_xlabel("Price (cents)")
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_title("Calibration at Price Extremes")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left")

    ax2 = axes[1]
    colors = ["#D65F5F" if (x < 0 and sig) else "#4C72B0" if sig else "#888888"
              for x, sig in zip(df["excess_win_rate"], df["significant"])]
    ax2.bar(df["price_bin"], df["excess_win_rate"], width=4, color=colors, alpha=0.7, edgecolor="none")
    ax2.errorbar(df["price_bin"], df["excess_win_rate"], yerr=1.96 * df["se_win_rate"],
                 fmt="none", color="gray", alpha=0.5, capsize=2)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Price (cents)")
    ax2.set_ylabel("Excess Win Rate (pp)")
    ax2.set_title("Mispricing by Price Level\n(colored = p < 0.05, error bars = 95% CI)")
    ax2.set_xlim(0, 100)

    ax3 = axes[2]
    low_tail = df[df["price_bin"] <= 10]
    high_tail = df[df["price_bin"] >= 90]
    mid = df[(df["price_bin"] > 10) & (df["price_bin"] < 90)]

    categories = ["Low (<10¢)", "Mid (10-90¢)", "High (>90¢)"]
    excess_rates = [
        low_tail["excess_win_rate"].mean(),
        mid["excess_win_rate"].mean(),
        high_tail["excess_win_rate"].mean()
    ]
    bar_colors = ["#D65F5F" if x < 0 else "#4C72B0" for x in excess_rates]
    ax3.bar(categories, excess_rates, color=bar_colors, alpha=0.7, edgecolor="none")
    ax3.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_ylabel("Avg Excess Win Rate (pp)")
    ax3.set_title("Tail vs Mid Mispricing")

    plt.tight_layout()
    fig.savefig(fig_dir / "upset_frequency.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "upset_frequency.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")
    print(f"\nStatistical Summary:")
    print(f"  H0: Actual win rate = Expected win rate (price bin midpoint)")
    for _, row in df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"  {row['price_bin']:5.1f}¢: excess={row['excess_win_rate']:+.2f}pp, z={row['z_stat']:.2f}, p={row['p_value']:.2e} {sig}")


if __name__ == "__main__":
    main()
