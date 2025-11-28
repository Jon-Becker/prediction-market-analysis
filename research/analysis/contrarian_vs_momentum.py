#!/usr/bin/env python3
"""Analyze contrarian vs momentum trading returns.

Examines whether traders who buy after price drops outperform those who
chase momentum. Measures returns conditional on recent price movement.
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
        WITH ordered_trades AS (
            SELECT
                t.ticker,
                t.created_time,
                t.taker_side,
                t.yes_price,
                t.no_price,
                t.count,
                m.result,
                LAG(t.yes_price, 1) OVER (PARTITION BY t.ticker ORDER BY t.created_time) AS prev_yes_price_1,
                LAG(t.yes_price, 5) OVER (PARTITION BY t.ticker ORDER BY t.created_time) AS prev_yes_price_5,
                LAG(t.yes_price, 10) OVER (PARTITION BY t.ticker ORDER BY t.created_time) AS prev_yes_price_10
            FROM '{trades_dir}/*.parquet' t
            JOIN '{markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE m.result IN ('yes', 'no')
        ),
        trade_with_movement AS (
            SELECT
                ticker,
                taker_side,
                yes_price,
                no_price,
                count,
                result,
                yes_price - prev_yes_price_1 AS price_change_1,
                yes_price - prev_yes_price_5 AS price_change_5,
                yes_price - prev_yes_price_10 AS price_change_10,
                (CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) / 100.0 AS price_frac,
                CASE WHEN taker_side = result THEN 1.0 ELSE 0.0 END AS won,
                count * (CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) / 100.0 AS volume_usd
            FROM ordered_trades
            WHERE prev_yes_price_5 IS NOT NULL
        )
        SELECT
            CASE
                WHEN price_change_5 < -10 THEN -15
                WHEN price_change_5 < -5 THEN -7.5
                WHEN price_change_5 < -2 THEN -3.5
                WHEN price_change_5 < -1 THEN -1.5
                WHEN price_change_5 < 0 THEN -0.5
                WHEN price_change_5 = 0 THEN 0
                WHEN price_change_5 <= 1 THEN 0.5
                WHEN price_change_5 <= 2 THEN 1.5
                WHEN price_change_5 <= 5 THEN 3.5
                WHEN price_change_5 <= 10 THEN 7.5
                ELSE 15
            END AS price_change_bin,
            CASE
                WHEN price_change_5 < -10 THEN '<-10'
                WHEN price_change_5 < -5 THEN '-10 to -5'
                WHEN price_change_5 < -2 THEN '-5 to -2'
                WHEN price_change_5 < -1 THEN '-2 to -1'
                WHEN price_change_5 < 0 THEN '-1 to 0'
                WHEN price_change_5 = 0 THEN '0'
                WHEN price_change_5 <= 1 THEN '0 to 1'
                WHEN price_change_5 <= 2 THEN '1 to 2'
                WHEN price_change_5 <= 5 THEN '2 to 5'
                WHEN price_change_5 <= 10 THEN '5 to 10'
                ELSE '>10'
            END AS price_change_label,
            AVG(won) AS win_rate,
            AVG(price_frac) AS expected_win_rate,
            AVG(won - price_frac) AS excess_win_rate,
            VAR_POP(won - price_frac) AS var_excess,
            COUNT(*) AS n_trades,
            SUM(volume_usd) AS total_volume
        FROM trade_with_movement
        GROUP BY price_change_bin, price_change_label
        HAVING COUNT(*) >= 1000
        ORDER BY price_change_bin
        """
    ).df()

    # Z-test: H0: excess_win_rate = 0
    df["se_excess"] = np.sqrt(df["var_excess"] / df["n_trades"])
    df["z_stat"] = df["excess_win_rate"] / df["se_excess"]
    df["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(df["z_stat"])))
    df["significant"] = df["p_value"] < 0.05

    df_by_side = con.execute(
        f"""
        WITH ordered_trades AS (
            SELECT
                t.ticker,
                t.created_time,
                t.taker_side,
                t.yes_price,
                t.no_price,
                t.count,
                m.result,
                LAG(t.yes_price, 5) OVER (PARTITION BY t.ticker ORDER BY t.created_time) AS prev_yes_price_5
            FROM '{trades_dir}/*.parquet' t
            JOIN '{markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE m.result IN ('yes', 'no')
        ),
        trade_with_movement AS (
            SELECT
                taker_side,
                yes_price - prev_yes_price_5 AS price_change_5,
                (CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) / 100.0 AS price_frac,
                CASE WHEN taker_side = result THEN 1.0 ELSE 0.0 END AS won
            FROM ordered_trades
            WHERE prev_yes_price_5 IS NOT NULL
        ),
        categorized AS (
            SELECT
                taker_side,
                CASE
                    WHEN taker_side = 'yes' AND price_change_5 > 0 THEN 'Momentum (buy rising)'
                    WHEN taker_side = 'yes' AND price_change_5 < 0 THEN 'Contrarian (buy falling)'
                    WHEN taker_side = 'no' AND price_change_5 < 0 THEN 'Momentum (sell falling)'
                    WHEN taker_side = 'no' AND price_change_5 > 0 THEN 'Contrarian (sell rising)'
                    ELSE 'Neutral'
                END AS strategy,
                won,
                price_frac
            FROM trade_with_movement
        )
        SELECT
            strategy,
            AVG(won) AS win_rate,
            AVG(price_frac) AS expected_win_rate,
            AVG(won - price_frac) AS excess_win_rate,
            VAR_POP(won - price_frac) AS var_excess,
            COUNT(*) AS n_trades
        FROM categorized
        WHERE strategy != 'Neutral'
        GROUP BY strategy
        ORDER BY strategy
        """
    ).df()

    df_by_side["se_excess"] = np.sqrt(df_by_side["var_excess"] / df_by_side["n_trades"])
    df_by_side["z_stat"] = df_by_side["excess_win_rate"] / df_by_side["se_excess"]
    df_by_side["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(df_by_side["z_stat"])))
    df_by_side["significant"] = df_by_side["p_value"] < 0.05

    df.to_csv(fig_dir / "contrarian_vs_momentum.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax1 = axes[0]
    colors = ["#D65F5F" if x < 0 else "#55A868" if x > 0 else "#888888" for x in df["price_change_bin"]]
    x = np.arange(len(df))
    ax1.bar(x, df["excess_win_rate"] * 100, color=colors, alpha=0.7, edgecolor="none")
    ax1.errorbar(x, df["excess_win_rate"] * 100, yerr=1.96 * df["se_excess"] * 100,
                 fmt="none", color="gray", alpha=0.5, capsize=2)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["price_change_label"], rotation=45, ha="right", fontsize=8)
    ax1.set_xlabel("Recent Price Change (cents, last 5 trades)")
    ax1.set_ylabel("Excess Win Rate (pp)")
    ax1.set_title("Returns by Recent Price Movement\n(error bars = 95% CI)")

    ax2 = axes[1]
    sizes = np.clip(df["n_trades"] / df["n_trades"].max() * 300, 30, 300)
    ax2.scatter(df["price_change_bin"], df["excess_win_rate"] * 100, s=sizes, c=colors, alpha=0.7, edgecolor="none")
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Recent Price Change (cents)")
    ax2.set_ylabel("Excess Win Rate (pp)")
    ax2.set_title("Excess Returns vs Price Movement")

    ax3 = axes[2]
    strategies = df_by_side["strategy"].tolist()
    x3 = np.arange(len(strategies))
    strategy_colors = ["#D65F5F" if "Contrarian" in s else "#55A868" for s in strategies]
    ax3.bar(x3, df_by_side["excess_win_rate"] * 100, color=strategy_colors, alpha=0.7, edgecolor="none")
    ax3.errorbar(x3, df_by_side["excess_win_rate"] * 100, yerr=1.96 * df_by_side["se_excess"] * 100,
                 fmt="none", color="gray", alpha=0.5, capsize=3)
    ax3.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(strategies, rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Excess Win Rate (pp)")
    ax3.set_title("Contrarian vs Momentum Strategies\n(error bars = 95% CI)")

    plt.tight_layout()
    fig.savefig(fig_dir / "contrarian_vs_momentum.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "contrarian_vs_momentum.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")
    print(f"\nStatistical Summary - Price Change Bins:")
    print(f"  H0: Excess win rate = 0")
    for _, row in df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"  {row['price_change_label']:>10}: excess={row['excess_win_rate']*100:+.3f}pp, z={row['z_stat']:.2f}, p={row['p_value']:.2e} {sig}")

    print(f"\nStatistical Summary - Strategies:")
    for _, row in df_by_side.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"  {row['strategy']}: excess={row['excess_win_rate']*100:+.3f}pp, z={row['z_stat']:.2f}, p={row['p_value']:.2e} {sig}")


if __name__ == "__main__":
    main()
