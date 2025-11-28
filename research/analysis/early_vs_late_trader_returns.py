#!/usr/bin/env python3
"""Analyze early vs late trader returns.

Examines whether traders who enter early in a market's life have different
returns than those who trade close to resolution. Tests for information timing.
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
        WITH market_times AS (
            SELECT
                ticker,
                result,
                open_time,
                close_time,
                EXTRACT(EPOCH FROM (close_time - open_time)) AS market_duration_secs
            FROM '{markets_dir}/*.parquet'
            WHERE result IN ('yes', 'no')
              AND open_time IS NOT NULL
              AND close_time IS NOT NULL
              AND close_time > open_time
        ),
        trade_timing AS (
            SELECT
                t.ticker,
                EXTRACT(EPOCH FROM (t.created_time - m.open_time)) / NULLIF(m.market_duration_secs, 0) AS pct_through_market,
                (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS price_frac,
                CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet' t
            JOIN market_times m ON t.ticker = m.ticker
            WHERE t.created_time >= m.open_time
              AND t.created_time <= m.close_time
        ),
        binned AS (
            SELECT
                FLOOR(pct_through_market * 10) / 10.0 AS timing_bin,
                AVG(won) AS win_rate,
                AVG(price_frac) AS expected_win_rate,
                AVG(won - price_frac) AS excess_win_rate,
                VAR_POP(won - price_frac) AS var_excess,
                SUM(volume_usd) AS total_volume,
                COUNT(*) AS n_trades
            FROM trade_timing
            WHERE pct_through_market BETWEEN 0 AND 1
            GROUP BY timing_bin
        )
        SELECT
            timing_bin * 100 AS pct_through,
            win_rate,
            expected_win_rate,
            excess_win_rate,
            var_excess,
            total_volume,
            n_trades
        FROM binned
        ORDER BY timing_bin
        """
    ).df()

    # Z-test: H0: excess_win_rate = 0
    df["se_excess"] = np.sqrt(df["var_excess"] / df["n_trades"])
    df["z_stat"] = df["excess_win_rate"] / df["se_excess"]
    df["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(df["z_stat"])))
    df["significant"] = df["p_value"] < 0.05

    df_size_timing = con.execute(
        f"""
        WITH market_times AS (
            SELECT
                ticker,
                result,
                open_time,
                close_time,
                EXTRACT(EPOCH FROM (close_time - open_time)) AS market_duration_secs
            FROM '{markets_dir}/*.parquet'
            WHERE result IN ('yes', 'no')
              AND open_time IS NOT NULL
              AND close_time IS NOT NULL
              AND close_time > open_time
        ),
        trade_timing AS (
            SELECT
                EXTRACT(EPOCH FROM (t.created_time - m.open_time)) / NULLIF(m.market_duration_secs, 0) AS pct_through_market,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS trade_size_usd
            FROM '{trades_dir}/*.parquet' t
            JOIN market_times m ON t.ticker = m.ticker
            WHERE t.created_time >= m.open_time
              AND t.created_time <= m.close_time
        )
        SELECT
            FLOOR(pct_through_market * 10) / 10.0 * 100 AS pct_through,
            AVG(trade_size_usd) AS avg_trade_size,
            MEDIAN(trade_size_usd) AS median_trade_size,
            COUNT(*) AS n_trades
        FROM trade_timing
        WHERE pct_through_market BETWEEN 0 AND 1
        GROUP BY FLOOR(pct_through_market * 10)
        ORDER BY pct_through
        """
    ).df()

    df.to_csv(fig_dir / "early_vs_late_trader_returns.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax1 = axes[0]
    sizes = np.clip(df["n_trades"] / df["n_trades"].max() * 300, 30, 300)
    colors = ["#D65F5F" if sig else "#4C72B0" for sig in df["significant"]]
    ax1.scatter(df["pct_through"], df["excess_win_rate"] * 100, s=sizes, c=colors, alpha=0.7, edgecolor="none")
    ax1.errorbar(df["pct_through"], df["excess_win_rate"] * 100, yerr=1.96 * df["se_excess"] * 100,
                 fmt="none", color="gray", alpha=0.5, capsize=3)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    x = df["pct_through"].values
    y = df["excess_win_rate"].values * 100
    weights = df["n_trades"].values
    coeffs = np.polyfit(x, y, 1, w=np.sqrt(weights))
    x_fit = np.linspace(0, 100, 100)
    y_fit = np.polyval(coeffs, x_fit)
    ax1.plot(x_fit, y_fit, color="#C44E52", linewidth=2, label=f"slope: {coeffs[0]*100:.3f}pp per 100%")
    ax1.legend(loc="lower right")

    ax1.set_xlabel("% Through Market Lifetime")
    ax1.set_ylabel("Excess Win Rate (pp)")
    ax1.set_title("Trader Returns by Entry Timing\n(red = p < 0.05, error bars = 95% CI)")
    ax1.set_xlim(-5, 105)

    ax2 = axes[1]
    ax2.bar(df["pct_through"], df["total_volume"] / 1e6, width=8, color="#4C72B0", alpha=0.7, edgecolor="none")
    ax2.set_xlabel("% Through Market Lifetime")
    ax2.set_ylabel("Total Volume ($ millions)")
    ax2.set_title("Volume Distribution Over Market Life")
    ax2.set_xlim(-5, 105)

    ax3 = axes[2]
    ax3.plot(df_size_timing["pct_through"], df_size_timing["avg_trade_size"], marker="o", color="#4C72B0", linewidth=2, label="Mean")
    ax3.plot(df_size_timing["pct_through"], df_size_timing["median_trade_size"], marker="s", color="#55A868", linewidth=2, label="Median")
    ax3.set_xlabel("% Through Market Lifetime")
    ax3.set_ylabel("Trade Size (USD)")
    ax3.set_title("Trade Size Over Market Life")
    ax3.set_xlim(-5, 105)
    ax3.legend()

    plt.tight_layout()
    fig.savefig(fig_dir / "early_vs_late_trader_returns.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "early_vs_late_trader_returns.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")
    print(f"\nStatistical Summary:")
    print(f"  H0: Excess win rate = 0 (no timing advantage)")
    for _, row in df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"  {row['pct_through']:3.0f}%: excess={row['excess_win_rate']*100:+.3f}pp, z={row['z_stat']:.2f}, p={row['p_value']:.2e} {sig}")


if __name__ == "__main__":
    main()
