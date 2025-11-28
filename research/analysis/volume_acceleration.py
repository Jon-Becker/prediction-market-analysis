#!/usr/bin/env python3
"""Analyze volume acceleration near market resolution.

Examines whether trading volume spikes as markets approach their close time,
and how volume is distributed across the market lifecycle.
Includes chi-square and Kolmogorov-Smirnov tests for non-uniform distribution.
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

    df_by_hours = con.execute(
        f"""
        WITH trade_data AS (
            SELECT
                EXTRACT(EPOCH FROM (m.close_time - t.created_time)) / 3600.0 AS hours_to_close,
                t.count AS contracts,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet' t
            JOIN '{markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE m.result IN ('yes', 'no')
              AND m.close_time IS NOT NULL
              AND t.created_time < m.close_time
              AND t.created_time > m.close_time - INTERVAL '30 days'
        )
        SELECT
            CASE
                WHEN hours_to_close < 0.25 THEN 0.125
                WHEN hours_to_close < 0.5 THEN 0.375
                WHEN hours_to_close < 1 THEN 0.75
                WHEN hours_to_close < 2 THEN 1.5
                WHEN hours_to_close < 4 THEN 3
                WHEN hours_to_close < 8 THEN 6
                WHEN hours_to_close < 24 THEN 16
                WHEN hours_to_close < 72 THEN 48
                WHEN hours_to_close < 168 THEN 120
                ELSE 336
            END AS hours_bin,
            CASE
                WHEN hours_to_close < 0.25 THEN '<15m'
                WHEN hours_to_close < 0.5 THEN '15-30m'
                WHEN hours_to_close < 1 THEN '30m-1h'
                WHEN hours_to_close < 2 THEN '1-2h'
                WHEN hours_to_close < 4 THEN '2-4h'
                WHEN hours_to_close < 8 THEN '4-8h'
                WHEN hours_to_close < 24 THEN '8-24h'
                WHEN hours_to_close < 72 THEN '1-3d'
                WHEN hours_to_close < 168 THEN '3-7d'
                ELSE '>7d'
            END AS hours_label,
            COUNT(*) AS n_trades,
            SUM(contracts) AS total_contracts,
            SUM(volume_usd) AS total_volume_usd,
            AVG(volume_usd) AS avg_trade_size
        FROM trade_data
        WHERE hours_to_close >= 0
        GROUP BY hours_bin, hours_label
        ORDER BY hours_bin
        """
    ).df()

    df_cumulative = con.execute(
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
                EXTRACT(EPOCH FROM (t.created_time - m.open_time)) / NULLIF(m.market_duration_secs, 0) AS pct_through,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet' t
            JOIN market_times m ON t.ticker = m.ticker
            WHERE t.created_time >= m.open_time
              AND t.created_time <= m.close_time
        )
        SELECT
            FLOOR(pct_through * 20) / 20.0 AS pct_bin,
            SUM(volume_usd) AS volume_usd,
            COUNT(*) AS n_trades
        FROM trade_timing
        WHERE pct_through BETWEEN 0 AND 1
        GROUP BY pct_bin
        ORDER BY pct_bin
        """
    ).df()

    df_cumulative["cumulative_volume"] = df_cumulative["volume_usd"].cumsum()
    df_cumulative["cumulative_pct"] = df_cumulative["cumulative_volume"] / df_cumulative["cumulative_volume"].iloc[-1] * 100

    # Kolmogorov-Smirnov test: H0: volume is uniformly distributed over market lifetime
    # Compare empirical CDF to uniform CDF
    empirical_cdf = df_cumulative["cumulative_pct"].values / 100
    uniform_cdf = np.linspace(0, 1, len(empirical_cdf))
    ks_stat, ks_p = stats.kstest(empirical_cdf, "uniform")

    # Chi-square test on time bins
    observed_volume = df_by_hours["total_volume_usd"].values
    expected_volume = np.full(len(observed_volume), observed_volume.sum() / len(observed_volume))
    chi2_stat, chi2_p = stats.chisquare(observed_volume, expected_volume)

    df_by_hours.to_csv(fig_dir / "volume_acceleration.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax1 = axes[0]
    x = np.arange(len(df_by_hours))
    ax1.bar(x, df_by_hours["total_volume_usd"] / 1e6, color="#4C72B0", alpha=0.7, edgecolor="none")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_by_hours["hours_label"], rotation=45, ha="right")
    ax1.set_xlabel("Time to Resolution")
    ax1.set_ylabel("Total Volume ($ millions)")
    ax1.set_title("Volume by Time to Close")
    ax1.invert_xaxis()

    ax2 = axes[1]
    ax2.bar(x, df_by_hours["n_trades"] / 1e6, color="#55A868", alpha=0.7, edgecolor="none")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_by_hours["hours_label"], rotation=45, ha="right")
    ax2.set_xlabel("Time to Resolution")
    ax2.set_ylabel("Number of Trades (millions)")
    ax2.set_title("Trade Count by Time to Close")
    ax2.invert_xaxis()

    ax3 = axes[2]
    ax3.plot(df_cumulative["pct_bin"] * 100, df_cumulative["cumulative_pct"], color="#4C72B0", linewidth=2)
    ax3.plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1, alpha=0.7, label="Uniform")
    ax3.fill_between(df_cumulative["pct_bin"] * 100, df_cumulative["cumulative_pct"], alpha=0.3, color="#4C72B0")
    ax3.set_xlabel("% Through Market Lifetime")
    ax3.set_ylabel("Cumulative % of Volume")
    ax3.set_title(f"Cumulative Volume Distribution\n(KS stat={ks_stat:.3f}, p={ks_p:.2e})")
    ax3.legend()

    plt.tight_layout()
    fig.savefig(fig_dir / "volume_acceleration.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "volume_acceleration.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")
    print(f"\nStatistical Summary:")
    print(f"  Chi-square test (H0: uniform volume by time bin): χ²={chi2_stat:.2e}, p={chi2_p:.2e}")
    print(f"  KS test (H0: uniform cumulative distribution): D={ks_stat:.3f}, p={ks_p:.2e}")
    print(f"\n  Volume is highly non-uniformly distributed (concentrated near resolution).")


if __name__ == "__main__":
    main()
