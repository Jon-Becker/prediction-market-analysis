#!/usr/bin/env python3
"""Analyze intraday and weekday trading patterns.

Examines trading activity and accuracy by hour of day and day of week
to identify temporal patterns and potential efficiency differences.
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

    df_hourly = con.execute(
        f"""
        WITH trade_data AS (
            SELECT
                EXTRACT(HOUR FROM t.created_time AT TIME ZONE 'America/New_York') AS hour_et,
                (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS price_frac,
                CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet' t
            JOIN '{markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE m.result IN ('yes', 'no')
        )
        SELECT
            hour_et,
            AVG(won) AS win_rate,
            AVG(price_frac) AS expected_win_rate,
            AVG(won - price_frac) AS excess_win_rate,
            VAR_POP(won - price_frac) AS var_excess,
            COUNT(*) AS n_trades,
            SUM(volume_usd) AS total_volume,
            AVG(volume_usd) AS avg_trade_size
        FROM trade_data
        GROUP BY hour_et
        ORDER BY hour_et
        """
    ).df()

    df_hourly["se_excess"] = np.sqrt(df_hourly["var_excess"] / df_hourly["n_trades"])
    df_hourly["z_stat"] = df_hourly["excess_win_rate"] / df_hourly["se_excess"]
    df_hourly["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(df_hourly["z_stat"])))
    df_hourly["significant"] = df_hourly["p_value"] < 0.05

    df_daily = con.execute(
        f"""
        WITH trade_data AS (
            SELECT
                EXTRACT(DOW FROM t.created_time AT TIME ZONE 'America/New_York') AS day_of_week,
                (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS price_frac,
                CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet' t
            JOIN '{markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE m.result IN ('yes', 'no')
        )
        SELECT
            day_of_week,
            CASE day_of_week
                WHEN 0 THEN 'Sun'
                WHEN 1 THEN 'Mon'
                WHEN 2 THEN 'Tue'
                WHEN 3 THEN 'Wed'
                WHEN 4 THEN 'Thu'
                WHEN 5 THEN 'Fri'
                WHEN 6 THEN 'Sat'
            END AS day_label,
            AVG(won) AS win_rate,
            AVG(price_frac) AS expected_win_rate,
            AVG(won - price_frac) AS excess_win_rate,
            VAR_POP(won - price_frac) AS var_excess,
            COUNT(*) AS n_trades,
            SUM(volume_usd) AS total_volume,
            AVG(volume_usd) AS avg_trade_size
        FROM trade_data
        GROUP BY day_of_week
        ORDER BY day_of_week
        """
    ).df()

    df_daily["se_excess"] = np.sqrt(df_daily["var_excess"] / df_daily["n_trades"])
    df_daily["z_stat"] = df_daily["excess_win_rate"] / df_daily["se_excess"]
    df_daily["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(df_daily["z_stat"])))
    df_daily["significant"] = df_daily["p_value"] < 0.05

    df_weekend = con.execute(
        f"""
        WITH trade_data AS (
            SELECT
                CASE
                    WHEN EXTRACT(DOW FROM t.created_time AT TIME ZONE 'America/New_York') IN (0, 6) THEN 'Weekend'
                    ELSE 'Weekday'
                END AS period,
                (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS price_frac,
                CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet' t
            JOIN '{markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE m.result IN ('yes', 'no')
        )
        SELECT
            period,
            AVG(won) AS win_rate,
            AVG(price_frac) AS expected_win_rate,
            AVG(won - price_frac) AS excess_win_rate,
            AVG(ABS(won - price_frac)) AS mean_abs_error,
            COUNT(*) AS n_trades,
            SUM(volume_usd) AS total_volume
        FROM trade_data
        GROUP BY period
        """
    ).df()

    df_hourly.to_csv(fig_dir / "intraday_weekday_patterns.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax1 = axes[0, 0]
    ax1.bar(df_hourly["hour_et"], df_hourly["n_trades"] / 1e6, color="#4C72B0", alpha=0.7, edgecolor="none")
    ax1.set_xlabel("Hour (ET)")
    ax1.set_ylabel("Number of Trades (millions)")
    ax1.set_title("Trading Volume by Hour")
    ax1.set_xticks(range(0, 24, 3))

    ax2 = axes[0, 1]
    colors = ["#D65F5F" if sig else "#888888" for sig in df_hourly["significant"]]
    ax2.bar(df_hourly["hour_et"], df_hourly["excess_win_rate"] * 100, color=colors, alpha=0.7, edgecolor="none")
    ax2.errorbar(df_hourly["hour_et"], df_hourly["excess_win_rate"] * 100, yerr=1.96 * df_hourly["se_excess"] * 100,
                 fmt="none", color="gray", alpha=0.4, capsize=2)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Hour (ET)")
    ax2.set_ylabel("Excess Win Rate (pp)")
    ax2.set_title("Market Efficiency by Hour\n(red = p < 0.05)")
    ax2.set_xticks(range(0, 24, 3))

    ax3 = axes[0, 2]
    ax3.plot(df_hourly["hour_et"], df_hourly["avg_trade_size"], marker="o", color="#4C72B0", linewidth=2)
    ax3.set_xlabel("Hour (ET)")
    ax3.set_ylabel("Avg Trade Size (USD)")
    ax3.set_title("Trade Size by Hour")
    ax3.set_xticks(range(0, 24, 3))

    ax4 = axes[1, 0]
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df_daily_sorted = df_daily.set_index("day_label").loc[day_order].reset_index()
    weekend_colors = ["#D65F5F" if d in ["Sat", "Sun"] else "#4C72B0" for d in day_order]
    ax4.bar(range(7), df_daily_sorted["n_trades"] / 1e6, color=weekend_colors, alpha=0.7, edgecolor="none")
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(day_order)
    ax4.set_xlabel("Day of Week")
    ax4.set_ylabel("Number of Trades (millions)")
    ax4.set_title("Trading Volume by Day")

    ax5 = axes[1, 1]
    excess_colors = ["#D65F5F" if sig else "#888888" for sig in df_daily_sorted["significant"]]
    ax5.bar(range(7), df_daily_sorted["excess_win_rate"] * 100, color=excess_colors, alpha=0.7, edgecolor="none")
    ax5.errorbar(range(7), df_daily_sorted["excess_win_rate"] * 100, yerr=1.96 * df_daily_sorted["se_excess"] * 100,
                 fmt="none", color="gray", alpha=0.5, capsize=3)
    ax5.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax5.set_xticks(range(7))
    ax5.set_xticklabels(day_order)
    ax5.set_xlabel("Day of Week")
    ax5.set_ylabel("Excess Win Rate (pp)")
    ax5.set_title("Market Efficiency by Day\n(red = p < 0.05)")

    ax6 = axes[1, 2]
    x6 = np.arange(2)
    df_weekend_sorted = df_weekend.set_index("period").loc[["Weekday", "Weekend"]].reset_index()
    ax6.bar(x6, df_weekend_sorted["mean_abs_error"] * 100, color=["#4C72B0", "#D65F5F"], alpha=0.7, edgecolor="none")
    ax6.set_xticks(x6)
    ax6.set_xticklabels(["Weekday", "Weekend"])
    ax6.set_ylabel("Mean Absolute Error (pp)")
    ax6.set_title("Weekend vs Weekday Accuracy")

    plt.tight_layout()
    fig.savefig(fig_dir / "intraday_weekday_patterns.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "intraday_weekday_patterns.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")
    print(f"\nStatistical Summary - Hourly (H0: excess = 0):")
    for _, row in df_hourly.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"  {int(row['hour_et']):2d}:00 ET: excess={row['excess_win_rate']*100:+.3f}pp, z={row['z_stat']:.2f}, p={row['p_value']:.2e} {sig}")

    print(f"\nStatistical Summary - Daily:")
    for _, row in df_daily.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"  {row['day_label']}: excess={row['excess_win_rate']*100:+.3f}pp, z={row['z_stat']:.2f}, p={row['p_value']:.2e} {sig}")


if __name__ == "__main__":
    main()
