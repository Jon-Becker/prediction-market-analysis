#!/usr/bin/env python3
"""Analyze excess returns by hour of day (ET).

Examines whether trading performance varies by time of day, potentially
revealing when informed vs. uninformed participants are most active.
"""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np


def main():
    base_dir = Path(__file__).parent.parent.parent
    trades_dir = base_dir / "data" / "trades"
    markets_dir = base_dir / "data" / "markets"
    fig_dir = base_dir / "research" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Compute excess returns by hour of day (ET)
    df = con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
        ),
        trade_data AS (
            SELECT
                EXTRACT(HOUR FROM t.created_time) AS hour_et,
                CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                t.count AS contracts,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
        )
        SELECT
            hour_et,
            AVG(won) AS win_rate,
            AVG(price / 100.0) AS avg_implied_prob,
            AVG(won - price / 100.0) AS excess_return,
            VAR_SAMP(won - price / 100.0) AS var_excess,
            SUM(contracts) AS total_contracts,
            SUM(volume_usd) AS total_volume_usd,
            COUNT(*) AS n_trades
        FROM trade_data
        GROUP BY hour_et
        ORDER BY hour_et
        """
    ).df()

    # Calculate standard error and 95% CI
    df["se"] = np.sqrt(df["var_excess"] / df["n_trades"])
    df["ci_lower"] = df["excess_return"] - 1.96 * df["se"]
    df["ci_upper"] = df["excess_return"] + 1.96 * df["se"]

    df.to_csv(fig_dir / "returns_by_hour.csv", index=False)

    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 6))

    hours = df["hour_et"].values
    excess = df["excess_return"].values * 100  # Convert to percentage points

    ax1.bar(hours, excess, color="#4C72B0", alpha=0.7, width=0.8)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    ax1.set_xlabel("Hour of Day (ET)")
    ax1.set_ylabel("Excess Return (pp)")
    ax1.set_title("Excess Return by Hour of Day")
    ax1.set_xlim(-0.5, 23.5)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
    ax1.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(fig_dir / "returns_by_hour.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "returns_by_hour.pdf", bbox_inches="tight")
    plt.close(fig)

    # JSON output for paper
    json_data = {
        "type": "bar",
        "title": "Excess Return by Hour of Day (ET)",
        "data": [
            {
                "hour": int(row["hour_et"]),
                "Excess Return": round(row["excess_return"] * 100, 2),
            }
            for _, row in df.iterrows()
        ],
        "xKey": "hour",
        "yKeys": ["Excess Return"],
        "yUnit": "percent",
    }
    with open(fig_dir / "returns_by_hour.json", "w") as f:
        json.dump(json_data, f)

    # Print summary statistics
    print(f"Outputs saved to {fig_dir}")
    print("\n=== Returns by Hour Statistics ===")
    print(f"Overall excess return: {df['excess_return'].mean() * 100:.2f} pp")
    print(f"Min excess return: {df['excess_return'].min() * 100:.2f} pp at {int(df.loc[df['excess_return'].idxmin(), 'hour_et']):02d}:00 ET")
    print(f"Max excess return: {df['excess_return'].max() * 100:.2f} pp at {int(df.loc[df['excess_return'].idxmax(), 'hour_et']):02d}:00 ET")
    print(f"Range: {(df['excess_return'].max() - df['excess_return'].min()) * 100:.2f} pp")

    # Trading session analysis
    print("\n=== By Trading Session ===")
    overnight = df[(df["hour_et"] >= 0) & (df["hour_et"] < 6)]
    morning = df[(df["hour_et"] >= 6) & (df["hour_et"] < 12)]
    afternoon = df[(df["hour_et"] >= 12) & (df["hour_et"] < 18)]
    evening = df[(df["hour_et"] >= 18) & (df["hour_et"] < 24)]

    for name, subset in [
        ("Overnight (00:00-06:00)", overnight),
        ("Morning (06:00-12:00)", morning),
        ("Afternoon (12:00-18:00)", afternoon),
        ("Evening (18:00-24:00)", evening),
    ]:
        if len(subset) > 0:
            # Volume-weighted excess return
            weighted_excess = (subset["excess_return"] * subset["total_contracts"]).sum() / subset["total_contracts"].sum() * 100
            total_vol = subset["total_volume_usd"].sum()
            total_trades = subset["n_trades"].sum()
            print(f"\n{name}:")
            print(f"  Excess return: {weighted_excess:+.2f} pp")
            print(f"  Volume: ${total_vol/1e9:.2f}B ({total_vol/df['total_volume_usd'].sum()*100:.1f}%)")
            print(f"  Trades: {total_trades:,.0f}")


if __name__ == "__main__":
    main()
