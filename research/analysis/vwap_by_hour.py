#!/usr/bin/env python3
"""Analyze volume-weighted average price by hour of day (ET).

Examines whether trading patterns vary by time of day, potentially
revealing when retail vs. institutional participants are most active.
Lower VWAP suggests more longshot buying; higher VWAP suggests more
favorite buying.
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

    # Compute VWAP by hour of day (ET)
    # Trade timestamps are already in America/New_York timezone
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
                t.count AS contracts,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
        )
        SELECT
            hour_et,
            SUM(price * contracts) / SUM(contracts) AS vwap,
            SUM(contracts) AS total_contracts,
            SUM(volume_usd) AS total_volume_usd,
            COUNT(*) AS n_trades,
            AVG(price) AS avg_price,
            STDDEV_SAMP(price) AS std_price
        FROM trade_data
        GROUP BY hour_et
        ORDER BY hour_et
        """
    ).df()

    # Calculate standard error for VWAP (approximation using std of prices)
    df["se"] = df["std_price"] / np.sqrt(df["n_trades"])
    df["ci_lower"] = df["vwap"] - 1.96 * df["se"]
    df["ci_upper"] = df["vwap"] + 1.96 * df["se"]

    df.to_csv(fig_dir / "vwap_by_hour.csv", index=False)

    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot VWAP as line with CI band
    hours = df["hour_et"].values
    vwap = df["vwap"].values
    ci_lower = df["ci_lower"].values
    ci_upper = df["ci_upper"].values

    ax1.fill_between(hours, ci_lower, ci_upper, alpha=0.2, color="#4C72B0")
    ax1.plot(hours, vwap, color="#4C72B0", linewidth=2, marker="o", markersize=6, label="VWAP")
    ax1.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Fair odds (50¢)")

    ax1.set_xlabel("Hour of Day (ET)")
    ax1.set_ylabel("Volume-Weighted Avg Price (cents)")
    ax1.set_title("Volume-Weighted Average Price by Hour of Day")
    ax1.set_xlim(-0.5, 23.5)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Add volume bars on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(hours, df["total_volume_usd"] / 1e9, alpha=0.3, color="#2ecc71", width=0.8, label="Volume")
    ax2.set_ylabel("Volume ($ Billions)", color="#2ecc71")
    ax2.tick_params(axis="y", labelcolor="#2ecc71")

    plt.tight_layout()
    fig.savefig(fig_dir / "vwap_by_hour.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "vwap_by_hour.pdf", bbox_inches="tight")
    plt.close(fig)

    # JSON output for paper - simple line chart
    json_data = {
        "type": "line",
        "title": "Volume-Weighted Average Price by Hour of Day (ET)",
        "data": [
            {
                "hour": int(row["hour_et"]),
                "VWAP": round(row["vwap"], 2),
            }
            for _, row in df.iterrows()
        ],
        "xKey": "hour",
        "yKeys": ["VWAP"],
        "yUnit": "cents",
    }
    with open(fig_dir / "vwap_by_hour.json", "w") as f:
        json.dump(json_data, f)

    # Print summary statistics
    print(f"Outputs saved to {fig_dir}")
    print("\n=== VWAP by Hour Statistics ===")
    print(f"Overall VWAP: {df['vwap'].mean():.2f}¢")
    print(f"Min VWAP: {df['vwap'].min():.2f}¢ at {int(df.loc[df['vwap'].idxmin(), 'hour_et']):02d}:00 ET")
    print(f"Max VWAP: {df['vwap'].max():.2f}¢ at {int(df.loc[df['vwap'].idxmax(), 'hour_et']):02d}:00 ET")
    print(f"Range: {df['vwap'].max() - df['vwap'].min():.2f}¢")

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
            weighted_vwap = (subset["vwap"] * subset["total_contracts"]).sum() / subset["total_contracts"].sum()
            total_vol = subset["total_volume_usd"].sum()
            total_trades = subset["n_trades"].sum()
            print(f"\n{name}:")
            print(f"  VWAP: {weighted_vwap:.2f}¢")
            print(f"  Volume: ${total_vol/1e9:.2f}B ({total_vol/df['total_volume_usd'].sum()*100:.1f}%)")
            print(f"  Trades: {total_trades:,.0f}")


if __name__ == "__main__":
    main()
