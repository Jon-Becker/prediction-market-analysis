#!/usr/bin/env python3
"""Analyze win rate by price to assess market calibration."""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt


def main():
    base_dir = Path(__file__).parent.parent.parent
    trades_dir = base_dir / "data" / "trades"
    markets_dir = base_dir / "data" / "markets"
    fig_dir = base_dir / "research" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Include both taker and maker sides for accurate calibration
    df = con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
        ),
        all_positions AS (
            -- Taker side
            SELECT
                CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker

            UNION ALL

            -- Maker side (counterparty)
            SELECT
                CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
        )
        SELECT
            price,
            COUNT(*) AS total_trades,
            SUM(won) AS wins,
            100.0 * SUM(won) / COUNT(*) AS win_rate
        FROM all_positions
        GROUP BY price
        ORDER BY price
        """
    ).df()

    df.to_csv(fig_dir / "win_rate_by_price.csv", index=False)

    # Generate JSON for actual_win_rate_vs_contract_price.json
    calibration_json = {
        "type": "line",
        "title": "Actual Win Rate vs Contract Price",
        "data": [
            {
                "price": int(row["price"]),
                "actual": round(row["win_rate"], 2),
                "implied": int(row["price"]),
            }
            for _, row in df.iterrows()
            if 1 <= row["price"] <= 99
        ],
        "xKey": "price",
        "yKeys": ["actual", "implied"],
        "strokeDasharrays": [None, "5 5"],
        "yUnit": "percent",
        "xLabel": "Contract Price (cents)",
        "yLabel": "Actual Win Rate (%)",
    }
    with open(fig_dir / "actual_win_rate_vs_contract_price.json", "w") as f:
        json.dump(calibration_json, f, indent=2)

    # Generate sample_size_by_price.json (bucketed)
    buckets = [
        ("1-10¢", 1, 10),
        ("11-20¢", 11, 20),
        ("21-30¢", 21, 30),
        ("31-40¢", 31, 40),
        ("41-50¢", 41, 50),
        ("51-60¢", 51, 60),
        ("61-70¢", 61, 70),
        ("71-80¢", 71, 80),
        ("81-90¢", 81, 90),
        ("91-99¢", 91, 99),
    ]
    sample_size_data = []
    for label, low, high in buckets:
        bucket_df = df[(df["price"] >= low) & (df["price"] <= high)]
        total_trades = int(bucket_df["total_trades"].sum())
        sample_size_data.append({"price": label, "trades": total_trades})

    sample_size_json = {
        "type": "bar",
        "title": "Trades by Price Bucket",
        "data": sample_size_data,
        "xKey": "price",
        "yKeys": ["trades"],
        "yUnit": "number",
        "yLabel": "Number of Trades",
        "xLabel": "Contract Price (cents)",
    }
    with open(fig_dir / "sample_size_by_price.json", "w") as f:
        json.dump(sample_size_json, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df["price"], df["win_rate"], s=30, alpha=0.8, color="#4C72B0", edgecolors="none")
    ax.plot([0, 100], [0, 100], linestyle="--", color="#D65F5F", linewidth=1.5, label="Perfect calibration")
    ax.set_xlabel("Contract Price (cents)")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate vs Price: Market Calibration")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xticks(range(0, 101, 1), minor=True)
    ax.set_yticks(range(0, 101, 10))
    ax.set_yticks(range(0, 101, 1), minor=True)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(fig_dir / "win_rate_by_price.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "win_rate_by_price.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()
