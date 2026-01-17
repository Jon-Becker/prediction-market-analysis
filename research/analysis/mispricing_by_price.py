#!/usr/bin/env python3
"""Analyze mispricing percentage by contract price for takers, makers, and combined."""

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

    # Query for taker, maker, and combined mispricing by price
    df = con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
        ),
        taker_positions AS (
            SELECT
                CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
        ),
        maker_positions AS (
            SELECT
                CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
        ),
        taker_stats AS (
            SELECT
                price,
                COUNT(*) AS total_trades,
                SUM(won) AS wins,
                100.0 * SUM(won) / COUNT(*) AS win_rate
            FROM taker_positions
            GROUP BY price
        ),
        maker_stats AS (
            SELECT
                price,
                COUNT(*) AS total_trades,
                SUM(won) AS wins,
                100.0 * SUM(won) / COUNT(*) AS win_rate
            FROM maker_positions
            GROUP BY price
        ),
        combined_positions AS (
            SELECT * FROM taker_positions
            UNION ALL
            SELECT * FROM maker_positions
        ),
        combined_stats AS (
            SELECT
                price,
                COUNT(*) AS total_trades,
                SUM(won) AS wins,
                100.0 * SUM(won) / COUNT(*) AS win_rate
            FROM combined_positions
            GROUP BY price
        )
        SELECT
            t.price,
            t.total_trades AS taker_trades,
            t.wins AS taker_wins,
            t.win_rate AS taker_win_rate,
            m.total_trades AS maker_trades,
            m.wins AS maker_wins,
            m.win_rate AS maker_win_rate,
            c.total_trades AS combined_trades,
            c.wins AS combined_wins,
            c.win_rate AS combined_win_rate
        FROM taker_stats t
        JOIN maker_stats m ON t.price = m.price
        JOIN combined_stats c ON t.price = c.price
        WHERE t.price BETWEEN 1 AND 99
        ORDER BY t.price
        """
    ).df()

    # Calculate mispricing: (actual_win_rate - implied_probability) / implied_probability * 100
    # Price is in cents (1-99), so implied probability = price
    df["implied_probability"] = df["price"].astype(float)
    df["taker_mispricing_pct"] = (
        (df["taker_win_rate"] - df["implied_probability"]) / df["implied_probability"] * 100
    )
    df["maker_mispricing_pct"] = (
        (df["maker_win_rate"] - df["implied_probability"]) / df["implied_probability"] * 100
    )
    df["combined_mispricing_pct"] = (
        (df["combined_win_rate"] - df["implied_probability"]) / df["implied_probability"] * 100
    )

    df.to_csv(fig_dir / "mispricing_by_price.csv", index=False)

    # Generate JSON for paper
    # Note: The paper uses mispricing as (actual - implied), not percentage
    # So we need to calculate it as win_rate - price (in percentage points)
    json_data = {
        "type": "line",
        "title": "Mispricing by Contract Price",
        "xKey": "price",
        "yKeys": ["Taker", "Maker", "Combined"],
        "yUnit": "percent",
        "xLabel": "Contract Price (cents)",
        "yLabel": "Mispricing (pp)",
        "colors": {"Taker": "#ef4444", "Maker": "#10b981", "Combined": "#6366f1"},
        "data": [
            {
                "price": int(row["price"]),
                "Taker": round(row["taker_win_rate"] - row["implied_probability"], 2),
                "Maker": round(row["maker_win_rate"] - row["implied_probability"], 2),
                "Combined": round(row["combined_win_rate"] - row["implied_probability"], 2),
            }
            for _, row in df.iterrows()
        ],
    }
    with open(fig_dir / "mispricing_by_price.json", "w") as f:
        json.dump(json_data, f, indent=2)

    # Plot all three series
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        df["price"],
        df["taker_mispricing_pct"],
        s=30,
        alpha=0.7,
        color="#e74c3c",
        edgecolors="none",
        label="Taker",
    )
    ax.scatter(
        df["price"],
        df["maker_mispricing_pct"],
        s=30,
        alpha=0.7,
        color="#2ecc71",
        edgecolors="none",
        label="Maker",
    )
    ax.scatter(
        df["price"],
        df["combined_mispricing_pct"],
        s=30,
        alpha=0.7,
        color="#4C72B0",
        edgecolors="none",
        label="Combined",
    )
    ax.axhline(y=0, linestyle="--", color="gray", linewidth=1.5, label="Perfect calibration")
    ax.set_xlabel("Contract Price (cents)")
    ax.set_ylabel("Mispricing (%)")
    ax.set_title("Mispricing by Contract Price")
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xticks(range(0, 101, 1), minor=True)
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(fig_dir / "mispricing_by_price.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "mispricing_by_price.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()
