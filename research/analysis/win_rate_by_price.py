#!/usr/bin/env python3
"""Analyze win rate by price to assess market calibration."""

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

    df = con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
        )
        SELECT
            CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
            COUNT(*) AS total_trades,
            SUM(CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END) AS wins,
            100.0 * SUM(CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END) / COUNT(*) AS win_rate
        FROM '{trades_dir}/*.parquet' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
        GROUP BY price
        ORDER BY price
        """
    ).df()

    df.to_csv(fig_dir / "win_rate_by_price.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df["price"], df["win_rate"], s=20, alpha=0.8, color="#4C72B0", edgecolors="none")
    ax.plot([0, 100], [0, 100], linestyle="--", color="#D65F5F", linewidth=1.5, label="Perfect calibration")
    ax.set_xlabel("Contract Price (cents)")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate vs Price: Market Calibration")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(fig_dir / "win_rate_by_price.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "win_rate_by_price.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()
