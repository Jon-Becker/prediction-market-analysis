#!/usr/bin/env python3
"""Analyze trade size (USD) vs average win rate, controlling for price. MAD outlier filtering."""

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

    # Compute excess win rate with fixed upper bound to exclude extreme outliers
    upper = 10000  # $10k upper bound
    df = con.execute(
        f"""
        WITH trade_data AS (
            SELECT
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS trade_size_usd,
                CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS expected_win_rate
            FROM '{trades_dir}/*.parquet' t
            JOIN '{markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE m.result IN ('yes', 'no')
        ),
        filtered AS (
            SELECT *
            FROM trade_data
            WHERE trade_size_usd BETWEEN 0 AND {upper}
        ),
        binned AS (
            SELECT
                FLOOR(trade_size_usd / 100) * 100 AS trade_size_bin,
                AVG(won) AS win_rate,
                AVG(expected_win_rate) AS expected_win_rate,
                AVG(won - expected_win_rate) AS excess_win_rate,
                COUNT(*) AS n_trades
            FROM filtered
            GROUP BY trade_size_bin
            HAVING COUNT(*) >= 100
        )
        SELECT trade_size_bin, win_rate, expected_win_rate, excess_win_rate, n_trades
        FROM binned
        ORDER BY trade_size_bin
        """
    ).df()

    df.to_csv(fig_dir / "trade_size_vs_win_rate_iqr.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    sizes = np.clip(df["n_trades"] / df["n_trades"].max() * 200, 10, 200)
    ax.scatter(
        df["trade_size_bin"],
        df["excess_win_rate"] * 100,
        s=sizes,
        alpha=0.6,
        color="#4C72B0",
        edgecolor="none",
    )
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Line of best fit (weighted by n_trades)
    x = df["trade_size_bin"].values
    y = df["excess_win_rate"].values * 100
    weights = df["n_trades"].values
    coeffs = np.polyfit(x, y, 1, w=np.sqrt(weights))
    ax.plot(x, np.polyval(coeffs, x), color="#C44E52", linewidth=2, label=f"fit: {coeffs[0]:.2e}x + {coeffs[1]:.2f}")
    ax.legend(loc="lower right")

    ax.set_xlabel("Trade Size (USD)")
    ax.set_ylabel("Excess Win Rate (pp)")
    ax.set_title(f"Trade Size vs Excess Win Rate (<${upper:,})")

    plt.tight_layout()
    fig.savefig(fig_dir / "trade_size_vs_win_rate_iqr.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "trade_size_vs_win_rate_iqr.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()
