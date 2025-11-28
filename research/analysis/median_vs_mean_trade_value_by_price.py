#!/usr/bin/env python3
"""Analyze median vs mean trade value in USD by price."""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt


def main():
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data" / "trades"
    fig_dir = base_dir / "research" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    df = con.execute(
        f"""
        SELECT
            CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END AS price,
            AVG(count * (CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) / 100.0) AS mean_trade_value,
            MEDIAN(count * (CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) / 100.0) AS median_trade_value
        FROM '{data_dir}/*.parquet'
        GROUP BY price
        ORDER BY price
        """
    ).df()

    df.to_csv(fig_dir / "median_vs_mean_trade_value_by_price.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.4
    ax.bar(df["price"] - width/2, df["mean_trade_value"], width=width, color="#4C72B0", edgecolor="none", label="Mean")
    ax.bar(df["price"] + width/2, df["median_trade_value"], width=width, color="#DD8452", edgecolor="none", label="Median")
    ax.set_xlabel("Contract Price (cents)")
    ax.set_ylabel("Trade Value (USD)")
    ax.set_title("Median vs Mean Trade Value by Price")
    ax.set_xlim(0, 100)
    ax.legend()

    plt.tight_layout()
    fig.savefig(fig_dir / "median_vs_mean_trade_value_by_price.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "median_vs_mean_trade_value_by_price.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()
