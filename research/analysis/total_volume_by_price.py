#!/usr/bin/env python3
"""Analyze total trading volume in USD by price."""

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
            SUM(count * (CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) / 100.0) AS total_volume
        FROM '{data_dir}/*.parquet'
        GROUP BY price
        ORDER BY price
        """
    ).df()

    df.to_csv(fig_dir / "total_volume_by_price.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df["price"], df["total_volume"] / 1e6, width=0.8, color="#4C72B0", edgecolor="none")
    ax.set_xlabel("Contract Price (cents)")
    ax.set_ylabel("Total Volume (millions USD)")
    ax.set_title("Total Trading Volume in USD by Price")
    ax.set_xlim(0, 100)

    plt.tight_layout()
    fig.savefig(fig_dir / "total_volume_by_price.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "total_volume_by_price.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()
