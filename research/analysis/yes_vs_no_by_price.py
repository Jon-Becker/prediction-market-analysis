#!/usr/bin/env python3
"""Analyze YES/NO preference by price, comparing takers vs makers."""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt


def main():
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data" / "trades"
    fig_dir = base_dir / "research" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Get taker and maker YES/NO volumes at each price
    df = con.execute(
        f"""
        WITH taker_yes AS (
            SELECT yes_price AS price, SUM(count) AS contracts
            FROM '{data_dir}/*.parquet'
            WHERE taker_side = 'yes'
            GROUP BY yes_price
        ),
        taker_no AS (
            SELECT no_price AS price, SUM(count) AS contracts
            FROM '{data_dir}/*.parquet'
            WHERE taker_side = 'no'
            GROUP BY no_price
        ),
        maker_yes AS (
            SELECT yes_price AS price, SUM(count) AS contracts
            FROM '{data_dir}/*.parquet'
            WHERE taker_side = 'no'
            GROUP BY yes_price
        ),
        maker_no AS (
            SELECT no_price AS price, SUM(count) AS contracts
            FROM '{data_dir}/*.parquet'
            WHERE taker_side = 'yes'
            GROUP BY no_price
        ),
        all_prices AS (
            SELECT DISTINCT price FROM (
                SELECT price FROM taker_yes
                UNION SELECT price FROM taker_no
                UNION SELECT price FROM maker_yes
                UNION SELECT price FROM maker_no
            )
            WHERE price BETWEEN 1 AND 99
        )
        SELECT
            p.price,
            COALESCE(ty.contracts, 0) AS taker_yes,
            COALESCE(tn.contracts, 0) AS taker_no,
            COALESCE(my.contracts, 0) AS maker_yes,
            COALESCE(mn.contracts, 0) AS maker_no
        FROM all_prices p
        LEFT JOIN taker_yes ty ON p.price = ty.price
        LEFT JOIN taker_no tn ON p.price = tn.price
        LEFT JOIN maker_yes my ON p.price = my.price
        LEFT JOIN maker_no mn ON p.price = mn.price
        ORDER BY p.price
        """
    ).df()

    # Calculate percentages for 100% stacked bar
    df["total"] = df["taker_yes"] + df["taker_no"] + df["maker_yes"] + df["maker_no"]
    df["taker_yes_pct"] = df["taker_yes"] / df["total"] * 100
    df["taker_no_pct"] = df["taker_no"] / df["total"] * 100
    df["maker_yes_pct"] = df["maker_yes"] / df["total"] * 100
    df["maker_no_pct"] = df["maker_no"] / df["total"] * 100

    df.to_csv(fig_dir / "yes_vs_no_by_price.csv", index=False)

    # Generate JSON for paper
    json_data = {
        "type": "stacked-area-100",
        "title": "YES vs NO Volume by Price",
        "xKey": "price",
        "xLabel": "YES Contract Price",
        "yKeys": ["taker_yes", "maker_yes", "taker_no", "maker_no"],
        "yLabels": ["Taker YES", "Maker YES", "Taker NO", "Maker NO"],
        "yLabel": "Share of Volume (%)",
        "data": [
            {
                "price": int(row["price"]),
                "taker_yes": round(row["taker_yes_pct"], 2),
                "maker_yes": round(row["maker_yes_pct"], 2),
                "taker_no": round(row["taker_no_pct"], 2),
                "maker_no": round(row["maker_no_pct"], 2),
            }
            for _, row in df.iterrows()
        ],
    }
    with open(fig_dir / "yes_vs_no_by_price_taker_maker.json", "w") as f:
        json.dump(json_data, f, indent=2)

    # 100% stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(df["price"], df["taker_yes_pct"], width=1, color="#2ecc71", label="Taker YES")
    ax.bar(
        df["price"],
        df["maker_yes_pct"],
        width=1,
        color="#27ae60",
        label="Maker YES",
        bottom=df["taker_yes_pct"],
    )
    ax.bar(
        df["price"],
        df["taker_no_pct"],
        width=1,
        color="#e74c3c",
        label="Taker NO",
        bottom=df["taker_yes_pct"] + df["maker_yes_pct"],
    )
    ax.bar(
        df["price"],
        df["maker_no_pct"],
        width=1,
        color="#c0392b",
        label="Maker NO",
        bottom=df["taker_yes_pct"] + df["maker_yes_pct"] + df["taker_no_pct"],
    )

    ax.set_xlabel("Contract Price (cents)")
    ax.set_ylabel("Share of Volume (%)")
    ax.set_title("YES vs NO by Price: Taker vs Maker Breakdown")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(fig_dir / "yes_vs_no_by_price.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "yes_vs_no_by_price.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()
