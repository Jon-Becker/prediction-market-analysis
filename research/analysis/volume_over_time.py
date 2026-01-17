#!/usr/bin/env python3
"""Analyze trading volume over time across all Kalshi markets."""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd


def main():
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data" / "trades"
    fig_dir = base_dir / "research" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    df = con.execute(
        f"""
        SELECT
            DATE_TRUNC('quarter', created_time) AS quarter,
            SUM(count) AS volume_usd
        FROM '{data_dir}/*.parquet'
        GROUP BY quarter
        ORDER BY quarter
        """
    ).df()

    df.to_csv(fig_dir / "volume_over_time.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(df["quarter"], df["volume_usd"] / 1e6, width=80, color="#4C72B0")
    bars[-1].set_hatch("//")
    bars[-1].set_edgecolor((1, 1, 1, 0.3))
    labels = [f"${v/1e3:.2f}B" if v > 999 else f"${v:.2f}M" for v in df["volume_usd"] / 1e6]
    ax.bar_label(bars, labels=labels, fontsize=7, rotation=90, label_type="center", color="white", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)
    ax.set_ylabel("Quarterly Volume (millions USD)")
    ax.set_title("Kalshi Quarterly Notional Volume")

    plt.tight_layout()
    fig.savefig(fig_dir / "volume_over_time.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "volume_over_time.pdf", bbox_inches="tight")
    plt.close(fig)

    # Generate JSON for paper
    json_data = {
        "type": "bar",
        "title": "Kalshi Quarterly Volume",
        "data": [
            {
                "quarter": f"Q{(pd.Timestamp(row['quarter']).month - 1) // 3 + 1} '{str(pd.Timestamp(row['quarter']).year)[2:]}",
                "volume": int(row["volume_usd"]),
            }
            for _, row in df.iterrows()
        ],
        "xKey": "quarter",
        "yKeys": ["volume"],
        "xLabel": "Quarter",
        "yLabel": "Volume (USD)",
        "yUnit": "dollars",
        "yScale": "log",
    }
    with open(fig_dir / "kalshi_quarterly_volume.json", "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()
