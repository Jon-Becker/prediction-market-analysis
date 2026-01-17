#!/usr/bin/env python3
"""Analyze longshot volume share evolution over time.

Computes quarterly volume share by price bucket to assess whether
taker preferences for longshots changed over time or remained constant.
"""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    base_dir = Path(__file__).parent.parent.parent
    trades_dir = base_dir / "data" / "trades"
    markets_dir = base_dir / "data" / "markets"
    fig_dir = base_dir / "research" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Compute quarterly volume by price bucket for takers
    df = con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
        ),
        taker_trades AS (
            SELECT
                DATE_TRUNC('quarter', t.created_time) AS quarter,
                CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                t.count AS contracts,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
        ),
        bucketed AS (
            SELECT
                quarter,
                CASE
                    WHEN price BETWEEN 1 AND 10 THEN '1-10¢'
                    WHEN price BETWEEN 11 AND 20 THEN '11-20¢'
                    WHEN price BETWEEN 21 AND 30 THEN '21-30¢'
                    WHEN price BETWEEN 31 AND 40 THEN '31-40¢'
                    WHEN price BETWEEN 41 AND 50 THEN '41-50¢'
                    WHEN price BETWEEN 51 AND 60 THEN '51-60¢'
                    WHEN price BETWEEN 61 AND 70 THEN '61-70¢'
                    WHEN price BETWEEN 71 AND 80 THEN '71-80¢'
                    WHEN price BETWEEN 81 AND 90 THEN '81-90¢'
                    WHEN price BETWEEN 91 AND 99 THEN '91-99¢'
                END AS price_bucket,
                CASE
                    WHEN price BETWEEN 1 AND 20 THEN 1
                    ELSE 0
                END AS is_longshot,
                volume_usd,
                contracts
            FROM taker_trades
        )
        SELECT
            quarter,
            price_bucket,
            is_longshot,
            SUM(volume_usd) AS volume_usd,
            SUM(contracts) AS contracts,
            COUNT(*) AS n_trades
        FROM bucketed
        GROUP BY quarter, price_bucket, is_longshot
        ORDER BY quarter, price_bucket
        """
    ).df()

    # Convert quarter to pandas datetime
    df["quarter"] = pd.to_datetime(df["quarter"])

    # Compute total volume per quarter
    quarterly_totals = df.groupby("quarter")["volume_usd"].sum().reset_index()
    quarterly_totals.columns = ["quarter", "total_volume"]

    # Merge to get shares
    df = df.merge(quarterly_totals, on="quarter")
    df["volume_share"] = df["volume_usd"] / df["total_volume"] * 100

    # Compute longshot share (1-20¢ and 81-99¢)
    longshot_df = df[df["is_longshot"] == 1].groupby("quarter").agg({
        "volume_usd": "sum",
        "contracts": "sum",
        "n_trades": "sum"
    }).reset_index()
    longshot_df = longshot_df.merge(quarterly_totals, on="quarter")
    longshot_df["longshot_share"] = longshot_df["volume_usd"] / longshot_df["total_volume"] * 100

    # Save to CSV
    longshot_df.to_csv(fig_dir / "longshot_volume_share_over_time.csv", index=False)

    # Pivot for stacked area chart
    pivot_df = df.pivot_table(
        index="quarter",
        columns="price_bucket",
        values="volume_share",
        aggfunc="sum"
    ).fillna(0)

    # Reorder columns logically
    bucket_order = ['1-10¢', '11-20¢', '21-30¢', '31-40¢', '41-50¢',
                    '51-60¢', '61-70¢', '71-80¢', '81-90¢', '91-99¢']
    pivot_df = pivot_df[[c for c in bucket_order if c in pivot_df.columns]]

    # Figure 1: Longshot share over time (simple line)
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    quarters = longshot_df["quarter"].values
    x = np.arange(len(quarters))
    quarter_labels = [f"{pd.Timestamp(q).year} Q{(pd.Timestamp(q).month-1)//3+1}"
                      for q in quarters]

    ax1.plot(x, longshot_df["longshot_share"], color="#9b59b6",
             linewidth=2, marker="o", markersize=6)
    ax1.fill_between(x, longshot_df["longshot_share"], alpha=0.3, color="#9b59b6")

    # Mark election
    election_idx = None
    for i, q in enumerate(quarters):
        ts = pd.Timestamp(q)
        if ts.year == 2024 and ts.month == 10:
            election_idx = i
            break

    if election_idx is not None:
        ax1.axvline(x=election_idx, color="blue", linestyle=":", linewidth=1.5, alpha=0.7)
        ax1.annotate("2024 Election", xy=(election_idx, ax1.get_ylim()[1] * 0.9),
                     fontsize=9, ha="center", color="blue")

    ax1.set_xlabel("Quarter")
    ax1.set_ylabel("Longshot Volume Share (%)")
    ax1.set_title("Taker Volume Share in Longshot Contracts (1-20¢)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(quarter_labels, rotation=45, ha="right")
    ax1.set_ylim(0, max(longshot_df["longshot_share"]) * 1.1)

    plt.tight_layout()
    fig1.savefig(fig_dir / "longshot_volume_share_over_time.png", dpi=300, bbox_inches="tight")
    fig1.savefig(fig_dir / "longshot_volume_share_over_time.pdf", bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Stacked area chart of all buckets
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(pivot_df.columns)))
    ax2.stackplot(range(len(pivot_df)), pivot_df.T.values, labels=pivot_df.columns,
                  colors=colors, alpha=0.8)

    quarter_labels_pivot = [f"{pd.Timestamp(q).year} Q{(pd.Timestamp(q).month-1)//3+1}"
                            for q in pivot_df.index]
    ax2.set_xticks(range(len(pivot_df)))
    ax2.set_xticklabels(quarter_labels_pivot, rotation=45, ha="right")
    ax2.set_xlabel("Quarter")
    ax2.set_ylabel("Volume Share (%)")
    ax2.set_title("Taker Volume Distribution by Price Bucket")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    fig2.savefig(fig_dir / "volume_distribution_by_price_bucket.png", dpi=300, bbox_inches="tight")
    fig2.savefig(fig_dir / "volume_distribution_by_price_bucket.pdf", bbox_inches="tight")
    plt.close(fig2)

    # JSON output for paper - stacked bar chart showing all price buckets
    # Filter out quarters with < $1M volume (artifacts from early platform)
    valid_quarters = set(pd.to_datetime(quarterly_totals[quarterly_totals["total_volume"] >= 1e6]["quarter"]))
    pivot_filtered = pivot_df[pivot_df.index.isin(valid_quarters)]

    # Use snake_case keys for consistency
    bucket_keys = ["1_10", "11_20", "21_30", "31_40", "41_50",
                   "51_60", "61_70", "71_80", "81_90", "91_99"]
    bucket_labels = ["1-10¢", "11-20¢", "21-30¢", "31-40¢", "41-50¢",
                     "51-60¢", "61-70¢", "71-80¢", "81-90¢", "91-99¢"]
    bucket_key_map = dict(zip(bucket_order, bucket_keys))

    json_data = {
        "type": "stacked-bar-100",
        "title": "Taker Volume Distribution by Price",
        "xKey": "quarter",
        "yKeys": bucket_keys,
        "yLabels": bucket_labels,
        "data": [
            {
                "quarter": f"{pd.Timestamp(q).year} Q{(pd.Timestamp(q).month-1)//3+1}",
                **{bucket_key_map[bucket]: round(pivot_filtered.loc[q, bucket], 2)
                   if bucket in pivot_filtered.columns else 0
                   for bucket in bucket_order}
            }
            for q in pivot_filtered.index
        ],
    }
    with open(fig_dir / "longshot_volume_share_over_time.json", "w") as f:
        json.dump(json_data, f, indent=2)

    # Print summary
    print(f"Outputs saved to {fig_dir}")
    print("\n=== Longshot Volume Share by Quarter ===")
    print(f"{'Quarter':<12} {'Longshot %':>12} {'Volume ($M)':>12}")
    print("-" * 40)
    for _, row in longshot_df.iterrows():
        ts = pd.Timestamp(row["quarter"])
        q_str = f"{ts.year} Q{(ts.month-1)//3+1}"
        print(f"{q_str:<12} {row['longshot_share']:>11.1f}% ${row['total_volume']/1e6:>10.1f}M")

    # Summary statistics
    pre_election = longshot_df[longshot_df["quarter"] < "2024-10-01"]["longshot_share"]
    post_election = longshot_df[longshot_df["quarter"] >= "2024-10-01"]["longshot_share"]

    print(f"\n=== Summary ===")
    print(f"Pre-election avg longshot share: {pre_election.mean():.1f}%")
    print(f"Post-election avg longshot share: {post_election.mean():.1f}%")
    print(f"Change: {post_election.mean() - pre_election.mean():+.1f} pp")

    # Correlation with time
    from scipy import stats
    x_numeric = np.arange(len(longshot_df))
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x_numeric, longshot_df["longshot_share"]
    )
    print(f"\nLinear trend: {slope:+.2f} pp/quarter")
    print(f"R-squared: {r_value**2:.3f}")
    print(f"P-value: {p_value:.4f}")


if __name__ == "__main__":
    main()
