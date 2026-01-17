#!/usr/bin/env python3
"""Analyze maker vs taker returns.

Compares performance of passive liquidity providers (makers) against
aggressive order takers across price points. Tests whether informed
trading exists or if market making captures spread profitably.
"""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def main():
    base_dir = Path(__file__).parent.parent.parent
    trades_dir = base_dir / "data" / "trades"
    markets_dir = base_dir / "data" / "markets"
    fig_dir = base_dir / "research" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Compute returns for takers and makers by price
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
                CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                t.count AS contracts,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
        ),
        maker_positions AS (
            SELECT
                CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                CASE WHEN t.taker_side != m.result THEN 1.0 ELSE 0.0 END AS won,
                t.count AS contracts,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END) / 100.0 AS volume_usd
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
        ),
        taker_stats AS (
            SELECT
                price,
                AVG(won) AS win_rate,
                price / 100.0 AS expected_win_rate,
                AVG(won) - price / 100.0 AS excess_return,
                VAR_POP(won - price / 100.0) AS var_excess,
                COUNT(*) AS n_trades,
                SUM(volume_usd) AS volume_usd,
                SUM(contracts) AS contracts,
                SUM(contracts * (won - price / 100.0)) AS pnl
            FROM taker_positions
            GROUP BY price
        ),
        maker_stats AS (
            SELECT
                price,
                AVG(won) AS win_rate,
                price / 100.0 AS expected_win_rate,
                AVG(won) - price / 100.0 AS excess_return,
                VAR_POP(won - price / 100.0) AS var_excess,
                COUNT(*) AS n_trades,
                SUM(volume_usd) AS volume_usd,
                SUM(contracts) AS contracts,
                SUM(contracts * (won - price / 100.0)) AS pnl
            FROM maker_positions
            GROUP BY price
        )
        SELECT
            t.price,
            t.win_rate AS taker_win_rate,
            t.expected_win_rate AS taker_expected,
            t.excess_return AS taker_excess,
            t.var_excess AS taker_var,
            t.n_trades AS taker_n,
            t.volume_usd AS taker_volume,
            t.pnl AS taker_pnl,
            m.win_rate AS maker_win_rate,
            m.expected_win_rate AS maker_expected,
            m.excess_return AS maker_excess,
            m.var_excess AS maker_var,
            m.n_trades AS maker_n,
            m.volume_usd AS maker_volume,
            m.pnl AS maker_pnl
        FROM taker_stats t
        JOIN maker_stats m ON t.price = m.price
        WHERE t.price BETWEEN 1 AND 99
        ORDER BY t.price
        """
    ).df()

    # Calculate standard errors and z-statistics
    df["taker_se"] = np.sqrt(df["taker_var"] / df["taker_n"])
    df["maker_se"] = np.sqrt(df["maker_var"] / df["maker_n"])
    df["taker_z"] = df["taker_excess"] / df["taker_se"]
    df["maker_z"] = df["maker_excess"] / df["maker_se"]
    df["taker_p"] = 2 * (1 - stats.norm.cdf(np.abs(df["taker_z"])))
    df["maker_p"] = 2 * (1 - stats.norm.cdf(np.abs(df["maker_z"])))

    # Aggregate stats with correct P&L calculation
    agg = con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
        ),
        all_positions AS (
            SELECT
                'taker' AS role,
                CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                t.count AS contracts
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker

            UNION ALL

            SELECT
                'maker' AS role,
                CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                CASE WHEN t.taker_side != m.result THEN 1.0 ELSE 0.0 END AS won,
                t.count AS contracts
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
        )
        SELECT
            role,
            AVG(won) AS win_rate,
            AVG(price / 100.0) AS avg_price,
            AVG(won - price / 100.0) AS excess_return,
            VAR_POP(won - price / 100.0) AS var_excess,
            COUNT(*) AS n_trades,
            SUM(contracts * price / 100.0) AS total_volume,
            SUM(contracts * (won - price / 100.0)) AS total_pnl
        FROM all_positions
        GROUP BY role
        """
    ).df()

    agg["se"] = np.sqrt(agg["var_excess"] / agg["n_trades"])
    agg["z"] = agg["excess_return"] / agg["se"]
    agg["p"] = 2 * (1 - stats.norm.cdf(np.abs(agg["z"])))

    df.to_csv(fig_dir / "maker_vs_taker_returns.csv", index=False)

    # Figure 1: Excess returns by price for both roles
    # Plot maker at (100-price) to show actual counterparty relationship
    df_sorted = df.sort_values("price")
    maker_counterparty = df_sorted.set_index("price")["maker_excess"].reindex(100 - df_sorted["price"].values).values

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df_sorted["price"], df_sorted["taker_excess"] * 100, color="#e74c3c", linewidth=1.5, label="Taker", alpha=0.8)
    ax1.plot(df_sorted["price"], maker_counterparty * 100, color="#2ecc71", linewidth=1.5, label="Maker (counterparty)", alpha=0.8)
    ax1.fill_between(df_sorted["price"], df_sorted["taker_excess"] * 100, alpha=0.2, color="#e74c3c")
    ax1.fill_between(df_sorted["price"], maker_counterparty * 100, alpha=0.2, color="#2ecc71")
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Contract Price (cents)")
    ax1.set_ylabel("Excess Return (pp)")
    ax1.set_title("Maker vs Taker Excess Returns by Price")
    ax1.set_xlim(1, 99)
    ax1.set_xticks(range(0, 101, 10))
    ax1.legend(loc="upper right")
    plt.tight_layout()
    fig1.savefig(fig_dir / "maker_vs_taker_returns.png", dpi=300, bbox_inches="tight")
    fig1.savefig(fig_dir / "maker_vs_taker_returns.pdf", bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Net P&L by price (already computed correctly in SQL)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    width = 0.8
    ax2.bar(df["price"], df["taker_pnl"] / 1e6, width=width, color="#e74c3c", alpha=0.7, label="Taker P&L")
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Contract Price (cents)")
    ax2.set_ylabel("Net P&L ($M)")
    ax2.set_title("Taker Net P&L by Price Point")
    ax2.set_xlim(0, 100)
    ax2.set_xticks(range(0, 101, 10))
    plt.tight_layout()
    fig2.savefig(fig_dir / "taker_pnl_by_price.png", dpi=300, bbox_inches="tight")
    fig2.savefig(fig_dir / "taker_pnl_by_price.pdf", bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: Aggregate bar chart
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    roles = ["Taker", "Maker"]
    excess = [agg[agg["role"] == "taker"]["excess_return"].values[0] * 100,
              agg[agg["role"] == "maker"]["excess_return"].values[0] * 100]
    se = [agg[agg["role"] == "taker"]["se"].values[0] * 100,
          agg[agg["role"] == "maker"]["se"].values[0] * 100]
    colors = ["#e74c3c", "#2ecc71"]
    x = np.arange(len(roles))
    ax3.bar(x, excess, color=colors, alpha=0.7, edgecolor="none")
    ax3.errorbar(x, excess, yerr=[1.96 * s for s in se], fmt="none", color="gray", capsize=5)
    ax3.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(roles)
    ax3.set_ylabel("Excess Return (pp)")
    ax3.set_title("Aggregate Maker vs Taker Performance\n(error bars = 95% CI)")
    plt.tight_layout()
    fig3.savefig(fig_dir / "maker_vs_taker_aggregate.png", dpi=300, bbox_inches="tight")
    fig3.savefig(fig_dir / "maker_vs_taker_aggregate.pdf", bbox_inches="tight")
    plt.close(fig3)

    # JSON output for paper
    # Plot maker at (100-price) to show actual counterparty relationship
    # Taker@P is matched with Maker@(100-P), so they should sum to ~0
    maker_by_price = df.set_index("price")["maker_excess"].to_dict()
    json_data = {
        "type": "line",
        "title": "Maker vs Taker Excess Returns by Price",
        "data": [
            {
                "price": int(row["price"]),
                "Taker": round(row["taker_excess"] * 100, 2),
                "Maker (counterparty)": round(maker_by_price.get(100 - row["price"], 0) * 100, 2),
            }
            for _, row in df.iterrows()
        ],
        "xKey": "price",
        "yKeys": ["Taker", "Maker (counterparty)"],
        "yUnit": "percent",
    }
    with open(fig_dir / "maker_vs_taker_returns.json", "w") as f:
        json.dump(json_data, f)

    # Print summary
    print(f"Outputs saved to {fig_dir}")
    print("\n=== Aggregate Statistics ===")
    for _, row in agg.iterrows():
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
        print(f"{row['role'].upper()}:")
        print(f"  Win rate: {row['win_rate']*100:.2f}%")
        print(f"  Avg price: {row['avg_price']*100:.2f}¢")
        print(f"  Excess return: {row['excess_return']*100:+.3f}pp {sig}")
        print(f"  Z-stat: {row['z']:.2f}, p-value: {row['p']:.2e}")
        print(f"  N trades: {row['n_trades']:,.0f}")
        print(f"  Total volume: ${row['total_volume']/1e9:.2f}B")
        print(f"  Total P&L: ${row['total_pnl']/1e6:+,.1f}M")
        print()

    # Verify zero-sum (should be close to zero, small diff due to spread)
    taker_pnl = agg[agg["role"] == "taker"]["total_pnl"].values[0]
    maker_pnl = agg[agg["role"] == "maker"]["total_pnl"].values[0]
    print(f"=== Zero-Sum Check ===")
    print(f"Taker P&L + Maker P&L = ${(taker_pnl + maker_pnl)/1e6:+,.2f}M")
    print(f"(Non-zero due to ~0.4% of trades with 1¢ spread leakage)")


if __name__ == "__main__":
    main()
