#!/usr/bin/env python3
"""Analyze risk-adjusted USD volume vs win rate.

Risk adjustment scales volume by price variance (p*(1-p)), which captures
the uncertainty/risk of the trade. Bets near 50% have maximum risk weighting,
while bets near 0% or 100% have minimal risk weighting.
Includes z-tests for statistical significance.
"""

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

    df = con.execute(
        f"""
        WITH trade_data AS (
            SELECT
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS trade_size_usd,
                (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS price_frac,
                CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS expected_win_rate
            FROM '{trades_dir}/*.parquet' t
            JOIN '{markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE m.result IN ('yes', 'no')
        ),
        risk_adjusted AS (
            SELECT
                -- Risk-adjusted volume: scale by 4*p*(1-p), normalized so max=1 at p=0.5
                trade_size_usd * 4.0 * price_frac * (1.0 - price_frac) AS risk_adj_volume,
                won,
                expected_win_rate
            FROM trade_data
            WHERE trade_size_usd > 0
        ),
        binned AS (
            SELECT
                POWER(10, FLOOR(LOG10(GREATEST(risk_adj_volume, 0.01)) * 4) / 4.0) AS bin_lower,
                AVG(won) AS win_rate,
                AVG(expected_win_rate) AS expected_win_rate,
                AVG(won - expected_win_rate) AS excess_win_rate,
                VAR_POP(won - expected_win_rate) AS var_excess,
                COUNT(*) AS n_trades
            FROM risk_adjusted
            GROUP BY bin_lower
            HAVING COUNT(*) >= 1000
        )
        SELECT bin_lower AS risk_adj_volume_bin, win_rate, expected_win_rate, excess_win_rate, var_excess, n_trades
        FROM binned
        ORDER BY bin_lower
        """
    ).df()

    # Z-test: H0: excess_win_rate = 0
    df["se_excess"] = np.sqrt(df["var_excess"] / df["n_trades"])
    df["z_stat"] = df["excess_win_rate"] / df["se_excess"]
    df["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(df["z_stat"])))
    df["significant"] = df["p_value"] < 0.05

    df.to_csv(fig_dir / "risk_adjusted_volume_vs_win_rate.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    sizes = np.clip(df["n_trades"] / df["n_trades"].max() * 200, 10, 200)
    colors = ["#D65F5F" if sig else "#4C72B0" for sig in df["significant"]]
    ax.scatter(
        df["risk_adj_volume_bin"],
        df["excess_win_rate"] * 100,
        s=sizes,
        alpha=0.6,
        c=colors,
        edgecolor="none",
    )
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    x = df["risk_adj_volume_bin"].values
    y = df["excess_win_rate"].values * 100
    weights = df["n_trades"].values
    log_x = np.log10(x)
    coeffs = np.polyfit(log_x, y, 1, w=np.sqrt(weights))
    x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    y_fit = np.polyval(coeffs, np.log10(x_fit))
    ax.plot(x_fit, y_fit, color="#C44E52", linewidth=2, label=f"fit: {coeffs[0]:.2f}*log10(x) + {coeffs[1]:.2f}")
    ax.legend(loc="lower right")

    ax.set_xscale("log")
    ax.set_xlabel("Risk-Adjusted Volume (USD)")
    ax.set_ylabel("Excess Win Rate (pp)")
    ax.set_title("Risk-Adjusted Volume vs Excess Win Rate\n(red = p < 0.05)")

    plt.tight_layout()
    fig.savefig(fig_dir / "risk_adjusted_volume_vs_win_rate.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "risk_adjusted_volume_vs_win_rate.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")
    print(f"\nStatistical Summary:")
    print(f"  H0: Excess win rate = 0")
    for _, row in df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"  ${row['risk_adj_volume_bin']:>10.2f}: excess={row['excess_win_rate']*100:+.3f}pp, z={row['z_stat']:.2f}, p={row['p_value']:.2e} {sig}")


if __name__ == "__main__":
    main()
