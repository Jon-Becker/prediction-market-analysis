#!/usr/bin/env python3
"""Compare expected value of YES vs NO bets at each price level.

Analyzes whether there's an EV advantage to betting YES vs NO at different price points.
Includes both maker and taker sides of all trades.

EV Formula:
For a bet at price P with actual win rate W:
  EV = W * (100 - P) - (1 - W) * P = 100*W - P

If perfectly calibrated (W = P/100), EV = 0.
Longshot bias means W < P/100 for low P (negative EV for YES longshots).
"""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def main():
    base_dir = Path(__file__).parent.parent.parent
    trades_dir = base_dir / "data" / "trades"
    markets_dir = base_dir / "data" / "markets"
    fig_dir = base_dir / "research" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Calculate YES win rate at each yes_price
    # Every trade has a yes_price; YES wins if market result = 'yes'
    # Both maker and taker sides are implicitly included since we aggregate by price
    yes_df = con.execute(
        f"""
        SELECT
            t.yes_price AS price,
            SUM(CASE WHEN m.result = 'yes' THEN t.count ELSE 0 END) * 1.0 / SUM(t.count) AS win_rate,
            SUM(t.count) AS total_contracts
        FROM '{trades_dir}/*.parquet' t
        INNER JOIN '{markets_dir}/*.parquet' m ON t.ticker = m.ticker
        WHERE m.result IN ('yes', 'no')
          AND t.yes_price BETWEEN 1 AND 99
        GROUP BY t.yes_price
        ORDER BY t.yes_price
        """
    ).df()

    # Calculate NO win rate at each no_price
    # Every trade has a no_price = 100 - yes_price; NO wins if market result = 'no'
    no_df = con.execute(
        f"""
        SELECT
            t.no_price AS price,
            SUM(CASE WHEN m.result = 'no' THEN t.count ELSE 0 END) * 1.0 / SUM(t.count) AS win_rate,
            SUM(t.count) AS total_contracts
        FROM '{trades_dir}/*.parquet' t
        INNER JOIN '{markets_dir}/*.parquet' m ON t.ticker = m.ticker
        WHERE m.result IN ('yes', 'no')
          AND t.no_price BETWEEN 1 AND 99
        GROUP BY t.no_price
        ORDER BY t.no_price
        """
    ).df()

    # Calculate EV = 100 * win_rate - price
    # This is the expected profit/loss per contract in cents
    yes_df["ev"] = 100 * yes_df["win_rate"] - yes_df["price"]
    no_df["ev"] = 100 * no_df["win_rate"] - no_df["price"]

    # Also calculate implied vs actual
    yes_df["implied_prob"] = yes_df["price"] / 100
    yes_df["actual_prob"] = yes_df["win_rate"]
    no_df["implied_prob"] = no_df["price"] / 100
    no_df["actual_prob"] = no_df["win_rate"]

    # Calculate statistical significance (z-test for proportion)
    # H0: actual win rate = implied probability (price/100)
    # z = (observed - expected) / sqrt(expected * (1 - expected) / n)
    def calc_stats(df):
        p_expected = df["implied_prob"]
        p_observed = df["actual_prob"]
        n = df["total_contracts"]

        # Standard error under null hypothesis
        se = np.sqrt(p_expected * (1 - p_expected) / n)

        # Z-score
        df["z_score"] = (p_observed - p_expected) / se

        # Two-tailed p-value
        df["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(df["z_score"])))

        # 95% confidence interval for actual win rate (using observed proportion)
        se_obs = np.sqrt(p_observed * (1 - p_observed) / n)
        df["ci_lower"] = p_observed - 1.96 * se_obs
        df["ci_upper"] = p_observed + 1.96 * se_obs

        # Whether implied prob falls within CI (i.e., NOT significantly different)
        df["calibrated"] = (df["implied_prob"] >= df["ci_lower"]) & (
            df["implied_prob"] <= df["ci_upper"]
        )

        return df

    yes_df = calc_stats(yes_df)
    no_df = calc_stats(no_df)

    # Save data
    yes_df.to_csv(fig_dir / "ev_yes_by_price.csv", index=False)
    no_df.to_csv(fig_dir / "ev_no_by_price.csv", index=False)

    # === Figure 1: EV comparison with stacked area effect ===
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot lines
    ax.plot(yes_df["price"], yes_df["ev"], label="YES bets", color="#2ecc71", linewidth=2.5)
    ax.plot(no_df["price"], no_df["ev"], label="NO bets", color="#e74c3c", linewidth=2.5)

    # Fill areas to zero for visual effect
    ax.fill_between(yes_df["price"], yes_df["ev"], 0, alpha=0.3, color="#2ecc71")
    ax.fill_between(no_df["price"], no_df["ev"], 0, alpha=0.3, color="#e74c3c")

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=1)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Purchase Price (cents)")
    ax.set_ylabel("Expected Value (cents per contract)")
    ax.set_title("Expected Value of YES vs NO Bets by Price Level\n(Including both maker and taker sides)")
    ax.set_xlim(1, 99)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add annotations for key insights
    yes_min_idx = yes_df["ev"].idxmin()
    yes_min_price = yes_df.loc[yes_min_idx, "price"]
    yes_min_ev = yes_df.loc[yes_min_idx, "ev"]
    ax.annotate(f"YES worst: {yes_min_ev:.1f}¢ at {yes_min_price}¢",
                xy=(yes_min_price, yes_min_ev), xytext=(yes_min_price + 15, yes_min_ev - 3),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))

    no_max_idx = no_df["ev"].idxmax()
    no_max_price = no_df.loc[no_max_idx, "price"]
    no_max_ev = no_df.loc[no_max_idx, "ev"]
    ax.annotate(f"NO best: +{no_max_ev:.1f}¢ at {no_max_price}¢",
                xy=(no_max_price, no_max_ev), xytext=(no_max_price - 20, no_max_ev + 3),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))

    plt.tight_layout()
    fig.savefig(fig_dir / "ev_yes_vs_no.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "ev_yes_vs_no.pdf", bbox_inches="tight")
    plt.close(fig)

    # === Figure 2: Direct comparison on same scale ===
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create symmetric view: at each price P
    # - YES EV at P: betting event happens when priced at P
    # - NO EV at P: betting event doesn't happen when priced at P (underlying yes = 100-P)

    ax.stackplot(yes_df["price"],
                 np.maximum(yes_df["ev"], 0),
                 np.maximum(no_df["ev"], 0),
                 labels=["YES +EV", "NO +EV"],
                 colors=["#2ecc71", "#e74c3c"],
                 alpha=0.7)

    ax.set_xlabel("Purchase Price (cents)")
    ax.set_ylabel("Expected Value (cents per contract)")
    ax.set_title("Positive Expected Value by Bet Type and Price")
    ax.set_xlim(1, 99)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "ev_yes_vs_no_stacked.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "ev_yes_vs_no_stacked.pdf", bbox_inches="tight")
    plt.close(fig)

    # === Figure 3: Combined EV by price (risk) ===
    # At each price P, show EV for both YES at P and NO at P
    # This answers: "If I'm paying P cents, should I bet YES or NO?"
    combined_df = pd.DataFrame({"price": range(1, 100)})
    combined_df = combined_df.merge(
        yes_df[["price", "ev", "actual_prob", "total_contracts", "z_score"]].rename(
            columns={
                "ev": "yes_ev",
                "actual_prob": "yes_win_rate",
                "total_contracts": "yes_contracts",
                "z_score": "yes_z",
            }
        ),
        on="price",
        how="left",
    )
    combined_df = combined_df.merge(
        no_df[["price", "ev", "actual_prob", "total_contracts", "z_score"]].rename(
            columns={
                "ev": "no_ev",
                "actual_prob": "no_win_rate",
                "total_contracts": "no_contracts",
                "z_score": "no_z",
            }
        ),
        on="price",
        how="left",
    )

    # Calculate combined metrics
    combined_df["implied_prob"] = combined_df["price"] / 100
    combined_df["best_ev"] = np.maximum(combined_df["yes_ev"], combined_df["no_ev"])
    combined_df["best_bet"] = np.where(
        combined_df["yes_ev"] > combined_df["no_ev"], "YES", "NO"
    )
    combined_df["ev_diff"] = combined_df["yes_ev"] - combined_df["no_ev"]

    # Save combined CSV
    combined_df.to_csv(fig_dir / "ev_combined_by_price.csv", index=False)

    # Plot combined figure
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        combined_df["price"],
        combined_df["yes_ev"],
        label="YES bet at this price",
        color="#2ecc71",
        linewidth=2,
    )
    ax.plot(
        combined_df["price"],
        combined_df["no_ev"],
        label="NO bet at this price",
        color="#e74c3c",
        linewidth=2,
    )
    ax.plot(
        combined_df["price"],
        combined_df["best_ev"],
        label="Best available bet",
        color="#3498db",
        linewidth=2.5,
        linestyle="--",
    )

    ax.fill_between(
        combined_df["price"],
        combined_df["yes_ev"],
        combined_df["no_ev"],
        alpha=0.2,
        color="gray",
    )

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=1)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.3)

    ax.set_xlabel("Price Paid (cents) = Risk")
    ax.set_ylabel("Expected Value (cents per contract)")
    ax.set_title("Combined EV by Price: YES vs NO at Same Cost\n(At price P: YES costs P¢, NO costs P¢)")
    ax.set_xlim(1, 99)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Annotate best overall
    best_idx = combined_df["best_ev"].idxmax()
    best_price = combined_df.loc[best_idx, "price"]
    best_ev_val = combined_df.loc[best_idx, "best_ev"]
    best_type = combined_df.loc[best_idx, "best_bet"]
    ax.annotate(
        f"Best: {best_type} at {best_price}¢ → +{best_ev_val:.1f}¢ EV",
        xy=(best_price, best_ev_val),
        xytext=(best_price + 10, best_ev_val + 2),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    plt.tight_layout()
    fig.savefig(fig_dir / "ev_combined_by_price.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "ev_combined_by_price.pdf", bbox_inches="tight")
    plt.close(fig)

    # === Figure 4: Calibration plot (actual vs implied probability) ===
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax.scatter(
        yes_df["implied_prob"],
        yes_df["actual_prob"],
        label="YES bets",
        color="#2ecc71",
        alpha=0.7,
        s=40,
    )
    ax.scatter(
        no_df["implied_prob"],
        no_df["actual_prob"],
        label="NO bets",
        color="#e74c3c",
        alpha=0.7,
        s=40,
    )

    ax.set_xlabel("Implied Probability (price / 100)")
    ax.set_ylabel("Actual Win Rate")
    ax.set_title("Calibration Plot: Implied vs Actual Probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "ev_calibration.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "ev_calibration.pdf", bbox_inches="tight")
    plt.close(fig)

    # === Create JSON for paper (ev_yes_vs_no.json) ===
    json_data = []
    for price in range(1, 100):
        yes_row = yes_df[yes_df["price"] == price]
        no_row = no_df[no_df["price"] == price]

        entry = {"price": price}
        if len(yes_row) > 0:
            entry["yes_ev"] = round(float(yes_row["ev"].values[0]), 2)
        else:
            entry["yes_ev"] = 0
        if len(no_row) > 0:
            entry["no_ev"] = round(float(no_row["ev"].values[0]), 2)
        else:
            entry["no_ev"] = 0

        json_data.append(entry)

    json_output = {
        "type": "line",
        "title": "Expected Value: YES vs NO Bets by Price",
        "xKey": "price",
        "xUnit": "cents",
        "yKeys": ["yes_ev", "no_ev"],
        "yUnit": "cents",
        "data": json_data,
    }

    json_path = fig_dir / "ev_yes_vs_no.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"JSON saved to {json_path}")

    # === Create JSON for longshot_ev_asymmetry.json (percentage returns) ===
    # Historical return = (100 * win_rate - price) / price * 100
    longshot_data = []
    for price in range(1, 100):
        yes_row = yes_df[yes_df["price"] == price]
        no_row = no_df[no_df["price"] == price]

        entry = {"price": price}
        if len(yes_row) > 0:
            yes_ev = float(yes_row["ev"].values[0])
            entry["yes_return"] = round(yes_ev / price * 100, 2)
        else:
            entry["yes_return"] = 0
        if len(no_row) > 0:
            no_ev = float(no_row["ev"].values[0])
            entry["no_return"] = round(no_ev / price * 100, 2)
        else:
            entry["no_return"] = 0

        longshot_data.append(entry)

    longshot_output = {
        "type": "line",
        "title": "Historical Return by Contract Price: YES vs NO",
        "xKey": "price",
        "xLabel": "Cost Basis (cents)",
        "yKeys": ["yes_return", "no_return"],
        "yLabels": ["YES Bet", "NO Bet"],
        "yUnit": "percent",
        "yLabel": "Historical Return (%)",
        "data": longshot_data,
    }

    longshot_path = fig_dir / "longshot_ev_asymmetry.json"
    with open(longshot_path, "w") as f:
        json.dump(longshot_output, f, indent=2)
    print(f"JSON saved to {longshot_path}")

    print(f"Figures saved to {fig_dir}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: Expected Value Analysis")
    print("=" * 70)

    print("\nYES Bets:")
    print(f"  Best EV: {yes_df['ev'].max():.2f}¢ at {yes_df.loc[yes_df['ev'].idxmax(), 'price']}¢")
    print(f"  Worst EV: {yes_df['ev'].min():.2f}¢ at {yes_df.loc[yes_df['ev'].idxmin(), 'price']}¢")
    print(f"  EV at 50¢: {yes_df[yes_df['price'] == 50]['ev'].values[0]:.2f}¢")

    print("\nNO Bets:")
    print(f"  Best EV: {no_df['ev'].max():.2f}¢ at {no_df.loc[no_df['ev'].idxmax(), 'price']}¢")
    print(f"  Worst EV: {no_df['ev'].min():.2f}¢ at {no_df.loc[no_df['ev'].idxmin(), 'price']}¢")
    print(f"  EV at 50¢: {no_df[no_df['price'] == 50]['ev'].values[0]:.2f}¢")

    # Key insight: where does YES EV cross from negative to positive?
    yes_positive = yes_df[yes_df["ev"] > 0]
    if len(yes_positive) > 0:
        print(f"\nYES becomes +EV above: {yes_positive['price'].min()}¢")

    no_positive = no_df[no_df["ev"] > 0]
    if len(no_positive) > 0:
        print(f"NO becomes +EV above: {no_positive['price'].min()}¢")

    # Volume-weighted EV
    yes_vol_ev = (yes_df["ev"] * yes_df["total_contracts"]).sum() / yes_df["total_contracts"].sum()
    no_vol_ev = (no_df["ev"] * no_df["total_contracts"]).sum() / no_df["total_contracts"].sum()
    print(f"\nVolume-weighted average EV:")
    print(f"  YES bets: {yes_vol_ev:.2f}¢ per contract")
    print(f"  NO bets: {no_vol_ev:.2f}¢ per contract")

    # Statistical significance summary
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 70)

    yes_calibrated = yes_df["calibrated"].sum()
    no_calibrated = no_df["calibrated"].sum()
    print(f"\nPrice levels where implied prob falls within 95% CI (calibrated):")
    print(f"  YES bets: {yes_calibrated}/99 ({yes_calibrated/99*100:.1f}%)")
    print(f"  NO bets: {no_calibrated}/99 ({no_calibrated/99*100:.1f}%)")

    print("\nSample z-scores (deviation from perfect calibration):")
    print("  Price | YES z-score | NO z-score | Interpretation")
    print("  " + "-" * 60)
    for price in [10, 25, 50, 75, 90]:
        yes_z = yes_df[yes_df["price"] == price]["z_score"].values[0]
        no_z = no_df[no_df["price"] == price]["z_score"].values[0]
        yes_dir = "undercalibrated" if yes_z < 0 else "overcalibrated"
        no_dir = "undercalibrated" if no_z < 0 else "overcalibrated"
        print(f"  {price:3d}¢  | {yes_z:+10.1f} | {no_z:+10.1f} | YES {yes_dir}")

    # Overall test: are deviations systematic?
    yes_neg = (yes_df["z_score"] < 0).sum()
    no_neg = (no_df["z_score"] < 0).sum()
    print(f"\nDirection of miscalibration:")
    print(f"  YES: {yes_neg}/99 price levels undercalibrated (actual < implied)")
    print(f"  NO: {no_neg}/99 price levels undercalibrated (actual < implied)")

    # Average absolute z-score (measure of overall miscalibration)
    yes_avg_z = np.abs(yes_df["z_score"]).mean()
    no_avg_z = np.abs(no_df["z_score"]).mean()
    print(f"\nMean |z-score| (higher = more miscalibrated):")
    print(f"  YES bets: {yes_avg_z:.1f}")
    print(f"  NO bets: {no_avg_z:.1f}")

    print("\nNote: With n > 100M contracts per price level, even tiny deviations")
    print("from perfect calibration produce massive z-scores. All p-values ≈ 0.")

    # Combined analysis summary
    print("\n" + "=" * 70)
    print("COMBINED ANALYSIS: Best Bet at Each Price")
    print("=" * 70)

    yes_better = (combined_df["yes_ev"] > combined_df["no_ev"]).sum()
    no_better = 99 - yes_better
    print(f"\nPrice levels where YES is better: {yes_better}/99")
    print(f"Price levels where NO is better: {no_better}/99")

    # Best overall opportunities
    print("\nTop 5 +EV opportunities (by price paid):")
    top5 = combined_df.nlargest(5, "best_ev")[["price", "best_bet", "best_ev"]]
    for _, row in top5.iterrows():
        print(f"  {row['best_bet']} at {int(row['price'])}¢: +{row['best_ev']:.2f}¢ EV")

    # Worst opportunities
    print("\nWorst 5 -EV traps (by price paid):")
    worst_yes = combined_df.nsmallest(5, "yes_ev")[["price", "yes_ev"]]
    worst_no = combined_df.nsmallest(5, "no_ev")[["price", "no_ev"]]
    combined_worst = []
    for _, row in worst_yes.iterrows():
        combined_worst.append(("YES", int(row["price"]), row["yes_ev"]))
    for _, row in worst_no.iterrows():
        combined_worst.append(("NO", int(row["price"]), row["no_ev"]))
    combined_worst.sort(key=lambda x: x[2])
    for bet, price, ev in combined_worst[:5]:
        print(f"  {bet} at {price}¢: {ev:.2f}¢ EV")

    # Average EV if you always pick the best bet
    avg_best_ev = combined_df["best_ev"].mean()
    print(f"\nAverage EV if always picking best bet: +{avg_best_ev:.2f}¢ per contract")


if __name__ == "__main__":
    main()
