"""Compare expected value of YES vs NO bets at each price level.

Analyzes whether there's an EV advantage to betting YES vs NO at different price points.
Includes both maker and taker sides of all trades.

EV Formula:
For a bet at price P with actual win rate W:
  EV = W * (100 - P) - (1 - W) * P = 100*W - P

If perfectly calibrated (W = P/100), EV = 0.
Longshot bias means W < P/100 for low P (negative EV for YES longshots).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class EvYesVsNoAnalysis(Analysis):
    """Analyze expected value of YES vs NO bets by price on Kalshi."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="ev_yes_vs_no",
            description="Expected value comparison of YES vs NO bets by price level",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Calculate YES win rate at each yes_price
        yes_df = con.execute(
            f"""
            SELECT
                t.yes_price AS price,
                SUM(CASE WHEN m.result = 'yes' THEN t.count ELSE 0 END) * 1.0 / SUM(t.count) AS win_rate,
                SUM(t.count) AS total_contracts
            FROM '{self.trades_dir}/*.parquet' t
            INNER JOIN '{self.markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE m.result IN ('yes', 'no')
              AND t.yes_price BETWEEN 1 AND 99
            GROUP BY t.yes_price
            ORDER BY t.yes_price
            """
        ).df()

        # Calculate NO win rate at each no_price
        no_df = con.execute(
            f"""
            SELECT
                t.no_price AS price,
                SUM(CASE WHEN m.result = 'no' THEN t.count ELSE 0 END) * 1.0 / SUM(t.count) AS win_rate,
                SUM(t.count) AS total_contracts
            FROM '{self.trades_dir}/*.parquet' t
            INNER JOIN '{self.markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE m.result IN ('yes', 'no')
              AND t.no_price BETWEEN 1 AND 99
            GROUP BY t.no_price
            ORDER BY t.no_price
            """
        ).df()

        # Calculate EV = 100 * win_rate - price
        yes_df["ev"] = 100 * yes_df["win_rate"] - yes_df["price"]
        no_df["ev"] = 100 * no_df["win_rate"] - no_df["price"]

        yes_df["implied_prob"] = yes_df["price"] / 100
        yes_df["actual_prob"] = yes_df["win_rate"]
        no_df["implied_prob"] = no_df["price"] / 100
        no_df["actual_prob"] = no_df["win_rate"]

        # Create combined dataframe for output
        combined_df = pd.DataFrame({"price": range(1, 100)})
        combined_df = combined_df.merge(
            yes_df[["price", "ev", "actual_prob", "total_contracts"]].rename(
                columns={
                    "ev": "yes_ev",
                    "actual_prob": "yes_win_rate",
                    "total_contracts": "yes_contracts",
                }
            ),
            on="price",
            how="left",
        )
        combined_df = combined_df.merge(
            no_df[["price", "ev", "actual_prob", "total_contracts"]].rename(
                columns={
                    "ev": "no_ev",
                    "actual_prob": "no_win_rate",
                    "total_contracts": "no_contracts",
                }
            ),
            on="price",
            how="left",
        )

        combined_df["implied_prob"] = combined_df["price"] / 100
        combined_df["best_ev"] = np.maximum(
            combined_df["yes_ev"].fillna(-100), combined_df["no_ev"].fillna(-100)
        )
        combined_df["best_bet"] = np.where(
            combined_df["yes_ev"].fillna(-100) > combined_df["no_ev"].fillna(-100),
            "YES",
            "NO",
        )

        fig = self._create_figure(yes_df, no_df)
        chart = self._create_chart(yes_df, no_df)

        return AnalysisOutput(figure=fig, data=combined_df, chart=chart)

    def _create_figure(
        self, yes_df: pd.DataFrame, no_df: pd.DataFrame
    ) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(
            yes_df["price"], yes_df["ev"], label="YES bets", color="#2ecc71", linewidth=2.5
        )
        ax.plot(
            no_df["price"], no_df["ev"], label="NO bets", color="#e74c3c", linewidth=2.5
        )

        ax.fill_between(yes_df["price"], yes_df["ev"], 0, alpha=0.3, color="#2ecc71")
        ax.fill_between(no_df["price"], no_df["ev"], 0, alpha=0.3, color="#e74c3c")

        ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=1)
        ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)

        ax.set_xlabel("Purchase Price (cents)")
        ax.set_ylabel("Expected Value (cents per contract)")
        ax.set_title(
            "Expected Value of YES vs NO Bets by Price Level\n"
            "(Including both maker and taker sides)"
        )
        ax.set_xlim(1, 99)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Add annotations for key insights
        yes_min_idx = yes_df["ev"].idxmin()
        yes_min_price = yes_df.loc[yes_min_idx, "price"]
        yes_min_ev = yes_df.loc[yes_min_idx, "ev"]
        ax.annotate(
            f"YES worst: {yes_min_ev:.1f} at {yes_min_price}c",
            xy=(yes_min_price, yes_min_ev),
            xytext=(yes_min_price + 15, yes_min_ev - 3),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

        no_max_idx = no_df["ev"].idxmax()
        no_max_price = no_df.loc[no_max_idx, "price"]
        no_max_ev = no_df.loc[no_max_idx, "ev"]
        ax.annotate(
            f"NO best: +{no_max_ev:.1f} at {no_max_price}c",
            xy=(no_max_price, no_max_ev),
            xytext=(no_max_price - 20, no_max_ev + 3),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

        plt.tight_layout()
        return fig

    def _create_chart(
        self, yes_df: pd.DataFrame, no_df: pd.DataFrame
    ) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = []
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

            chart_data.append(entry)

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="price",
            yKeys=["yes_ev", "no_ev"],
            title="Expected Value: YES vs NO Bets by Price",
            yUnit=UnitType.CENTS,
            xLabel="Purchase Price (cents)",
            yLabel="Expected Value (cents per contract)",
            colors={"yes_ev": "#2ecc71", "no_ev": "#e74c3c"},
        )
