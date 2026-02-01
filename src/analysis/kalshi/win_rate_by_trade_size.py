"""Analyze win rate by trade size with confidence intervals.

Examines whether larger trades are more informed (higher excess win rate),
controlling for contract price. Computes 95% confidence intervals for
each trade size bucket.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, ScaleType, UnitType


class WinRateByTradeSizeAnalysis(Analysis):
    """Analyze win rate by trade size on Kalshi."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="win_rate_by_trade_size",
            description="Win rate by trade size with price adjustment",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Compute excess win rate by trade size bin with variance for CI
        # Uses log-scale bins and controls for price by computing excess win rate
        df = con.execute(
            f"""
            WITH trade_data AS (
                SELECT
                    t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS trade_size_usd,
                    CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                    (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS expected_win_rate
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN '{self.markets_dir}/*.parquet' m ON t.ticker = m.ticker
                WHERE m.status = 'finalized'
                  AND m.result IN ('yes', 'no')
            ),
            binned AS (
                SELECT
                    POWER(10, FLOOR(LOG10(GREATEST(trade_size_usd, 0.01)) * 4) / 4.0) AS bin_lower,
                    AVG(won) AS win_rate,
                    AVG(expected_win_rate) AS expected_win_rate,
                    AVG(won - expected_win_rate) AS excess_win_rate,
                    VAR_SAMP(won - expected_win_rate) AS var_excess,
                    COUNT(*) AS n_trades,
                    SUM(trade_size_usd) AS total_volume
                FROM trade_data
                GROUP BY bin_lower
                HAVING COUNT(*) >= 10
            )
            SELECT
                bin_lower AS trade_size_bin,
                win_rate,
                expected_win_rate,
                excess_win_rate,
                var_excess,
                n_trades,
                total_volume
            FROM binned
            ORDER BY bin_lower
            """
        ).df()

        # Calculate standard error and 95% CI
        df["se"] = np.sqrt(df["var_excess"] / df["n_trades"])
        df["ci_lower"] = df["excess_win_rate"] - 1.96 * df["se"]
        df["ci_upper"] = df["excess_win_rate"] + 1.96 * df["se"]

        fig = self._create_figure(df)
        chart = self._create_chart(df)

        return AnalysisOutput(figure=fig, data=df, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(10, 6))

        x = df["trade_size_bin"].values
        y = df["excess_win_rate"].values * 100
        ci_lower = df["ci_lower"].values * 100
        ci_upper = df["ci_upper"].values * 100

        # Plot confidence band
        ax.fill_between(x, ci_lower, ci_upper, alpha=0.2, color="#4C72B0", label="95% CI")

        # Plot line
        ax.plot(
            x,
            y,
            color="#4C72B0",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Excess Win Rate",
        )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_xscale("log")
        ax.set_xlabel("Trade Size (USD)")
        ax.set_ylabel("Excess Win Rate (pp)")
        ax.set_title("Win Rate by Trade Size (price-adjusted)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3, which="both")

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "trade_size": round(row["trade_size_bin"], 2),
                "Excess Win Rate": round(row["excess_win_rate"] * 100, 2),
                "95% CI Lower": round(row["ci_lower"] * 100, 2),
                "95% CI Upper": round(row["ci_upper"] * 100, 2),
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="trade_size",
            yKeys=["Excess Win Rate", "95% CI Lower", "95% CI Upper"],
            title="Excess Win Rate by Trade Size",
            xScale=ScaleType.LOG,
            yUnit=UnitType.PERCENT,
            colors={
                "Excess Win Rate": "#4C72B0",
                "95% CI Lower": "#4C72B0",
                "95% CI Upper": "#4C72B0",
            },
            strokeDasharrays=[None, "5 5", "5 5"],
        )
