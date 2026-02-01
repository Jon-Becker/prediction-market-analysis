"""Analyze volume-weighted average price by hour of day (ET).

Examines whether trading patterns vary by time of day, potentially
revealing when retail vs. institutional participants are most active.
Lower VWAP suggests more longshot buying; higher VWAP suggests more
favorite buying.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class VwapByHourAnalysis(Analysis):
    """Analyze volume-weighted average price by hour of day on Kalshi."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="vwap_by_hour",
            description="Volume-weighted average price by hour of day (ET)",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Compute VWAP by hour of day (ET)
        # Trade timestamps are already in America/New_York timezone
        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, result
                FROM '{self.markets_dir}/*.parquet'
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
            ),
            trade_data AS (
                SELECT
                    EXTRACT(HOUR FROM t.created_time) AS hour_et,
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                    t.count AS contracts,
                    t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            )
            SELECT
                hour_et,
                SUM(price * contracts) / SUM(contracts) AS vwap,
                SUM(contracts) AS total_contracts,
                SUM(volume_usd) AS total_volume_usd,
                COUNT(*) AS n_trades,
                AVG(price) AS avg_price,
                STDDEV_SAMP(price) AS std_price
            FROM trade_data
            GROUP BY hour_et
            ORDER BY hour_et
            """
        ).df()

        # Calculate standard error for VWAP (approximation using std of prices)
        df["se"] = df["std_price"] / np.sqrt(df["n_trades"])
        df["ci_lower"] = df["vwap"] - 1.96 * df["se"]
        df["ci_upper"] = df["vwap"] + 1.96 * df["se"]

        fig = self._create_figure(df)
        chart = self._create_chart(df)

        return AnalysisOutput(figure=fig, data=df, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot VWAP as line with CI band
        hours = df["hour_et"].values
        vwap = df["vwap"].values
        ci_lower = df["ci_lower"].values
        ci_upper = df["ci_upper"].values

        ax1.fill_between(hours, ci_lower, ci_upper, alpha=0.2, color="#4C72B0")
        ax1.plot(
            hours,
            vwap,
            color="#4C72B0",
            linewidth=2,
            marker="o",
            markersize=6,
            label="VWAP",
        )
        ax1.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Fair odds (50c)")

        ax1.set_xlabel("Hour of Day (ET)")
        ax1.set_ylabel("Volume-Weighted Avg Price (cents)")
        ax1.set_title("Volume-Weighted Average Price by Hour of Day")
        ax1.set_xlim(-0.5, 23.5)
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Add volume bars on secondary axis
        ax2 = ax1.twinx()
        ax2.bar(
            hours,
            df["total_volume_usd"] / 1e9,
            alpha=0.3,
            color="#2ecc71",
            width=0.8,
            label="Volume",
        )
        ax2.set_ylabel("Volume ($ Billions)", color="#2ecc71")
        ax2.tick_params(axis="y", labelcolor="#2ecc71")

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "hour": int(row["hour_et"]),
                "VWAP": round(row["vwap"], 2),
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="hour",
            yKeys=["VWAP"],
            title="Volume-Weighted Average Price by Hour of Day (ET)",
            yUnit=UnitType.CENTS,
        )
