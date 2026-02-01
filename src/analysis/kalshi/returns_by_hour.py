"""Analyze excess returns by hour of day (ET).

Examines whether trading performance varies by time of day, potentially
revealing when informed vs. uninformed participants are most active.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class ReturnsByHourAnalysis(Analysis):
    """Analyze excess returns by hour of day on Kalshi."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="returns_by_hour",
            description="Excess returns by hour of day (ET)",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Compute excess returns by hour of day (ET)
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
                    CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                    t.count AS contracts,
                    t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            )
            SELECT
                hour_et,
                AVG(won) AS win_rate,
                AVG(price / 100.0) AS avg_implied_prob,
                AVG(won - price / 100.0) AS excess_return,
                VAR_SAMP(won - price / 100.0) AS var_excess,
                SUM(contracts) AS total_contracts,
                SUM(volume_usd) AS total_volume_usd,
                COUNT(*) AS n_trades
            FROM trade_data
            GROUP BY hour_et
            ORDER BY hour_et
            """
        ).df()

        # Calculate standard error and 95% CI
        df["se"] = np.sqrt(df["var_excess"] / df["n_trades"])
        df["ci_lower"] = df["excess_return"] - 1.96 * df["se"]
        df["ci_upper"] = df["excess_return"] + 1.96 * df["se"]

        fig = self._create_figure(df)
        chart = self._create_chart(df)

        return AnalysisOutput(figure=fig, data=df, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax1 = plt.subplots(figsize=(12, 6))

        hours = df["hour_et"].values
        excess = df["excess_return"].values * 100  # Convert to percentage points

        ax1.bar(hours, excess, color="#4C72B0", alpha=0.7, width=0.8)
        ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

        ax1.set_xlabel("Hour of Day (ET)")
        ax1.set_ylabel("Excess Return (pp)")
        ax1.set_title("Excess Return by Hour of Day")
        ax1.set_xlim(-0.5, 23.5)
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
        ax1.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "hour": int(row["hour_et"]),
                "Excess Return": round(row["excess_return"] * 100, 2),
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="hour",
            yKeys=["Excess Return"],
            title="Excess Return by Hour of Day (ET)",
            yUnit=UnitType.PERCENT,
        )
