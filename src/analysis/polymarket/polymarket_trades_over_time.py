"""Analyze Polymarket trade counts over time at block-level granularity."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class PolymarketTradesOverTimeAnalysis(Analysis):
    """Analyze trade counts per block on Polymarket (extremely granular)."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        legacy_trades_dir: Path | str | None = None,
        blocks_dir: Path | str | None = None,
    ):
        super().__init__(
            name="polymarket_trades_over_time",
            description="Trade counts per block on Polymarket",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "polymarket" / "trades")
        self.legacy_trades_dir = Path(legacy_trades_dir or base_dir / "data" / "polymarket" / "legacy_trades")
        self.blocks_dir = Path(blocks_dir or base_dir / "data" / "polymarket" / "blocks")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        with self.progress("Counting trades per block"):
            # Count trades per block (no interpolation - only blocks with trades)
            # Combines CTF Exchange trades and legacy FPMM trades
            trades_per_block = con.execute(
                f"""
                SELECT
                    block_number,
                    SUM(trade_count) AS trade_count
                FROM (
                    SELECT block_number, COUNT(*) AS trade_count
                    FROM '{self.trades_dir}/*.parquet'
                    GROUP BY block_number
                    UNION ALL
                    SELECT block_number, COUNT(*) AS trade_count
                    FROM '{self.legacy_trades_dir}/*.parquet'
                    GROUP BY block_number
                )
                GROUP BY block_number
                ORDER BY block_number
                """
            ).df()

        with self.progress("Joining with block timestamps"):
            # Join with blocks to get timestamps
            con.register("trades_per_block", trades_per_block)
            df = con.execute(
                f"""
                SELECT
                    t.block_number,
                    b.timestamp,
                    t.trade_count
                FROM trades_per_block t
                JOIN '{self.blocks_dir}/*.parquet' b ON t.block_number = b.block_number
                ORDER BY t.block_number
                """
            ).df()

        # Convert timestamp to datetime (timestamp is ISO string format)
        df["datetime"] = pd.to_datetime(df["timestamp"])

        fig = self._create_figure(df)
        chart = self._create_chart(df)

        return AnalysisOutput(figure=fig, data=df, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(
            df["datetime"],
            df["trade_count"],
            linewidth=0.1,
            color="#4C72B0",
            alpha=0.7,
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Trades per Block")
        ax.set_title("Polymarket Trades Over Time (Per Block)")

        # Format x-axis
        fig.autofmt_xdate()

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        # Use vectorized conversion for performance with large datasets
        chart_df = pd.DataFrame(
            {
                "block": df["block_number"].astype(int),
                "timestamp": df["timestamp"],
                "trades": df["trade_count"].astype(int),
            }
        )
        chart_data = chart_df.to_dict(orient="records")

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="timestamp",
            yKeys=["trades"],
            title="Polymarket Trades Over Time (Per Block)",
            xLabel="Timestamp",
            yLabel="Trades per Block",
            yUnit=UnitType.NUMBER,
        )
