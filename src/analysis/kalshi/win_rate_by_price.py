"""Analyze win rate by price to assess market calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class WinRateByPriceAnalysis(Analysis):
    """Analyze win rate by price to assess market calibration on Kalshi."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="win_rate_by_price",
            description="Win rate vs price market calibration analysis",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, result
                FROM '{self.markets_dir}/*.parquet'
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
            ),
            all_positions AS (
                -- Taker side
                SELECT
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                    CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker

                UNION ALL

                -- Maker side (counterparty)
                SELECT
                    CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                    CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            )
            SELECT
                price,
                COUNT(*) AS total_trades,
                SUM(won) AS wins,
                100.0 * SUM(won) / COUNT(*) AS win_rate
            FROM all_positions
            GROUP BY price
            ORDER BY price
            """
        ).df()

        fig = self._create_figure(df)
        chart = self._create_chart(df)

        return AnalysisOutput(figure=fig, data=df, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(
            df["price"],
            df["win_rate"],
            s=30,
            alpha=0.8,
            color="#4C72B0",
            edgecolors="none",
        )
        ax.plot(
            [0, 100],
            [0, 100],
            linestyle="--",
            color="#D65F5F",
            linewidth=1.5,
            label="Perfect calibration",
        )
        ax.set_xlabel("Contract Price (cents)")
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Win Rate vs Price: Market Calibration")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 101, 10))
        ax.set_xticks(range(0, 101, 1), minor=True)
        ax.set_yticks(range(0, 101, 10))
        ax.set_yticks(range(0, 101, 1), minor=True)
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "price": int(row["price"]),
                "actual": round(row["win_rate"], 2),
                "implied": int(row["price"]),
            }
            for _, row in df.iterrows()
            if 1 <= row["price"] <= 99
        ]

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="price",
            yKeys=["actual", "implied"],
            title="Actual Win Rate vs Contract Price",
            strokeDasharrays=[None, "5 5"],
            yUnit=UnitType.PERCENT,
            xLabel="Contract Price (cents)",
            yLabel="Actual Win Rate (%)",
        )
