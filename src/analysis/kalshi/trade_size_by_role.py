"""Analyze trade size distribution by role (maker vs taker).

Tests whether makers place larger trades than takers on average,
which would be consistent with sophisticated, well-capitalized participants.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class TradeSizeByRoleAnalysis(Analysis):
    """Analyze trade size distribution by role (maker vs taker) on Kalshi."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="trade_size_by_role",
            description="Trade size comparison between makers and takers",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Get aggregate trade size stats for takers and makers
        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, result
                FROM '{self.markets_dir}/*.parquet'
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
            ),
            taker_trades AS (
                SELECT
                    t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS trade_size_usd,
                    t.count AS contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            ),
            maker_trades AS (
                SELECT
                    t.count * (CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END) / 100.0 AS trade_size_usd,
                    t.count AS contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            )
            SELECT
                'taker' AS role,
                AVG(trade_size_usd) AS mean_trade_size,
                MEDIAN(trade_size_usd) AS median_trade_size,
                STDDEV_POP(trade_size_usd) AS std_trade_size,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY trade_size_usd) AS p25_trade_size,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY trade_size_usd) AS p75_trade_size,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY trade_size_usd) AS p90_trade_size,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY trade_size_usd) AS p95_trade_size,
                AVG(contracts) AS mean_contracts,
                MEDIAN(contracts) AS median_contracts,
                COUNT(*) AS n_trades,
                SUM(trade_size_usd) AS total_volume
            FROM taker_trades

            UNION ALL

            SELECT
                'maker' AS role,
                AVG(trade_size_usd) AS mean_trade_size,
                MEDIAN(trade_size_usd) AS median_trade_size,
                STDDEV_POP(trade_size_usd) AS std_trade_size,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY trade_size_usd) AS p25_trade_size,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY trade_size_usd) AS p75_trade_size,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY trade_size_usd) AS p90_trade_size,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY trade_size_usd) AS p95_trade_size,
                AVG(contracts) AS mean_contracts,
                MEDIAN(contracts) AS median_contracts,
                COUNT(*) AS n_trades,
                SUM(trade_size_usd) AS total_volume
            FROM maker_trades
            """
        ).df()

        fig = self._create_figure(df)
        chart = self._create_chart(df)

        return AnalysisOutput(figure=fig, data=df, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(2)
        width = 0.35

        mean_sizes = df.set_index("role")["mean_trade_size"]
        median_sizes = df.set_index("role")["median_trade_size"]

        ax.bar(
            x - width / 2,
            [mean_sizes["taker"], mean_sizes["maker"]],
            width,
            label="Mean",
            color="#3498db",
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            [median_sizes["taker"], median_sizes["maker"]],
            width,
            label="Median",
            color="#e74c3c",
            alpha=0.8,
        )
        ax.set_ylabel("Trade Size (USD)")
        ax.set_title("Trade Size by Role: Mean vs Median")
        ax.set_xticks(x)
        ax.set_xticklabels(["Taker", "Maker"])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for i, (mean, median) in enumerate(
            zip(
                [mean_sizes["taker"], mean_sizes["maker"]],
                [median_sizes["taker"], median_sizes["maker"]],
            )
        ):
            ax.annotate(f"${mean:.0f}", (i - width / 2, mean), ha="center", va="bottom", fontsize=9)
            ax.annotate(f"${median:.0f}", (i + width / 2, median), ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "role": row["role"].title(),
                "Mean": round(row["mean_trade_size"], 2),
                "Median": round(row["median_trade_size"], 2),
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="role",
            yKeys=["Mean", "Median"],
            title="Trade Size Statistics by Role",
            yUnit=UnitType.DOLLARS,
            xLabel="Role",
            yLabel="Trade Size (USD)",
        )
