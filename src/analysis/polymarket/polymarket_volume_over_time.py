"""Analyze Polymarket notional trading volume over time."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, ScaleType, UnitType


class PolymarketVolumeOverTimeAnalysis(Analysis):
    """Analyze quarterly notional trading volume on Polymarket."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        blocks_dir: Path | str | None = None,
    ):
        super().__init__(
            name="polymarket_volume_over_time",
            description="Quarterly notional volume analysis for Polymarket",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "polymarket" / "trades")
        self.blocks_dir = Path(blocks_dir or base_dir / "data" / "polymarket" / "blocks")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Load blocks lookup with computed bucket index for efficient joining
        # Blocks are at consistent 10800 intervals starting from 39992400
        con.execute(
            f"""
            CREATE TABLE blocks AS
            SELECT
                (block_number - 39992400) // 10800 AS bucket,
                timestamp
            FROM '{self.blocks_dir}/*.parquet'
            """
        )

        # Calculate quarterly notional volume using bucket join (much faster than ASOF)
        # Notional = outcome tokens traded (worth $1 at resolution if winning)
        # When maker_asset_id='0': maker pays USDC, receives taker_amount tokens
        # When taker_asset_id='0': taker pays USDC, receives maker_amount tokens
        df = con.execute(
            f"""
            SELECT
                DATE_TRUNC('quarter', TO_TIMESTAMP(b.timestamp)) AS quarter,
                SUM(
                    CASE
                        WHEN t.maker_asset_id = '0' THEN t.taker_amount
                        ELSE t.maker_amount
                    END
                ) / 1e6 AS volume_usd
            FROM '{self.trades_dir}/*.parquet' t
            JOIN blocks b ON (t.block_number - 39992400) // 10800 = b.bucket
            WHERE t.maker_asset_id = '0' OR t.taker_asset_id = '0'
            GROUP BY DATE_TRUNC('quarter', TO_TIMESTAMP(b.timestamp))
            ORDER BY quarter
            """
        ).df()

        fig = self._create_figure(df)
        chart = self._create_chart(df)

        return AnalysisOutput(figure=fig, data=df, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(df["quarter"], df["volume_usd"] / 1e6, width=80, color="#4C72B0")
        bars[-1].set_hatch("//")
        bars[-1].set_edgecolor((1, 1, 1, 0.3))
        labels = [f"${v / 1e3:.2f}B" if v > 999 else f"${v:.2f}M" for v in df["volume_usd"] / 1e6]
        ax.bar_label(
            bars,
            labels=labels,
            fontsize=7,
            rotation=90,
            label_type="center",
            color="white",
            fontweight="bold",
        )
        ax.set_xlabel("Date")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)
        ax.set_ylabel("Quarterly Volume (millions USD)")
        ax.set_title("Polymarket Quarterly Notional Volume")

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "quarter": f"Q{(pd.Timestamp(row['quarter']).month - 1) // 3 + 1} '{str(pd.Timestamp(row['quarter']).year)[2:]}",
                "volume": int(row["volume_usd"]),
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="quarter",
            yKeys=["volume"],
            title="Polymarket Quarterly Notional Volume",
            xLabel="Quarter",
            yLabel="Volume (USD)",
            yUnit=UnitType.DOLLARS,
            yScale=ScaleType.LOG,
        )
