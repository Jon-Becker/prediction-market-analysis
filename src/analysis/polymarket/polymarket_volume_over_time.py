"""Analyze trading volume over time across all Polymarket markets."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.polymarket.util.blocks import BLOCKS_DIR, FALLBACK_ANCHORS
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, ScaleType, UnitType


class PolymarketVolumeOverTimeAnalysis(Analysis):
    """Analyze trading volume over time across all Polymarket markets."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
    ):
        super().__init__(
            name="polymarket_volume_over_time",
            description="Polymarket quarterly trading volume analysis",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "polymarket" / "trades")

    def _register_block_to_timestamp_macro(self, con: duckdb.DuckDBPyConnection) -> None:
        """Register a DuckDB macro for block-to-timestamp interpolation."""
        # Try to load blocks from parquet files
        parquet_files = list(BLOCKS_DIR.glob("*.parquet")) if BLOCKS_DIR.exists() else []

        if parquet_files:
            con.execute(
                f"""
                CREATE TABLE blocks AS
                SELECT block_number, timestamp
                FROM '{BLOCKS_DIR}/*.parquet'
                ORDER BY block_number
                """
            )
            con.execute(
                """
                CREATE MACRO block_to_timestamp(block_num) AS (
                    SELECT CAST(
                        b1.timestamp + (block_num - b1.block_number) *
                        (b2.timestamp - b1.timestamp)::DOUBLE /
                        (b2.block_number - b1.block_number)::DOUBLE
                    AS BIGINT)
                    FROM blocks b1, blocks b2
                    WHERE b1.block_number <= block_num
                      AND b2.block_number >= block_num
                      AND b1.block_number = (SELECT MAX(block_number) FROM blocks WHERE block_number <= block_num)
                      AND b2.block_number = (SELECT MIN(block_number) FROM blocks WHERE block_number >= block_num)
                )
                """
            )
        else:
            # Fallback to hardcoded anchors with simple linear interpolation
            anchor_blocks = [b for b, _ in FALLBACK_ANCHORS]
            anchor_timestamps = [t for _, t in FALLBACK_ANCHORS]
            con.execute(
                f"""
                CREATE MACRO block_to_timestamp(block_num) AS (
                    CAST(
                        {anchor_timestamps[0]} + (block_num - {anchor_blocks[0]}) *
                        ({anchor_timestamps[-1]} - {anchor_timestamps[0]})::DOUBLE /
                        ({anchor_blocks[-1]} - {anchor_blocks[0]})::DOUBLE
                    AS BIGINT)
                )
                """
            )

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        self._register_block_to_timestamp_macro(con)

        # Volume is the USDC side of each trade:
        # - When maker_asset_id = '0', maker provides USDC (maker_amount)
        # - When taker_asset_id = '0', taker provides USDC (taker_amount)
        # Amounts are in 6-decimal USDC (1e6 = $1)
        df = con.execute(
            f"""
            SELECT
                DATE_TRUNC('quarter', to_timestamp(block_to_timestamp(block_number))) AS quarter,
                SUM(
                    CASE
                        WHEN maker_asset_id = '0' THEN maker_amount
                        WHEN taker_asset_id = '0' THEN taker_amount
                        ELSE 0
                    END
                ) / 1e6 AS volume_usd
            FROM '{self.trades_dir}/*.parquet'
            WHERE block_number IS NOT NULL
            GROUP BY quarter
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
            bars, labels=labels, fontsize=7, rotation=90, label_type="center", color="white", fontweight="bold"
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
            title="Polymarket Quarterly Volume",
            xLabel="Quarter",
            yLabel="Volume (USD)",
            yUnit=UnitType.DOLLARS,
            yScale=ScaleType.LOG,
        )
