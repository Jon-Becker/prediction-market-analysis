"""Analyze Polymarket notional trading volume over time."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, ScaleType, UnitType

# Bucket size for block-to-timestamp approximation (10800 blocks ~ 6 hours at 2 sec/block)
BLOCK_BUCKET_SIZE = 10800


class PolymarketVolumeOverTimeAnalysis(Analysis):
    """Analyze quarterly notional trading volume on Polymarket."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        legacy_trades_dir: Path | str | None = None,
        blocks_dir: Path | str | None = None,
        collateral_lookup_path: Path | str | None = None,
    ):
        super().__init__(
            name="polymarket_volume_over_time",
            description="Quarterly notional volume analysis for Polymarket",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "polymarket" / "trades")
        self.legacy_trades_dir = Path(legacy_trades_dir or base_dir / "data" / "polymarket" / "legacy_trades")
        self.blocks_dir = Path(blocks_dir or base_dir / "data" / "polymarket" / "blocks")
        self.collateral_lookup_path = Path(
            collateral_lookup_path or base_dir / "data" / "polymarket" / "fpmm_collateral_lookup.json"
        )

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Load USDC market addresses from collateral lookup (only include USDC markets)
        with open(self.collateral_lookup_path) as f:
            collateral_lookup = json.load(f)
        usdc_markets = [addr for addr, info in collateral_lookup.items() if info["collateral_symbol"] == "USDC"]

        # Create blocks lookup table with bucket index for efficient joining
        con.execute(
            f"""
            CREATE TABLE blocks AS
            SELECT
                block_number // {BLOCK_BUCKET_SIZE} AS bucket,
                FIRST(timestamp) AS timestamp
            FROM '{self.blocks_dir}/*.parquet'
            GROUP BY block_number // {BLOCK_BUCKET_SIZE}
            """
        )

        # Register USDC markets as a table for filtering
        con.execute("CREATE TABLE usdc_markets (fpmm_address VARCHAR)")
        con.executemany("INSERT INTO usdc_markets VALUES (?)", [(addr,) for addr in usdc_markets])

        # Legacy FPMM trades: amount is in USDC (6 decimals) for USDC-collateralized markets
        # Only include markets with USDC collateral
        legacy_volume_query = f"""
            SELECT
                DATE_TRUNC('quarter', b.timestamp::TIMESTAMP) AS quarter,
                SUM(t.amount::BIGINT) / 1e6 AS volume_usd
            FROM '{self.legacy_trades_dir}/*.parquet' t
            JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
            WHERE t.fpmm_address IN (SELECT fpmm_address FROM usdc_markets)
            GROUP BY DATE_TRUNC('quarter', b.timestamp::TIMESTAMP)
        """

        # CTF Exchange trades: notional = outcome tokens traded
        # When maker_asset_id='0': maker pays USDC, receives taker_amount tokens
        # When taker_asset_id='0': taker pays USDC, receives maker_amount tokens
        ctf_volume_query = f"""
            SELECT
                DATE_TRUNC('quarter', b.timestamp::TIMESTAMP) AS quarter,
                SUM(
                    CASE
                        WHEN t.maker_asset_id = '0' THEN t.taker_amount
                        ELSE t.maker_amount
                    END
                ) / 1e6 AS volume_usd
            FROM '{self.trades_dir}/*.parquet' t
            JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
            WHERE t.maker_asset_id = '0' OR t.taker_asset_id = '0'
            GROUP BY DATE_TRUNC('quarter', b.timestamp::TIMESTAMP)
        """

        # Combine both sources and aggregate by quarter
        df = con.execute(
            f"""
            SELECT quarter, SUM(volume_usd) AS volume_usd
            FROM (
                {legacy_volume_query}
                UNION ALL
                {ctf_volume_query}
            )
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
