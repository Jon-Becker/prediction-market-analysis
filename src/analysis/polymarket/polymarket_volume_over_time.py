"""Analyze Polymarket notional trading volume over time."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, ScaleType, UnitType

# Bucket size for block-to-timestamp approximation (10800 blocks ~ 6 hours at 2 sec/block)
BLOCK_BUCKET_SIZE = 10800


class PolymarketVolumeOverTimeAnalysis(Analysis):
    """Analyze monthly notional trading volume on Polymarket."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        legacy_trades_dir: Path | str | None = None,
        blocks_dir: Path | str | None = None,
        collateral_lookup_path: Path | str | None = None,
    ):
        super().__init__(
            name="polymarket_volume_over_time",
            description="Monthly notional volume analysis for Polymarket",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "polymarket" / "trades")
        self.blocks_dir = Path(blocks_dir or base_dir / "data" / "polymarket" / "blocks")
        

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Create blocks lookup table with bucket index for efficient joining
        con.execute(
            f"""
            CREATE TABLE blocks AS
            SELECT
                block_number // {BLOCK_BUCKET_SIZE} AS bucket,
                MIN(timestamp) AS timestamp
            FROM '{self.blocks_dir}/*.parquet'
            GROUP BY block_number // {BLOCK_BUCKET_SIZE}
            """
        )

        # CTF Exchange trades: volume = USDC cash flow (the side where asset_id = '0')
        # When maker_asset_id='0': maker pays maker_amount USDC, receives taker_amount tokens
        # When taker_asset_id='0': taker pays taker_amount USDC, receives maker_amount tokens
        # Double counting polymarket volume is a common issue as described by paradigm  
        # here https://www.paradigm.xyz/2025/12/polymarket-volume-is-being-double-counted
        # per the article, the best way to avoid double counting is to focus on either
        # maker or taker side volume (but not sum both)
        # """Taker-side volume can be obtained by filtering and summing OrderFilled 
        # events where the taker field is equal to one of the two exchange contracts""".
        # Another point of attention: Polymarket recently migrated to a v2 version of their exchanges
        # we have to consider both the v1 contracts and the v2 contracts

        CTF_exchange_v2 = "0xE111180000d2663C0091e4f400237545B87B996B"
        Neg_Risk_CTF_Exchange_v2 = "0xe2222d279d744050d28e00520010520000310F59"

        CTF_Exchange_v1 = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
        Neg_Risk_Exchange_v1 = "0xC5d563A36AE78145C45a50134d48A1215220f80a"


        # Note: the current indexer (src/indexers/polymarket/blockchain.py) only
        # backfills v1 contracts. v2 entries below are forward-compatible no-ops
        # until the indexer is extended.
        exchange_addresses = [
            CTF_Exchange_v1,
            Neg_Risk_Exchange_v1,
            CTF_exchange_v2,
            Neg_Risk_CTF_Exchange_v2,
        ]
        exchanges_in = ", ".join(f"lower('{a}')" for a in exchange_addresses)

        ctf_volume_query = f"""
            SELECT
                DATE_TRUNC('month', b.timestamp::TIMESTAMP) AS month,
                SUM(
                    CASE
                        WHEN t.maker_asset_id = '0' THEN t.maker_amount
                        ELSE t.taker_amount
                    END
                ) / 1e6 AS volume_usd
            FROM '{self.trades_dir}/*.parquet' t
            JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
            WHERE (t.maker_asset_id = '0' OR t.taker_asset_id = '0')
              AND lower(t.taker) IN ({exchanges_in})
            GROUP BY DATE_TRUNC('month', b.timestamp::TIMESTAMP)
            ORDER BY month
        """
        df = con.execute(ctf_volume_query).df()

        fig = self._create_figure(df)
        chart = self._create_chart(df)

        return AnalysisOutput(figure=fig, data=df, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure: one bar per month, one tick per month."""
        n = len(df)
        fig, ax = plt.subplots(figsize=(max(14, n * 0.22), 7))

        x = list(range(n))
        volume_m = (df["volume_usd"] / 1e6).to_numpy()  # in $M

        # Highlight Oct/Nov 2024 (the months the Paradigm article references)
        highlight_months = {pd.Timestamp("2024-10-01"), pd.Timestamp("2024-11-01")}
        colors = [
            "#DD8452" if pd.Timestamp(m) in highlight_months else "#4C72B0"
            for m in df["month"]
        ]
        bars = ax.bar(x, volume_m, width=0.85, color=colors, edgecolor="white", linewidth=0.3)

        # Mark the most recent (potentially in-progress) month
        bars[-1].set_hatch("//")
        bars[-1].set_edgecolor((1, 1, 1, 0.6))

        def fmt(v_m: float) -> str:
            if v_m >= 1000:
                return f"${v_m / 1000:.2f}B"
            if v_m >= 1:
                return f"${v_m:.1f}M"
            if v_m >= 0.001:
                return f"${v_m * 1000:.0f}k"
            return f"${v_m * 1e6:.0f}"

        labels = [fmt(v) for v in volume_m]
        ax.bar_label(bars, labels=labels, fontsize=7, rotation=90, padding=2, color="black")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [pd.Timestamp(m).strftime("%b %Y") for m in df["month"]],
            rotation=90,
            fontsize=8,
        )

        ax.set_ylim(bottom=0, top=volume_m.max() * 1.25)  # headroom so labels don't clip
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Volume (USD billions)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v / 1000:.1f}B"))
        ax.set_title("Polymarket Monthly USDC Volume")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.margins(x=0.005)

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "month": f"{pd.Timestamp(row['month']).strftime('%b %Y')}",

                "volume": int(row["volume_usd"]),
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="month",
            yKeys=["volume"],
            title="Polymarket Monthly USDC Volume",
            xLabel="Month",
            yLabel="Volume (USD)",
            yUnit=UnitType.DOLLARS,
            yScale=ScaleType.LOG,
        )
