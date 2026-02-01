"""Analyze longshot volume share evolution over time.

Computes quarterly volume share by price bucket to assess whether
taker preferences for longshots changed over time or remained constant.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class LongshotVolumeShareOverTimeAnalysis(Analysis):
    """Analyze longshot volume share over time on Kalshi."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="longshot_volume_share_over_time",
            description="Taker volume share in longshot contracts (1-20c) by quarter",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Compute quarterly volume by price bucket for takers
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
                    DATE_TRUNC('quarter', t.created_time) AS quarter,
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                    t.count AS contracts,
                    t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            ),
            bucketed AS (
                SELECT
                    quarter,
                    CASE
                        WHEN price BETWEEN 1 AND 10 THEN '1-10c'
                        WHEN price BETWEEN 11 AND 20 THEN '11-20c'
                        WHEN price BETWEEN 21 AND 30 THEN '21-30c'
                        WHEN price BETWEEN 31 AND 40 THEN '31-40c'
                        WHEN price BETWEEN 41 AND 50 THEN '41-50c'
                        WHEN price BETWEEN 51 AND 60 THEN '51-60c'
                        WHEN price BETWEEN 61 AND 70 THEN '61-70c'
                        WHEN price BETWEEN 71 AND 80 THEN '71-80c'
                        WHEN price BETWEEN 81 AND 90 THEN '81-90c'
                        WHEN price BETWEEN 91 AND 99 THEN '91-99c'
                    END AS price_bucket,
                    CASE
                        WHEN price BETWEEN 1 AND 20 THEN 1
                        ELSE 0
                    END AS is_longshot,
                    volume_usd,
                    contracts
                FROM taker_trades
            )
            SELECT
                quarter,
                price_bucket,
                is_longshot,
                SUM(volume_usd) AS volume_usd,
                SUM(contracts) AS contracts,
                COUNT(*) AS n_trades
            FROM bucketed
            GROUP BY quarter, price_bucket, is_longshot
            ORDER BY quarter, price_bucket
            """
        ).df()

        # Convert quarter to pandas datetime
        df["quarter"] = pd.to_datetime(df["quarter"])

        # Compute total volume per quarter
        quarterly_totals = df.groupby("quarter")["volume_usd"].sum().reset_index()
        quarterly_totals.columns = ["quarter", "total_volume"]

        # Merge to get shares
        df = df.merge(quarterly_totals, on="quarter")
        df["volume_share"] = df["volume_usd"] / df["total_volume"] * 100

        # Compute longshot share (1-20c)
        longshot_df = (
            df[df["is_longshot"] == 1]
            .groupby("quarter")
            .agg({"volume_usd": "sum", "contracts": "sum", "n_trades": "sum"})
            .reset_index()
        )
        longshot_df = longshot_df.merge(quarterly_totals, on="quarter")
        longshot_df["longshot_share"] = (
            longshot_df["volume_usd"] / longshot_df["total_volume"] * 100
        )

        fig = self._create_figure(longshot_df)
        chart = self._create_chart(longshot_df, df, quarterly_totals)

        return AnalysisOutput(figure=fig, data=longshot_df, chart=chart)

    def _create_figure(self, longshot_df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(12, 6))

        quarters = longshot_df["quarter"].values
        x = np.arange(len(quarters))
        quarter_labels = [
            f"{pd.Timestamp(q).year} Q{(pd.Timestamp(q).month-1)//3+1}"
            for q in quarters
        ]

        ax.plot(
            x,
            longshot_df["longshot_share"],
            color="#9b59b6",
            linewidth=2,
            marker="o",
            markersize=6,
        )
        ax.fill_between(x, longshot_df["longshot_share"], alpha=0.3, color="#9b59b6")

        # Mark election
        election_idx = None
        for i, q in enumerate(quarters):
            ts = pd.Timestamp(q)
            if ts.year == 2024 and ts.month == 10:
                election_idx = i
                break

        if election_idx is not None:
            ax.axvline(
                x=election_idx, color="blue", linestyle=":", linewidth=1.5, alpha=0.7
            )
            ax.annotate(
                "2024 Election",
                xy=(election_idx, ax.get_ylim()[1] * 0.9),
                fontsize=9,
                ha="center",
                color="blue",
            )

        ax.set_xlabel("Quarter")
        ax.set_ylabel("Longshot Volume Share (%)")
        ax.set_title("Taker Volume Share in Longshot Contracts (1-20c)")
        ax.set_xticks(x)
        ax.set_xticklabels(quarter_labels, rotation=45, ha="right")
        ax.set_ylim(0, max(longshot_df["longshot_share"]) * 1.1)

        plt.tight_layout()
        return fig

    def _create_chart(
        self,
        longshot_df: pd.DataFrame,
        df: pd.DataFrame,
        quarterly_totals: pd.DataFrame,
    ) -> ChartConfig:
        """Create the chart configuration for web display."""
        # Pivot for stacked bar chart
        pivot_df = df.pivot_table(
            index="quarter", columns="price_bucket", values="volume_share", aggfunc="sum"
        ).fillna(0)

        # Reorder columns logically
        bucket_order = [
            "1-10c",
            "11-20c",
            "21-30c",
            "31-40c",
            "41-50c",
            "51-60c",
            "61-70c",
            "71-80c",
            "81-90c",
            "91-99c",
        ]
        pivot_df = pivot_df[[c for c in bucket_order if c in pivot_df.columns]]

        # Filter out quarters with < $1M volume
        valid_quarters = set(
            pd.to_datetime(
                quarterly_totals[quarterly_totals["total_volume"] >= 1e6]["quarter"]
            )
        )
        pivot_filtered = pivot_df[pivot_df.index.isin(valid_quarters)]

        # Use snake_case keys for consistency
        bucket_keys = [
            "1_10",
            "11_20",
            "21_30",
            "31_40",
            "41_50",
            "51_60",
            "61_70",
            "71_80",
            "81_90",
            "91_99",
        ]
        bucket_key_map = dict(zip(bucket_order, bucket_keys))

        chart_data = [
            {
                "quarter": f"{pd.Timestamp(q).year} Q{(pd.Timestamp(q).month-1)//3+1}",
                **{
                    bucket_key_map[bucket]: round(pivot_filtered.loc[q, bucket], 2)
                    if bucket in pivot_filtered.columns
                    else 0
                    for bucket in bucket_order
                },
            }
            for q in pivot_filtered.index
        ]

        return ChartConfig(
            type=ChartType.STACKED_BAR_100,
            data=chart_data,
            xKey="quarter",
            yKeys=bucket_keys,
            title="Taker Volume Distribution by Price",
            yUnit=UnitType.PERCENT,
            xLabel="Quarter",
            yLabel="Volume Share (%)",
        )
