"""Analyze maker vs taker returns by market category.

Tests whether the maker/taker gap varies by market category. Hypothesis:
the gap should be larger in categories with more retail participation (sports)
and smaller in categories with more sophisticated participants (finance).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.kalshi.util.categories import CATEGORY_SQL, get_group
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class MakerTakerReturnsByCategoryAnalysis(Analysis):
    """Analyze maker vs taker returns by market category on Kalshi."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="maker_taker_returns_by_category",
            description="Maker vs taker excess returns by market category",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Get taker and maker returns by category
        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, event_ticker, result
                FROM '{self.markets_dir}/*.parquet'
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
            ),
            taker_positions AS (
                SELECT
                    {CATEGORY_SQL.replace("event_ticker", "m.event_ticker")} AS category,
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                    CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                    t.count AS contracts,
                    t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            ),
            maker_positions AS (
                SELECT
                    {CATEGORY_SQL.replace("event_ticker", "m.event_ticker")} AS category,
                    CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                    CASE WHEN t.taker_side != m.result THEN 1.0 ELSE 0.0 END AS won,
                    t.count AS contracts,
                    t.count * (CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END) / 100.0 AS volume_usd
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            ),
            taker_stats AS (
                SELECT
                    category,
                    AVG(won) AS win_rate,
                    AVG(price / 100.0) AS avg_price,
                    AVG(won - price / 100.0) AS excess_return,
                    VAR_POP(won - price / 100.0) AS var_excess,
                    COUNT(*) AS n_trades,
                    SUM(contracts) AS contracts,
                    SUM(volume_usd) AS volume_usd,
                    SUM(contracts * (won - price / 100.0)) AS pnl
                FROM taker_positions
                GROUP BY category
            ),
            maker_stats AS (
                SELECT
                    category,
                    AVG(won) AS win_rate,
                    AVG(price / 100.0) AS avg_price,
                    AVG(won - price / 100.0) AS excess_return,
                    VAR_POP(won - price / 100.0) AS var_excess,
                    COUNT(*) AS n_trades,
                    SUM(contracts) AS contracts,
                    SUM(volume_usd) AS volume_usd,
                    SUM(contracts * (won - price / 100.0)) AS pnl
                FROM maker_positions
                GROUP BY category
            )
            SELECT
                t.category,
                t.win_rate AS taker_win_rate,
                t.avg_price AS taker_avg_price,
                t.excess_return AS taker_excess,
                t.var_excess AS taker_var,
                t.n_trades AS taker_n,
                t.contracts AS taker_contracts,
                t.volume_usd AS taker_volume,
                t.pnl AS taker_pnl,
                m.win_rate AS maker_win_rate,
                m.avg_price AS maker_avg_price,
                m.excess_return AS maker_excess,
                m.var_excess AS maker_var,
                m.n_trades AS maker_n,
                m.contracts AS maker_contracts,
                m.volume_usd AS maker_volume,
                m.pnl AS maker_pnl
            FROM taker_stats t
            JOIN maker_stats m ON t.category = m.category
            ORDER BY t.volume_usd DESC
            """
        ).df()

        # Apply group mapping
        df["group"] = df["category"].apply(get_group)

        # Aggregate by group
        group_stats = []
        for group in df["group"].unique():
            group_data = df[df["group"] == group]

            # Volume-weighted excess returns
            taker_vol_weighted = (group_data["taker_excess"] * group_data["taker_contracts"]).sum() / group_data[
                "taker_contracts"
            ].sum()
            maker_vol_weighted = (group_data["maker_excess"] * group_data["maker_contracts"]).sum() / group_data[
                "maker_contracts"
            ].sum()

            group_stats.append(
                {
                    "group": group,
                    "taker_excess": taker_vol_weighted * 100,
                    "maker_excess": maker_vol_weighted * 100,
                    "gap": (maker_vol_weighted - taker_vol_weighted) * 100,
                    "taker_n": int(group_data["taker_n"].sum()),
                    "maker_n": int(group_data["maker_n"].sum()),
                    "taker_volume": group_data["taker_volume"].sum(),
                    "maker_volume": group_data["maker_volume"].sum(),
                    "taker_pnl": group_data["taker_pnl"].sum(),
                    "maker_pnl": group_data["maker_pnl"].sum(),
                }
            )

        group_df = pd.DataFrame(group_stats)
        group_df = group_df.sort_values("taker_volume", ascending=False)

        fig = self._create_figure(group_df)
        chart = self._create_chart(group_df)

        return AnalysisOutput(figure=fig, data=group_df, chart=chart)

    def _create_figure(self, group_df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(12, 7))

        top_groups = group_df.head(8)
        x = np.arange(len(top_groups))
        width = 0.35

        ax.bar(
            x - width / 2,
            top_groups["taker_excess"],
            width,
            label="Taker",
            color="#e74c3c",
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            top_groups["maker_excess"],
            width,
            label="Maker",
            color="#2ecc71",
            alpha=0.8,
        )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Category")
        ax.set_ylabel("Volume-Weighted Excess Return (pp)")
        ax.set_title("Maker vs Taker Returns by Category")
        ax.set_xticks(x)
        ax.set_xticklabels(top_groups["group"], rotation=45, ha="right")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    def _create_chart(self, group_df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        top_groups = group_df.head(8)

        chart_data = [
            {
                "category": row["group"],
                "Taker Return": round(row["taker_excess"], 2),
                "Maker Return": round(row["maker_excess"], 2),
            }
            for _, row in top_groups.iterrows()
        ]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="category",
            yKeys=["Taker Return", "Maker Return"],
            title="Maker vs Taker Returns by Category",
            yUnit=UnitType.PERCENT,
            xLabel="Category",
            yLabel="Volume-Weighted Excess Return (pp)",
            colors={"Taker Return": "#e74c3c", "Maker Return": "#2ecc71"},
        )
