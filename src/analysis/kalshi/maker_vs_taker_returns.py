"""Analyze maker vs taker returns.

Compares performance of passive liquidity providers (makers) against
aggressive order takers across price points. Tests whether informed
trading exists or if market making captures spread profitably.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class MakerVsTakerReturnsAnalysis(Analysis):
    """Analyze maker vs taker returns by price."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="maker_vs_taker_returns",
            description="Maker vs taker excess returns by price analysis",
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
            taker_positions AS (
                SELECT
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                    CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                    t.count AS contracts,
                    t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            ),
            maker_positions AS (
                SELECT
                    CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                    CASE WHEN t.taker_side != m.result THEN 1.0 ELSE 0.0 END AS won,
                    t.count AS contracts,
                    t.count * (CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END) / 100.0 AS volume_usd
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            ),
            taker_stats AS (
                SELECT
                    price,
                    AVG(won) AS win_rate,
                    price / 100.0 AS expected_win_rate,
                    AVG(won) - price / 100.0 AS excess_return,
                    VAR_POP(won - price / 100.0) AS var_excess,
                    COUNT(*) AS n_trades,
                    SUM(volume_usd) AS volume_usd,
                    SUM(contracts) AS contracts,
                    SUM(contracts * (won - price / 100.0)) AS pnl
                FROM taker_positions
                GROUP BY price
            ),
            maker_stats AS (
                SELECT
                    price,
                    AVG(won) AS win_rate,
                    price / 100.0 AS expected_win_rate,
                    AVG(won) - price / 100.0 AS excess_return,
                    VAR_POP(won - price / 100.0) AS var_excess,
                    COUNT(*) AS n_trades,
                    SUM(volume_usd) AS volume_usd,
                    SUM(contracts) AS contracts,
                    SUM(contracts * (won - price / 100.0)) AS pnl
                FROM maker_positions
                GROUP BY price
            )
            SELECT
                t.price,
                t.win_rate AS taker_win_rate,
                t.expected_win_rate AS taker_expected,
                t.excess_return AS taker_excess,
                t.var_excess AS taker_var,
                t.n_trades AS taker_n,
                t.volume_usd AS taker_volume,
                t.pnl AS taker_pnl,
                m.win_rate AS maker_win_rate,
                m.expected_win_rate AS maker_expected,
                m.excess_return AS maker_excess,
                m.var_excess AS maker_var,
                m.n_trades AS maker_n,
                m.volume_usd AS maker_volume,
                m.pnl AS maker_pnl
            FROM taker_stats t
            JOIN maker_stats m ON t.price = m.price
            WHERE t.price BETWEEN 1 AND 99
            ORDER BY t.price
            """
        ).df()

        # Calculate standard errors and z-statistics
        df["taker_se"] = np.sqrt(df["taker_var"] / df["taker_n"])
        df["maker_se"] = np.sqrt(df["maker_var"] / df["maker_n"])
        df["taker_z"] = df["taker_excess"] / df["taker_se"]
        df["maker_z"] = df["maker_excess"] / df["maker_se"]
        df["taker_p"] = 2 * (1 - stats.norm.cdf(np.abs(df["taker_z"])))
        df["maker_p"] = 2 * (1 - stats.norm.cdf(np.abs(df["maker_z"])))

        fig = self._create_figure(df)
        chart = self._create_chart(df)

        return AnalysisOutput(figure=fig, data=df, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        df_sorted = df.sort_values("price")
        maker_counterparty = (
            df_sorted.set_index("price")["maker_excess"].reindex(100 - df_sorted["price"].values).values
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            df_sorted["price"],
            df_sorted["taker_excess"] * 100,
            color="#e74c3c",
            linewidth=1.5,
            label="Taker",
            alpha=0.8,
        )
        ax.plot(
            df_sorted["price"],
            maker_counterparty * 100,
            color="#2ecc71",
            linewidth=1.5,
            label="Maker (counterparty)",
            alpha=0.8,
        )
        ax.fill_between(df_sorted["price"], df_sorted["taker_excess"] * 100, alpha=0.2, color="#e74c3c")
        ax.fill_between(df_sorted["price"], maker_counterparty * 100, alpha=0.2, color="#2ecc71")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Contract Price (cents)")
        ax.set_ylabel("Excess Return (pp)")
        ax.set_title("Maker vs Taker Excess Returns by Price")
        ax.set_xlim(1, 99)
        ax.set_xticks(range(0, 101, 10))
        ax.legend(loc="upper right")
        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        maker_by_price = df.set_index("price")["maker_excess"].to_dict()
        chart_data = [
            {
                "price": int(row["price"]),
                "Taker": round(row["taker_excess"] * 100, 2),
                "Maker (counterparty)": round(maker_by_price.get(100 - row["price"], 0) * 100, 2),
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="price",
            yKeys=["Taker", "Maker (counterparty)"],
            title="Maker vs Taker Excess Returns by Price",
            yUnit=UnitType.PERCENT,
            xLabel="Contract Price (cents)",
            yLabel="Excess Return (pp)",
        )
