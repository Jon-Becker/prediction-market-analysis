"""Analyze maker win rate by position direction (YES vs NO).

Focused analysis for Section 6.4 of the paper. Tests whether makers who
buy NO outperform makers who buy YES at equivalent prices, confirming
selective positioning rather than passive accommodation.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class MakerWinRateByDirectionAnalysis(Analysis):
    """Analyze maker win rate by position direction (YES vs NO)."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="maker_win_rate_by_direction",
            description="Maker win rate by position direction (YES vs NO)",
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
            maker_yes_positions AS (
                -- Maker bought YES (taker sold YES = taker bought NO)
                SELECT
                    t.yes_price AS price,
                    CASE WHEN m.result = 'yes' THEN 1.0 ELSE 0.0 END AS won,
                    t.count AS contracts,
                    'YES' AS maker_side
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.taker_side = 'no'
            ),
            maker_no_positions AS (
                -- Maker bought NO (taker sold NO = taker bought YES)
                SELECT
                    t.no_price AS price,
                    CASE WHEN m.result = 'no' THEN 1.0 ELSE 0.0 END AS won,
                    t.count AS contracts,
                    'NO' AS maker_side
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.taker_side = 'yes'
            ),
            all_maker_positions AS (
                SELECT * FROM maker_yes_positions
                UNION ALL
                SELECT * FROM maker_no_positions
            )
            SELECT
                maker_side,
                price,
                SUM(won * contracts) / SUM(contracts) AS win_rate,
                price / 100.0 AS implied_prob,
                SUM(won * contracts) / SUM(contracts) - price / 100.0 AS mispricing,
                COUNT(*) AS n_trades,
                SUM(contracts) AS contracts
            FROM all_maker_positions
            WHERE price BETWEEN 1 AND 99
            GROUP BY maker_side, price
            ORDER BY maker_side, price
            """
        ).df()

        # Pivot for comparison
        df_yes = df[df["maker_side"] == "YES"][["price", "win_rate", "mispricing", "n_trades", "contracts"]].copy()
        df_yes = df_yes.rename(
            columns={
                "win_rate": "yes_win_rate",
                "mispricing": "yes_mispricing",
                "n_trades": "yes_n",
                "contracts": "yes_contracts",
            }
        )

        df_no = df[df["maker_side"] == "NO"][["price", "win_rate", "mispricing", "n_trades", "contracts"]].copy()
        df_no = df_no.rename(
            columns={
                "win_rate": "no_win_rate",
                "mispricing": "no_mispricing",
                "n_trades": "no_n",
                "contracts": "no_contracts",
            }
        )

        comparison = pd.merge(df_yes, df_no, on="price", how="outer")
        comparison["implied_prob"] = comparison["price"] / 100.0
        comparison = comparison.sort_values("price")

        fig = self._create_figure(comparison)
        chart = self._create_chart(comparison)

        return AnalysisOutput(figure=fig, data=comparison, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(
            df["price"],
            df["yes_win_rate"] * 100,
            color="#2ecc71",
            linewidth=1.5,
            label="Maker bought YES",
            alpha=0.8,
        )
        ax.plot(
            df["price"],
            df["no_win_rate"] * 100,
            color="#e74c3c",
            linewidth=1.5,
            label="Maker bought NO",
            alpha=0.8,
        )
        ax.plot(
            df["price"],
            df["implied_prob"] * 100,
            "k--",
            linewidth=1.5,
            alpha=0.7,
            label="Implied probability",
        )
        ax.set_xlabel("Maker's Purchase Price (cents)")
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Maker Win Rate by Position Direction")
        ax.set_xlim(1, 99)
        ax.set_xticks(range(0, 101, 10))
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "price": int(row["price"]),
                "Maker bought YES": round(row["yes_win_rate"] * 100, 2) if pd.notna(row["yes_win_rate"]) else None,
                "Maker bought NO": round(row["no_win_rate"] * 100, 2) if pd.notna(row["no_win_rate"]) else None,
                "Implied probability": round(row["implied_prob"] * 100, 2),
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="price",
            yKeys=["Maker bought YES", "Maker bought NO", "Implied probability"],
            title="Maker Win Rate by Position Direction",
            yUnit=UnitType.PERCENT,
            xLabel="Maker's Purchase Price (cents)",
            yLabel="Win Rate (%)",
            strokeDasharrays=[None, None, "5 5"],
        )
