"""Analyze maker returns by position direction (YES vs NO).

Tests whether maker profits are purely spread compensation or reflect
directional alpha by comparing maker performance when buying YES vs NO.
If makers systematically outperform on NO positions, this suggests selective
positioning rather than passive accommodation.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class MakerReturnsByDirectionAnalysis(Analysis):
    """Analyze maker returns by position direction (YES vs NO)."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="maker_returns_by_direction",
            description="Maker excess returns by position direction (YES vs NO)",
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
                AVG(won) AS win_rate,
                price / 100.0 AS expected_win_rate,
                AVG(won) - price / 100.0 AS excess_return,
                VAR_POP(won - price / 100.0) AS var_excess,
                COUNT(*) AS n_trades,
                SUM(contracts) AS contracts,
                SUM(contracts * price / 100.0) AS volume_usd
            FROM all_maker_positions
            WHERE price BETWEEN 1 AND 99
            GROUP BY maker_side, price
            ORDER BY maker_side, price
            """
        ).df()

        # Pivot to compare YES vs NO at each price
        df_yes = df[df["maker_side"] == "YES"].copy()
        df_no = df[df["maker_side"] == "NO"].copy()

        comparison = pd.merge(
            df_yes[["price", "win_rate", "excess_return", "n_trades", "contracts", "volume_usd"]].rename(
                columns={
                    "win_rate": "yes_win_rate",
                    "excess_return": "yes_excess",
                    "n_trades": "yes_n",
                    "contracts": "yes_contracts",
                    "volume_usd": "yes_volume",
                }
            ),
            df_no[["price", "win_rate", "excess_return", "n_trades", "contracts", "volume_usd"]].rename(
                columns={
                    "win_rate": "no_win_rate",
                    "excess_return": "no_excess",
                    "n_trades": "no_n",
                    "contracts": "no_contracts",
                    "volume_usd": "no_volume",
                }
            ),
            on="price",
            how="outer",
        )
        comparison = comparison.sort_values("price")
        comparison["diff"] = comparison["no_excess"] - comparison["yes_excess"]

        fig = self._create_figure(comparison)
        chart = self._create_chart(comparison)

        return AnalysisOutput(figure=fig, data=comparison, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(
            df["price"],
            df["yes_excess"] * 100,
            color="#2ecc71",
            linewidth=1.5,
            label="Maker bought YES",
            alpha=0.8,
        )
        ax.plot(
            df["price"],
            df["no_excess"] * 100,
            color="#e74c3c",
            linewidth=1.5,
            label="Maker bought NO",
            alpha=0.8,
        )
        ax.fill_between(df["price"], df["yes_excess"] * 100, alpha=0.2, color="#2ecc71")
        ax.fill_between(df["price"], df["no_excess"] * 100, alpha=0.2, color="#e74c3c")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Maker's Purchase Price (cents)")
        ax.set_ylabel("Excess Return (pp)")
        ax.set_title("Maker Excess Returns by Position Direction")
        ax.set_xlim(1, 99)
        ax.set_xticks(range(0, 101, 10))
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "price": int(row["price"]),
                "Maker bought YES": round(row["yes_excess"] * 100, 2) if pd.notna(row["yes_excess"]) else None,
                "Maker bought NO": round(row["no_excess"] * 100, 2) if pd.notna(row["no_excess"]) else None,
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="price",
            yKeys=["Maker bought YES", "Maker bought NO"],
            title="Maker Excess Returns by Position Direction",
            yUnit=UnitType.PERCENT,
            xLabel="Maker's Purchase Price (cents)",
            yLabel="Excess Return (pp)",
        )
