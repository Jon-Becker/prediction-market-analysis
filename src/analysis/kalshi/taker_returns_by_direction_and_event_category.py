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


class TakerReturnsByDirectionEventCategoryAnalysis(Analysis):
    """Analyze taker returns by position direction (YES vs NO) and event category."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="taker_returns_by_direction_event_category",
            description="Taker excess returns by position direction (YES vs NO) and event category",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    @staticmethod
    def _get_category_input(prompt: str, valid_categories: list) -> int:
        while True:
            try:
                event_category = input(prompt)
                if event_category in valid_categories:
                    return event_category
                else:
                    print(f"Invalid event category. Please enter one of the following: {', '.join(valid_categories)}")
            except ValueError:
                print("Invalid event category. Please enter a valid category.")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Get distinct event categories for user selection
        categories_df = con.execute(
            f"""
            SELECT DISTINCT event_category
            FROM '{self.markets_dir}/*.parquet'
            WHERE status = 'finalized'
            """
        ).df()
        valid_categories = categories_df["event_category"].dropna().tolist()

        category = self._get_category_input(
            "Enter event category to analyze (e.g., 'Politics', 'Sports'): ", valid_categories
        )

        self.name += f"_{category.replace(' ', '_').lower()}"

        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, result
                FROM '{self.markets_dir}/*.parquet'
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
                  AND event_category = '{category}'
            ),
            taker_yes_positions AS (
                -- Taker bought YES (maker sold YES = maker bought NO)
                SELECT
                    t.yes_price AS price,
                    CASE WHEN m.result = 'yes' THEN 1.0 ELSE 0.0 END AS won,
                    t.count AS contracts,
                    'YES' AS taker_side
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.taker_side = 'yes'
            ),
            taker_no_positions AS (
                -- Taker bought NO (maker sold NO = maker bought YES)
                SELECT
                    t.no_price AS price,
                    CASE WHEN m.result = 'no' THEN 1.0 ELSE 0.0 END AS won,
                    t.count AS contracts,
                    'NO' AS taker_side
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.taker_side = 'no'
            ),
            all_taker_positions AS (
                SELECT * FROM taker_yes_positions
                UNION ALL
                SELECT * FROM taker_no_positions
            )
            SELECT
                taker_side,
                price,
                AVG(won) AS win_rate,
                price / 100.0 AS expected_win_rate,
                AVG(won) - price / 100.0 AS excess_return,
                VAR_POP(won - price / 100.0) AS var_excess,
                COUNT(*) AS n_trades,
                SUM(contracts) AS contracts,
                SUM(contracts * price / 100.0) AS volume_usd
            FROM all_taker_positions
            WHERE price BETWEEN 1 AND 99
            GROUP BY taker_side, price
            ORDER BY taker_side, price
            """
        ).df()

        # Pivot to compare YES vs NO at each price
        df_yes = df[df["taker_side"] == "YES"].copy()
        df_no = df[df["taker_side"] == "NO"].copy()

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
            label="Taker bought YES",
            alpha=0.8,
        )
        ax.plot(
            df["price"],
            df["no_excess"] * 100,
            color="#e74c3c",
            linewidth=1.5,
            label="Taker bought NO",
            alpha=0.8,
        )
        ax.fill_between(df["price"], df["yes_excess"] * 100, alpha=0.2, color="#2ecc71")
        ax.fill_between(df["price"], df["no_excess"] * 100, alpha=0.2, color="#e74c3c")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Taker's Purchase Price (cents)")
        ax.set_ylabel("Excess Return (pp)")
        ax.set_title("Taker Excess Returns by Position Direction")
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
                "Taker bought YES": round(row["yes_excess"] * 100, 2) if pd.notna(row["yes_excess"]) else None,
                "Taker bought NO": round(row["no_excess"] * 100, 2) if pd.notna(row["no_excess"]) else None,
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="price",
            yKeys=["Taker bought YES", "Taker bought NO"],
            title="Taker Excess Returns by Position Direction",
            yUnit=UnitType.PERCENT,
            xLabel="Taker's Purchase Price (cents)",
            yLabel="Excess Return (pp)",
        )
