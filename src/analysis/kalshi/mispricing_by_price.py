"""Analyze mispricing percentage by contract price for takers, makers, and combined."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class MispricingByPriceAnalysis(Analysis):
    """Analyze mispricing by contract price on Kalshi."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="mispricing_by_price",
            description="Mispricing analysis by contract price for takers, makers, and combined",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Query for taker, maker, and combined mispricing by price
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
                    CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            ),
            maker_positions AS (
                SELECT
                    CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                    CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            ),
            taker_stats AS (
                SELECT
                    price,
                    COUNT(*) AS total_trades,
                    SUM(won) AS wins,
                    100.0 * SUM(won) / COUNT(*) AS win_rate
                FROM taker_positions
                GROUP BY price
            ),
            maker_stats AS (
                SELECT
                    price,
                    COUNT(*) AS total_trades,
                    SUM(won) AS wins,
                    100.0 * SUM(won) / COUNT(*) AS win_rate
                FROM maker_positions
                GROUP BY price
            ),
            combined_positions AS (
                SELECT * FROM taker_positions
                UNION ALL
                SELECT * FROM maker_positions
            ),
            combined_stats AS (
                SELECT
                    price,
                    COUNT(*) AS total_trades,
                    SUM(won) AS wins,
                    100.0 * SUM(won) / COUNT(*) AS win_rate
                FROM combined_positions
                GROUP BY price
            )
            SELECT
                t.price,
                t.total_trades AS taker_trades,
                t.wins AS taker_wins,
                t.win_rate AS taker_win_rate,
                m.total_trades AS maker_trades,
                m.wins AS maker_wins,
                m.win_rate AS maker_win_rate,
                c.total_trades AS combined_trades,
                c.wins AS combined_wins,
                c.win_rate AS combined_win_rate
            FROM taker_stats t
            JOIN maker_stats m ON t.price = m.price
            JOIN combined_stats c ON t.price = c.price
            WHERE t.price BETWEEN 1 AND 99
            ORDER BY t.price
            """
        ).df()

        # Calculate mispricing: (actual_win_rate - implied_probability) / implied_probability * 100
        # Price is in cents (1-99), so implied probability = price
        df["implied_probability"] = df["price"].astype(float)
        df["taker_mispricing_pct"] = (
            (df["taker_win_rate"] - df["implied_probability"])
            / df["implied_probability"]
            * 100
        )
        df["maker_mispricing_pct"] = (
            (df["maker_win_rate"] - df["implied_probability"])
            / df["implied_probability"]
            * 100
        )
        df["combined_mispricing_pct"] = (
            (df["combined_win_rate"] - df["implied_probability"])
            / df["implied_probability"]
            * 100
        )

        # Calculate mispricing in percentage points (pp) for chart
        df["taker_mispricing_pp"] = df["taker_win_rate"] - df["implied_probability"]
        df["maker_mispricing_pp"] = df["maker_win_rate"] - df["implied_probability"]
        df["combined_mispricing_pp"] = (
            df["combined_win_rate"] - df["implied_probability"]
        )

        fig = self._create_figure(df)
        chart = self._create_chart(df)

        return AnalysisOutput(figure=fig, data=df, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            df["price"],
            df["taker_mispricing_pct"],
            s=30,
            alpha=0.7,
            color="#e74c3c",
            edgecolors="none",
            label="Taker",
        )
        ax.scatter(
            df["price"],
            df["maker_mispricing_pct"],
            s=30,
            alpha=0.7,
            color="#2ecc71",
            edgecolors="none",
            label="Maker",
        )
        ax.scatter(
            df["price"],
            df["combined_mispricing_pct"],
            s=30,
            alpha=0.7,
            color="#4C72B0",
            edgecolors="none",
            label="Combined",
        )

        ax.axhline(
            y=0, linestyle="--", color="gray", linewidth=1.5, label="Perfect calibration"
        )
        ax.set_xlabel("Contract Price (cents)")
        ax.set_ylabel("Mispricing (%)")
        ax.set_title("Mispricing by Contract Price")
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 10))
        ax.set_xticks(range(0, 101, 1), minor=True)
        ax.legend(loc="lower right")

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "price": int(row["price"]),
                "Taker": round(row["taker_mispricing_pp"], 2),
                "Maker": round(row["maker_mispricing_pp"], 2),
                "Combined": round(row["combined_mispricing_pp"], 2),
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="price",
            yKeys=["Taker", "Maker", "Combined"],
            title="Mispricing by Contract Price",
            yUnit=UnitType.PERCENT,
            xLabel="Contract Price (cents)",
            yLabel="Mispricing (pp)",
            colors={"Taker": "#ef4444", "Maker": "#10b981", "Combined": "#6366f1"},
        )
