"""Plot Kalshi calibration deviation over time."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class KalshiCalibrationDeviationOverTimeAnalysis(Analysis):
    """Analyze cumulative calibration deviation over time on Kalshi."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kalshi_calibration_deviation_over_time",
            description="Kalshi calibration accuracy measured as mean absolute deviation over time",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Query all trades joined with resolved market outcomes
        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, result
                FROM '{self.markets_dir}/*.parquet'
                WHERE result IN ('yes', 'no')
            ),
            trade_positions AS (
                -- Buyer side (taker)
                SELECT
                    t.created_time,
                    CASE
                        WHEN t.taker_side = 'yes' THEN t.yes_price
                        ELSE t.no_price
                    END AS price,
                    CASE
                        WHEN t.taker_side = m.result THEN true
                        ELSE false
                    END AS won
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.yes_price > 0 AND t.no_price > 0

                UNION ALL

                -- Seller side (counterparty)
                SELECT
                    t.created_time,
                    CASE
                        WHEN t.taker_side = 'yes' THEN t.no_price
                        ELSE t.yes_price
                    END AS price,
                    CASE
                        WHEN t.taker_side != m.result THEN true
                        ELSE false
                    END AS won
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.yes_price > 0 AND t.no_price > 0
            )
            SELECT created_time, price, won
            FROM trade_positions
            WHERE price >= 1 AND price <= 99
            ORDER BY created_time
            """
        ).df()

        # Group by week and compute cumulative calibration deviation
        df["created_time"] = pd.to_datetime(df["created_time"], utc=True)
        min_date = df["created_time"].min()
        max_date = df["created_time"].max()

        # Calculate week boundaries
        week_dates = pd.date_range(start=min_date, end=max_date, freq="W")

        dates = []
        deviations = []

        for end_date in week_dates:
            # Get ALL trades from start up to this week (cumulative)
            cumulative_df = df[df["created_time"] <= end_date]

            # Aggregate by price across all historical trades
            agg = (
                cumulative_df.groupby("price")
                .agg(
                    total=("won", "count"),
                    wins=("won", "sum"),
                )
                .reset_index()
            )

            # Skip if not enough cumulative data yet
            if agg["total"].sum() < 1000:
                continue

            # Calculate cumulative win rates
            prices = agg["price"].values.astype(float)
            win_rates = 100.0 * agg["wins"].values / agg["total"].values

            # Calculate mean absolute deviation from perfect calibration
            absolute_deviations = np.abs(win_rates - prices)
            mean_deviation = np.mean(absolute_deviations)

            dates.append(end_date)
            deviations.append(mean_deviation)

        # Create output dataframe
        output_df = pd.DataFrame({"date": dates, "mean_absolute_deviation": deviations})

        fig = self._create_figure(dates, deviations)
        chart = self._create_chart(output_df)

        return AnalysisOutput(figure=fig, data=output_df, chart=chart)

    def _create_figure(self, dates: list, deviations: list) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(dates, deviations, color="#4C72B0", linewidth=2)
        ax.fill_between(dates, deviations, alpha=0.3, color="#4C72B0")

        ax.set_xlabel("Date")
        ax.set_ylabel("Mean Absolute Deviation (%)")
        ax.set_title("Kalshi: Calibration Accuracy Over Time")

        ax.axhline(
            y=0,
            color="#D65F5F",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Perfect calibration",
        )

        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        plt.tight_layout()
        return fig

    def _create_chart(self, output_df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "deviation": round(row["mean_absolute_deviation"], 2),
            }
            for _, row in output_df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.AREA,
            data=chart_data,
            xKey="date",
            yKeys=["deviation"],
            title="Kalshi: Calibration Accuracy Over Time",
            yUnit=UnitType.PERCENT,
            xLabel="Date",
            yLabel="Mean Absolute Deviation (%)",
        )
