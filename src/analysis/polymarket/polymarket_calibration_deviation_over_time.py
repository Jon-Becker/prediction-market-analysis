"""Plot Polymarket calibration deviation over time."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.polymarket.util import block_to_date
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType

# Polygon block time is ~2 seconds, so ~43,200 blocks per day
BLOCKS_PER_DAY = 43200
BLOCKS_PER_WEEK = BLOCKS_PER_DAY * 7


class PolymarketCalibrationDeviationOverTimeAnalysis(Analysis):
    """Plot Polymarket calibration deviation over time."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="polymarket_calibration_deviation_over_time",
            description="Polymarket calibration accuracy over time analysis",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "polymarket" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "polymarket" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Step 1: Build token -> won mapping for resolved markets
        markets_df = con.execute(
            f"""
            SELECT id, clob_token_ids, outcome_prices
            FROM '{self.markets_dir}/*.parquet'
            WHERE closed = true
            """
        ).df()

        token_won = {}
        for _, row in markets_df.iterrows():
            try:
                token_ids = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
                prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
                if not token_ids or not prices or len(token_ids) != 2 or len(prices) != 2:
                    continue
                p0, p1 = float(prices[0]), float(prices[1])
                if p0 > 0.99 and p1 < 0.01:
                    token_won[token_ids[0]] = True
                    token_won[token_ids[1]] = False
                elif p0 < 0.01 and p1 > 0.99:
                    token_won[token_ids[0]] = False
                    token_won[token_ids[1]] = True
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # Step 2: Register token mapping
        token_data = list(token_won.items())
        con.execute("CREATE TABLE token_resolution (token_id VARCHAR, won BOOLEAN)")
        con.executemany("INSERT INTO token_resolution VALUES (?, ?)", token_data)

        # Step 3: Query all trades with block numbers for time grouping
        trades_df = con.execute(
            f"""
            WITH trade_positions AS (
                -- Buyer side
                SELECT
                    t.block_number,
                    CASE
                        WHEN t.maker_asset_id = '0' THEN ROUND(100.0 * t.maker_amount / t.taker_amount)
                        ELSE ROUND(100.0 * t.taker_amount / t.maker_amount)
                    END AS price,
                    tr.won
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN token_resolution tr ON (
                    CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END = tr.token_id
                )
                WHERE t.taker_amount > 0 AND t.maker_amount > 0

                UNION ALL

                -- Seller side (counterparty)
                SELECT
                    t.block_number,
                    CASE
                        WHEN t.maker_asset_id = '0' THEN ROUND(100.0 - 100.0 * t.maker_amount / t.taker_amount)
                        ELSE ROUND(100.0 - 100.0 * t.taker_amount / t.maker_amount)
                    END AS price,
                    NOT tr.won AS won
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN token_resolution tr ON (
                    CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END = tr.token_id
                )
                WHERE t.taker_amount > 0 AND t.maker_amount > 0
            )
            SELECT block_number, price, won
            FROM trade_positions
            WHERE price >= 1 AND price <= 99
            ORDER BY block_number
            """
        ).df()

        # Step 4: Group by week and compute cumulative calibration deviation
        min_block = trades_df["block_number"].min()
        max_block = trades_df["block_number"].max()

        # Calculate week boundaries
        week_blocks = list(range(min_block, max_block + BLOCKS_PER_WEEK, BLOCKS_PER_WEEK))
        if week_blocks[-1] < max_block:
            week_blocks.append(max_block)

        dates = []
        deviations = []

        for end_block in week_blocks[1:]:
            # Get ALL trades from start up to this week (cumulative)
            cumulative_df = trades_df[trades_df["block_number"] <= end_block]

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
            # Perfect calibration: win_rate = price
            absolute_deviations = np.abs(win_rates - prices)
            mean_deviation = np.mean(absolute_deviations)

            current_date = block_to_date(end_block)
            dates.append(current_date)
            deviations.append(mean_deviation)

        # Create output dataframe
        df = pd.DataFrame({"date": dates, "deviation": deviations})

        fig = self._create_figure(dates, deviations)
        chart = self._create_chart(dates, deviations)

        return AnalysisOutput(figure=fig, data=df, chart=chart)

    def _create_figure(self, dates: list, deviations: list) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(dates, deviations, color="#4C72B0", linewidth=2)
        ax.fill_between(dates, deviations, alpha=0.3, color="#4C72B0")

        ax.set_xlabel("Date")
        ax.set_ylabel("Mean Absolute Deviation (%)")
        ax.set_title("Polymarket: Calibration Accuracy Over Time")

        # Add horizontal line at 0 for reference
        ax.axhline(y=0, color="#D65F5F", linestyle="--", linewidth=1, alpha=0.7, label="Perfect calibration")

        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        fig.autofmt_xdate()

        plt.tight_layout()
        return fig

    def _create_chart(self, dates: list, deviations: list) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "deviation": round(deviation, 2),
            }
            for date, deviation in zip(dates, deviations)
        ]

        return ChartConfig(
            type=ChartType.AREA,
            data=chart_data,
            xKey="date",
            yKeys=["deviation"],
            title="Polymarket: Calibration Accuracy Over Time",
            xLabel="Date",
            yLabel="Mean Absolute Deviation (%)",
            yUnit=UnitType.PERCENT,
        )
