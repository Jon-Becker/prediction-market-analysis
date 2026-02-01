"""Analyze Polymarket win rate by price to assess market calibration."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class PolymarketWinRateByPriceAnalysis(Analysis):
    """Analyze win rate by price to assess market calibration on Polymarket."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="polymarket_win_rate_by_price",
            description="Polymarket win rate vs price market calibration analysis",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "polymarket" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "polymarket" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Step 1: Build a mapping of token_id -> (market_id, won) for resolved markets
        # A market is resolved if one outcome price is > 0.99 and the other < 0.01
        markets_df = con.execute(
            f"""
            SELECT id, clob_token_ids, outcome_prices
            FROM '{self.markets_dir}/*.parquet'
            WHERE closed = true
            """
        ).df()

        # Build token -> won mapping
        token_won = {}
        for _, row in markets_df.iterrows():
            try:
                token_ids = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
                prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
                if not token_ids or not prices or len(token_ids) != 2 or len(prices) != 2:
                    continue
                p0, p1 = float(prices[0]), float(prices[1])
                # Check if clearly resolved (one near 1, other near 0)
                if p0 > 0.99 and p1 < 0.01:
                    token_won[token_ids[0]] = True
                    token_won[token_ids[1]] = False
                elif p0 < 0.01 and p1 > 0.99:
                    token_won[token_ids[0]] = False
                    token_won[token_ids[1]] = True
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # Step 2: Register the token mapping as a DuckDB table for efficient joining
        token_data = list(token_won.items())
        con.execute("CREATE TABLE token_resolution (token_id VARCHAR, won BOOLEAN)")
        con.executemany("INSERT INTO token_resolution VALUES (?, ?)", token_data)

        # Step 3: Query trades and join with resolution data
        # When maker_asset_id = 0, maker provides USDC and receives outcome tokens
        # Price = maker_amount / taker_amount (in cents, since both are in 6 decimals)
        # When taker_asset_id = 0, taker provides USDC and receives outcome tokens
        # Price = taker_amount / maker_amount
        df = con.execute(
            f"""
            WITH trade_positions AS (
                -- Buyer side (buying outcome tokens with USDC)
                SELECT
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

                -- Seller side (selling outcome tokens for USDC) - counterparty
                SELECT
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
            SELECT
                price,
                COUNT(*) AS total_trades,
                SUM(CASE WHEN won THEN 1 ELSE 0 END) AS wins,
                100.0 * SUM(CASE WHEN won THEN 1 ELSE 0 END) / COUNT(*) AS win_rate
            FROM trade_positions
            WHERE price >= 1 AND price <= 99
            GROUP BY price
            ORDER BY price
            """
        ).df()

        fig = self._create_figure(df)
        chart = self._create_chart(df)

        return AnalysisOutput(figure=fig, data=df, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(
            df["price"],
            df["win_rate"],
            s=30,
            alpha=0.8,
            color="#4C72B0",
            edgecolors="none",
        )
        ax.plot(
            [0, 100],
            [0, 100],
            linestyle="--",
            color="#D65F5F",
            linewidth=1.5,
            label="Perfect calibration",
        )
        ax.set_xlabel("Contract Price (cents)")
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Polymarket: Win Rate vs Price (Market Calibration)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 101, 10))
        ax.set_xticks(range(0, 101, 1), minor=True)
        ax.set_yticks(range(0, 101, 10))
        ax.set_yticks(range(0, 101, 1), minor=True)
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "price": int(row["price"]),
                "actual": round(row["win_rate"], 2),
                "implied": int(row["price"]),
            }
            for _, row in df.iterrows()
            if 1 <= row["price"] <= 99
        ]

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="price",
            yKeys=["actual", "implied"],
            title="Polymarket: Actual Win Rate vs Contract Price",
            strokeDasharrays=[None, "5 5"],
            yUnit=UnitType.PERCENT,
            xLabel="Contract Price (cents)",
            yLabel="Actual Win Rate (%)",
        )
