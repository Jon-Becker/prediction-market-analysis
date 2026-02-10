"""Analyze Polymarket win rate by price to assess market calibration."""

from __future__ import annotations

import json
import math
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
        legacy_trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        collateral_lookup_path: Path | str | None = None,
    ):
        super().__init__(
            name="polymarket_win_rate_by_price",
            description="Polymarket win rate vs price market calibration analysis",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "polymarket" / "trades")
        self.legacy_trades_dir = Path(legacy_trades_dir or base_dir / "data" / "polymarket" / "legacy_trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "polymarket" / "markets")
        self.collateral_lookup_path = Path(
            collateral_lookup_path or base_dir / "data" / "polymarket" / "fpmm_collateral_lookup.json"
        )

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Step 1: Build CTF token_id -> won mapping for resolved markets
        # A market is resolved if one outcome price is > 0.99 and the other < 0.01
        markets_df = con.execute(
            f"""
            SELECT id, clob_token_ids, outcome_prices, market_maker_address
            FROM '{self.markets_dir}/*.parquet'
            WHERE closed = true
            """
        ).df()

        token_won: dict[str, bool] = {}
        fpmm_resolution: dict[str, int] = {}

        for _, row in markets_df.iterrows():
            try:
                prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
                if not prices or len(prices) != 2:
                    continue
                p0, p1 = float(prices[0]), float(prices[1])

                # Determine winning outcome
                winning_outcome = None
                if p0 > 0.99 and p1 < 0.01:
                    winning_outcome = 0
                elif p0 < 0.01 and p1 > 0.99:
                    winning_outcome = 1
                else:
                    continue

                # CTF token resolution
                token_ids = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
                if token_ids and len(token_ids) == 2:
                    token_won[token_ids[0]] = winning_outcome == 0
                    token_won[token_ids[1]] = winning_outcome == 1

                # FPMM resolution
                fpmm_addr = row.get("market_maker_address")
                if fpmm_addr:
                    fpmm_resolution[fpmm_addr.lower()] = winning_outcome

            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # Step 2: Register CTF token mapping
        con.execute("CREATE TABLE token_resolution (token_id VARCHAR, won BOOLEAN)")
        con.executemany("INSERT INTO token_resolution VALUES (?, ?)", list(token_won.items()))

        # Step 3: Filter FPMM resolution to USDC markets only
        if self.collateral_lookup_path.exists():
            with open(self.collateral_lookup_path) as f:
                collateral_lookup = json.load(f)
            usdc_markets = {
                addr.lower() for addr, info in collateral_lookup.items() if info["collateral_symbol"] == "USDC"
            }
            fpmm_resolution = {k: v for k, v in fpmm_resolution.items() if k in usdc_markets}

        # Register FPMM resolution table
        con.execute("CREATE TABLE fpmm_resolution (fpmm_address VARCHAR, winning_outcome BIGINT)")
        if fpmm_resolution:
            con.executemany("INSERT INTO fpmm_resolution VALUES (?, ?)", list(fpmm_resolution.items()))

        # Step 4: Build CTF trade positions query
        ctf_trades_query = f"""
            -- CTF Buyer side (buying outcome tokens with USDC)
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

            -- CTF Seller side (selling outcome tokens for USDC) - counterparty
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
        """

        # Step 5: Build legacy FPMM trade positions query
        legacy_trades_query = ""
        if fpmm_resolution and self.legacy_trades_dir.exists():
            legacy_trades_query = f"""
                UNION ALL

                -- Legacy FPMM Buyer side
                SELECT
                    ROUND(100.0 * t.amount::DOUBLE / t.outcome_tokens::DOUBLE) AS price,
                    (t.outcome_index = r.winning_outcome) AS won
                FROM '{self.legacy_trades_dir}/*.parquet' t
                INNER JOIN fpmm_resolution r ON LOWER(t.fpmm_address) = r.fpmm_address
                WHERE t.outcome_tokens::DOUBLE > 0

                UNION ALL

                -- Legacy FPMM Seller side (counterparty)
                SELECT
                    ROUND(100.0 - 100.0 * t.amount::DOUBLE / t.outcome_tokens::DOUBLE) AS price,
                    (t.outcome_index != r.winning_outcome) AS won
                FROM '{self.legacy_trades_dir}/*.parquet' t
                INNER JOIN fpmm_resolution r ON LOWER(t.fpmm_address) = r.fpmm_address
                WHERE t.outcome_tokens::DOUBLE > 0
            """

        # Step 6: Aggregate all trade positions by price
        df = con.execute(
            f"""
            WITH trade_positions AS (
                {ctf_trades_query}
                {legacy_trades_query}
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

        # Compute calibration metrics from aggregated data
        metrics = self._compute_calibration_metrics(df)

        fig = self._create_figure(df, metrics)
        chart = self._create_chart(df)

        return AnalysisOutput(figure=fig, data=df, chart=chart, metadata=metrics)

    def _compute_calibration_metrics(self, df: pd.DataFrame) -> dict:
        """Compute Brier score and ECE from aggregated price data.

        Brier score = mean((p - y)²) where p is predicted prob, y is outcome (0 or 1)
        For each price bucket:
        - wins contribute: (price/100 - 1)² per trade
        - losses contribute: (price/100 - 0)² per trade

        ECE (Expected Calibration Error) = weighted avg of |win_rate - price| across bins

        NOTE: This computes Brier score at trade execution time, which is the correct
        methodology for measuring market calibration. Some analyses (e.g., Dune's
        polymarket_data table) use price snapshots near resolution (e.g., price_1d_before),
        which produces artificially low Brier scores (~0.05) because markets have already
        converged toward 0 or 1 as outcomes become obvious. Our approach answers the
        question traders care about: "When I buy at X%, does the outcome happen X% of
        the time?" Expected Brier score for a well-calibrated market with trades across
        all price levels is ~0.17, not 0.05.
        """
        total_trades = df["total_trades"].sum()

        # Brier score: compute from individual trade contributions
        brier_sum = 0.0
        for _, row in df.iterrows():
            p = row["price"] / 100.0  # Convert cents to probability
            wins = row["wins"]
            losses = row["total_trades"] - wins
            # Wins: (p - 1)², Losses: (p - 0)²
            brier_sum += wins * (p - 1) ** 2 + losses * p**2

        brier_score = brier_sum / total_trades if total_trades > 0 else 0.0

        # ECE: weighted average of |actual_rate - predicted_rate|
        ece_sum = 0.0
        for _, row in df.iterrows():
            predicted = row["price"] / 100.0
            actual = row["win_rate"] / 100.0
            weight = row["total_trades"]
            ece_sum += weight * abs(actual - predicted)

        ece = ece_sum / total_trades if total_trades > 0 else 0.0

        # Log loss: -mean(y * log(p) + (1-y) * log(1-p))
        epsilon = 1e-6
        log_loss_sum = 0.0
        for _, row in df.iterrows():
            p = max(min(row["price"] / 100.0, 1 - epsilon), epsilon)
            wins = row["wins"]
            losses = row["total_trades"] - wins
            # Wins: -log(p), Losses: -log(1-p)
            log_loss_sum += wins * (-math.log(p)) + losses * (-math.log(1 - p))

        log_loss = log_loss_sum / total_trades if total_trades > 0 else 0.0

        return {
            "brier_score": round(brier_score, 4),
            "log_loss": round(log_loss, 4),
            "ece": round(ece, 4),
            "total_trades": int(total_trades),
        }

    def _create_figure(self, df: pd.DataFrame, metrics: dict | None = None) -> plt.Figure:
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

        # Add calibration metrics to figure
        if metrics:
            metrics_text = (
                f"Brier Score: {metrics['brier_score']:.4f}\n"
                f"Log Loss: {metrics['log_loss']:.4f}\n"
                f"ECE: {metrics['ece']:.4f}\n"
                f"Trades: {metrics['total_trades']:,}"
            )
            ax.text(
                0.98,
                0.02,
                metrics_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

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
