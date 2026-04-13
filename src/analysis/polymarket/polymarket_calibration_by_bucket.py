"""Polymarket calibration analysis grouped by probability buckets.

Groups resolved Polymarket trades into decile probability buckets (0-10%,
10-20%, ..., 90-100%) and compares the mean predicted probability against
the actual resolution rate within each bucket. A perfectly calibrated market
would show predicted and actual values matching along the diagonal.

This complements the per-cent win_rate_by_price scatter by aggregating into
the standard calibration curve format used in forecasting evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class PolymarketCalibrationByBucketAnalysis(Analysis):
    """Calibration analysis with decile probability buckets for Polymarket."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        legacy_trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        collateral_lookup_path: Path | str | None = None,
    ):
        super().__init__(
            name="polymarket_calibration_by_bucket",
            description="Polymarket calibration curve grouped by decile probability buckets",
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
        with self.progress("Loading resolved markets"):
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

                winning_outcome = None
                if p0 > 0.99 and p1 < 0.01:
                    winning_outcome = 0
                elif p0 < 0.01 and p1 > 0.99:
                    winning_outcome = 1
                else:
                    continue

                token_ids = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
                if token_ids and len(token_ids) == 2:
                    token_won[token_ids[0]] = winning_outcome == 0
                    token_won[token_ids[1]] = winning_outcome == 1

                fpmm_addr = row.get("market_maker_address")
                if isinstance(fpmm_addr, str) and fpmm_addr:
                    fpmm_resolution[fpmm_addr.lower()] = winning_outcome

            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # Step 2: Register resolution tables
        con.execute("CREATE TABLE token_resolution (token_id VARCHAR, won BOOLEAN)")
        con.executemany("INSERT INTO token_resolution VALUES (?, ?)", list(token_won.items()))

        # Filter FPMM to USDC markets
        if self.collateral_lookup_path.exists():
            with open(self.collateral_lookup_path) as f:
                collateral_lookup = json.load(f)
            usdc_markets = {
                addr.lower() for addr, info in collateral_lookup.items() if info["collateral_symbol"] == "USDC"
            }
            fpmm_resolution = {k: v for k, v in fpmm_resolution.items() if k in usdc_markets}

        con.execute("CREATE TABLE fpmm_resolution (fpmm_address VARCHAR, winning_outcome BIGINT)")
        if fpmm_resolution:
            con.executemany("INSERT INTO fpmm_resolution VALUES (?, ?)", list(fpmm_resolution.items()))

        # Step 3: Query all trade positions with prices and outcomes
        with self.progress("Querying trade positions"):
            ctf_trades_query = f"""
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

            legacy_trades_query = ""
            if fpmm_resolution and self.legacy_trades_dir.exists():
                legacy_trades_query = f"""
                    UNION ALL

                    SELECT
                        ROUND(100.0 * t.amount::DOUBLE / t.outcome_tokens::DOUBLE) AS price,
                        (t.outcome_index = r.winning_outcome) AS won
                    FROM '{self.legacy_trades_dir}/*.parquet' t
                    INNER JOIN fpmm_resolution r ON LOWER(t.fpmm_address) = r.fpmm_address
                    WHERE t.outcome_tokens::DOUBLE > 0

                    UNION ALL

                    SELECT
                        ROUND(100.0 - 100.0 * t.amount::DOUBLE / t.outcome_tokens::DOUBLE) AS price,
                        (t.outcome_index != r.winning_outcome) AS won
                    FROM '{self.legacy_trades_dir}/*.parquet' t
                    INNER JOIN fpmm_resolution r ON LOWER(t.fpmm_address) = r.fpmm_address
                    WHERE t.outcome_tokens::DOUBLE > 0
                """

            # Aggregate into 10% buckets
            df = con.execute(
                f"""
                WITH trade_positions AS (
                    {ctf_trades_query}
                    {legacy_trades_query}
                )
                SELECT
                    FLOOR(price / 10) * 10 AS bucket_low,
                    COUNT(*) AS total_trades,
                    SUM(CASE WHEN won THEN 1 ELSE 0 END) AS wins,
                    AVG(price) AS mean_predicted,
                    100.0 * SUM(CASE WHEN won THEN 1 ELSE 0 END) / COUNT(*) AS actual_rate
                FROM trade_positions
                WHERE price >= 1 AND price <= 99
                GROUP BY FLOOR(price / 10) * 10
                ORDER BY bucket_low
                """
            ).df()

        # Compute summary metrics
        metrics = self._compute_metrics(df)

        fig = self._create_figure(df, metrics)
        chart = self._create_chart(df)

        return AnalysisOutput(figure=fig, data=df, chart=chart, metadata=metrics)

    def _compute_metrics(self, df: pd.DataFrame) -> dict:
        """Compute calibration summary metrics across buckets.

        ECE (Expected Calibration Error): trade-weighted mean of
        |actual_rate - mean_predicted| across buckets.

        MCE (Maximum Calibration Error): largest absolute deviation
        in any single bucket.
        """
        total_trades = df["total_trades"].sum()
        if total_trades == 0:
            return {"ece": 0.0, "mce": 0.0, "total_trades": 0, "num_buckets": 0}

        deviations = np.abs(df["actual_rate"] - df["mean_predicted"])

        ece = float(np.average(deviations, weights=df["total_trades"]))
        mce = float(deviations.max())

        # Brier score across buckets (approximate from aggregated data)
        brier_sum = 0.0
        for _, row in df.iterrows():
            p = row["mean_predicted"] / 100.0
            wins = row["wins"]
            losses = row["total_trades"] - wins
            brier_sum += wins * (p - 1) ** 2 + losses * p**2

        brier_score = brier_sum / total_trades

        return {
            "ece": round(ece, 2),
            "mce": round(mce, 2),
            "brier_score": round(brier_score, 4),
            "total_trades": int(total_trades),
            "num_buckets": len(df),
        }

    def _create_figure(self, df: pd.DataFrame, metrics: dict) -> plt.Figure:
        """Create a grouped bar chart comparing predicted vs actual per bucket."""
        fig, ax = plt.subplots(figsize=(12, 7))

        bucket_labels = [f"{int(row['bucket_low'])}-{int(row['bucket_low']) + 10}%" for _, row in df.iterrows()]
        x = np.arange(len(bucket_labels))
        bar_width = 0.35

        ax.bar(
            x - bar_width / 2,
            df["mean_predicted"],
            bar_width,
            label="Mean Predicted (%)",
            color="#4C72B0",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.bar(
            x + bar_width / 2,
            df["actual_rate"],
            bar_width,
            label="Actual Resolution Rate (%)",
            color="#DD8452",
            edgecolor="white",
            linewidth=0.5,
        )

        # Perfect calibration reference line
        ax.plot(
            x,
            df["mean_predicted"],
            linestyle="--",
            color="#D65F5F",
            linewidth=1,
            alpha=0.5,
        )

        ax.set_xlabel("Probability Bucket")
        ax.set_ylabel("Rate (%)")
        ax.set_title("Polymarket: Calibration by Probability Bucket")
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.3)

        # Trade count labels above actual bars
        for i, (_, row) in enumerate(df.iterrows()):
            count = int(row["total_trades"])
            if count >= 1_000_000:
                label = f"{count / 1_000_000:.1f}M"
            elif count >= 1_000:
                label = f"{count / 1_000:.0f}K"
            else:
                label = str(count)
            ax.text(
                x[i] + bar_width / 2,
                row["actual_rate"] + 1.5,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#666666",
            )

        # Metrics annotation
        metrics_text = (
            f"ECE: {metrics['ece']:.2f}%\n"
            f"MCE: {metrics['mce']:.2f}%\n"
            f"Brier: {metrics['brier_score']:.4f}\n"
            f"Trades: {metrics['total_trades']:,}"
        )
        ax.text(
            0.98,
            0.50,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="center",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "bucket": f"{int(row['bucket_low'])}-{int(row['bucket_low']) + 10}%",
                "predicted": round(row["mean_predicted"], 2),
                "actual": round(row["actual_rate"], 2),
                "trades": int(row["total_trades"]),
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="bucket",
            yKeys=["predicted", "actual"],
            title="Polymarket: Calibration by Probability Bucket",
            yUnit=UnitType.PERCENT,
            xLabel="Probability Bucket",
            yLabel="Rate (%)",
        )
