"""Maker profitability framework with order flow decomposition.

Extends maker/taker analysis to simulate market-making economics:
spread capture, inventory risk, adverse selection by trade size,
and cumulative PnL. Quantifies the structural edge available to
passive liquidity providers vs aggressive takers.

Based on Becker (2026) findings: takers exhibit negative excess returns
at 80 of 99 price levels, makers profit via structural arbitrage not
superior forecasting (Cohen's d ~ 0.02 between YES/NO maker returns).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class MakerProfitabilityFrameworkAnalysis(Analysis):
    """Market-making profitability framework with order flow decomposition.

    Analyzes maker economics across price levels: spread capture, adverse
    selection risk by trade size, cumulative PnL simulation, and inventory
    exposure tracking.
    """

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="maker_profitability_framework",
            description="Maker profitability framework with spread capture and adverse selection analysis",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Core query: maker and taker positions with trade size and direction
        with self.progress("Computing maker/taker order flow"):
            df = con.execute(
                f"""
                WITH resolved_markets AS (
                    SELECT ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized'
                      AND result IN ('yes', 'no')
                )
                SELECT
                    t.ticker,
                    t.created_time,
                    t.count AS contracts,
                    t.yes_price,
                    t.no_price,
                    t.taker_side,
                    m.result,
                    -- Taker metrics
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS taker_price,
                    CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS taker_won,
                    -- Maker metrics (counterparty)
                    CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS maker_price,
                    CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS maker_won,
                    -- Maker direction
                    CASE WHEN t.taker_side = 'yes' THEN 'no' ELSE 'yes' END AS maker_side
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                """
            ).df()

        if df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No resolved trades found", transform=ax.transAxes, ha="center")
            return AnalysisOutput(figure=fig)

        # Compute PnL per trade (in cents per contract)
        df["taker_pnl"] = df.apply(
            lambda r: (100 - r["taker_price"]) * r["contracts"]
            if r["taker_won"]
            else -r["taker_price"] * r["contracts"],
            axis=1,
        )
        df["maker_pnl"] = df.apply(
            lambda r: (100 - r["maker_price"]) * r["contracts"]
            if r["maker_won"]
            else -r["maker_price"] * r["contracts"],
            axis=1,
        )

        # Sort by time for cumulative PnL
        df = df.sort_values("created_time").reset_index(drop=True)

        # Analysis 1: Adverse selection by trade size
        size_analysis = self._adverse_selection_by_size(df)

        # Analysis 2: Maker excess return by direction (YES vs NO)
        direction_analysis = self._maker_direction_analysis(df)

        # Analysis 3: Cumulative PnL
        df["cum_taker_pnl"] = df["taker_pnl"].cumsum()
        df["cum_maker_pnl"] = df["maker_pnl"].cumsum()

        fig = self._create_figure(df, size_analysis, direction_analysis)
        chart = self._create_chart(size_analysis)

        # Summary statistics
        summary = pd.DataFrame(
            {
                "metric": [
                    "total_trades",
                    "total_contracts",
                    "maker_total_pnl_usd",
                    "taker_total_pnl_usd",
                    "maker_win_rate",
                    "taker_win_rate",
                    "maker_avg_excess_return",
                    "taker_avg_excess_return",
                    "maker_yes_excess",
                    "maker_no_excess",
                    "direction_cohens_d",
                ],
                "value": [
                    len(df),
                    int(df["contracts"].sum()),
                    round(df["maker_pnl"].sum() / 100, 2),
                    round(df["taker_pnl"].sum() / 100, 2),
                    round(df["maker_won"].mean(), 4),
                    round(df["taker_won"].mean(), 4),
                    round((df["maker_won"] - df["maker_price"] / 100).mean(), 4),
                    round((df["taker_won"] - df["taker_price"] / 100).mean(), 4),
                    round(direction_analysis["yes_excess"], 4),
                    round(direction_analysis["no_excess"], 4),
                    round(direction_analysis["cohens_d"], 4),
                ],
            }
        )

        return AnalysisOutput(
            figure=fig,
            data=summary,
            chart=chart,
            metadata={
                "maker_pnl_usd": round(df["maker_pnl"].sum() / 100, 2),
                "taker_pnl_usd": round(df["taker_pnl"].sum() / 100, 2),
                "cohens_d": round(direction_analysis["cohens_d"], 4),
            },
        )

    def _adverse_selection_by_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze maker excess returns by trade size bucket.

        Small fills = likely retail taker flow (biased).
        Large fills = potential informed flow (adverse selection risk).
        """
        size_quantiles = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
        size_edges = df["contracts"].quantile(size_quantiles).values
        size_edges = np.unique(size_edges)
        if len(size_edges) < 2:
            size_edges = np.array([df["contracts"].min(), df["contracts"].max()])

        labels = [f"{int(size_edges[i])}-{int(size_edges[i + 1])}" for i in range(len(size_edges) - 1)]
        df["size_bucket"] = pd.cut(df["contracts"], bins=size_edges, labels=labels, include_lowest=True)

        result = (
            df.groupby("size_bucket", observed=True)
            .agg(
                n_trades=("contracts", "count"),
                total_contracts=("contracts", "sum"),
                maker_win_rate=("maker_won", "mean"),
                taker_win_rate=("taker_won", "mean"),
                avg_maker_price=("maker_price", "mean"),
                avg_taker_price=("taker_price", "mean"),
                maker_pnl=("maker_pnl", "sum"),
                taker_pnl=("taker_pnl", "sum"),
            )
            .reset_index()
        )

        result["maker_expected"] = result["avg_maker_price"] / 100
        result["taker_expected"] = result["avg_taker_price"] / 100
        result["maker_excess"] = result["maker_win_rate"] - result["maker_expected"]
        result["taker_excess"] = result["taker_win_rate"] - result["taker_expected"]

        return result

    def _maker_direction_analysis(self, df: pd.DataFrame) -> dict:
        """Compare maker returns when buying YES vs NO.

        Near-identical returns proves structural (not informational) edge.
        """
        makers_yes = df[df["maker_side"] == "yes"]
        makers_no = df[df["maker_side"] == "no"]

        yes_excess = float(makers_yes["maker_won"].mean() - makers_yes["maker_price"].mean() / 100)
        no_excess = float(makers_no["maker_won"].mean() - makers_no["maker_price"].mean() / 100)

        # Cohen's d for effect size
        pooled_std = float(
            np.sqrt(
                (
                    makers_yes["maker_won"].var() * (len(makers_yes) - 1)
                    + makers_no["maker_won"].var() * (len(makers_no) - 1)
                )
                / (len(makers_yes) + len(makers_no) - 2)
            )
        )
        cohens_d = (yes_excess - no_excess) / pooled_std if pooled_std > 0 else 0.0

        return {
            "yes_excess": yes_excess,
            "no_excess": no_excess,
            "yes_n": len(makers_yes),
            "no_n": len(makers_no),
            "cohens_d": cohens_d,
        }

    def _create_figure(
        self,
        df: pd.DataFrame,
        size_analysis: pd.DataFrame,
        direction: dict,
    ) -> plt.Figure:
        """Create a 2x2 figure with cumulative PnL, adverse selection,
        maker direction comparison, and wealth transfer summary."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Maker Profitability Framework: Order Flow Decomposition", fontsize=14, fontweight="bold")

        # Top-left: Cumulative PnL (maker vs taker)
        ax = axes[0, 0]
        # Subsample for performance
        step = max(1, len(df) // 5000)
        ax.plot(
            df["cum_maker_pnl"].iloc[::step] / 100,
            color="#2ecc71",
            linewidth=1.5,
            label="Maker",
        )
        ax.plot(
            df["cum_taker_pnl"].iloc[::step] / 100,
            color="#e74c3c",
            linewidth=1.5,
            label="Taker",
        )
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Trade Index")
        ax.set_ylabel("Cumulative PnL ($)")
        ax.set_title("Wealth Transfer: Maker vs Taker")
        ax.legend(fontsize=9)
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

        # Top-right: Adverse selection by trade size
        ax = axes[0, 1]
        if not size_analysis.empty:
            x = range(len(size_analysis))
            width = 0.35
            ax.bar(
                [i - width / 2 for i in x],
                size_analysis["maker_excess"] * 100,
                width,
                color="#2ecc71",
                alpha=0.8,
                label="Maker excess",
            )
            ax.bar(
                [i + width / 2 for i in x],
                size_analysis["taker_excess"] * 100,
                width,
                color="#e74c3c",
                alpha=0.8,
                label="Taker excess",
            )
            ax.set_xticks(list(x))
            ax.set_xticklabels(size_analysis["size_bucket"], rotation=45, ha="right", fontsize=8)
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Trade Size (contracts)")
            ax.set_ylabel("Excess Return (pp)")
            ax.set_title("Adverse Selection: Excess Return by Trade Size")
            ax.legend(fontsize=9)

        # Bottom-left: Maker direction comparison (YES vs NO)
        ax = axes[1, 0]
        labels = ["Maker buying YES", "Maker buying NO"]
        values = [direction["yes_excess"] * 100, direction["no_excess"] * 100]
        colors = ["#3498db", "#9b59b6"]
        bars = ax.bar(labels, values, color=colors, alpha=0.8, width=0.5)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.2f}pp", ha="center", fontsize=10
            )
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Excess Return (pp)")
        ax.set_title(f"Maker Direction Symmetry (Cohen's d = {direction['cohens_d']:.3f})")

        # Bottom-right: Price-level profitability count
        ax = axes[1, 1]
        price_groups = (
            df.groupby("taker_price")
            .agg(
                taker_wr=("taker_won", "mean"),
            )
            .reset_index()
        )
        price_groups["excess"] = price_groups["taker_wr"] - price_groups["taker_price"] / 100
        n_negative = int((price_groups["excess"] < 0).sum())
        n_positive = int((price_groups["excess"] >= 0).sum())

        ax.bar(
            ["Taker loses\n(negative excess)", "Taker wins\n(positive excess)"],
            [n_negative, n_positive],
            color=["#e74c3c", "#2ecc71"],
            alpha=0.8,
            width=0.5,
        )
        for i, v in enumerate([n_negative, n_positive]):
            ax.text(i, v + 0.5, f"{v} / {n_negative + n_positive}", ha="center", fontsize=11)
        ax.set_ylabel("Number of Price Levels")
        ax.set_title("Taker Excess Return by Price Level")

        plt.tight_layout()
        return fig

    def _create_chart(self, size_analysis: pd.DataFrame) -> ChartConfig:
        """Create chart config for web display - adverse selection by trade size."""
        chart_data = [
            {
                "size": str(row["size_bucket"]),
                "Maker Excess": round(row["maker_excess"] * 100, 2),
                "Taker Excess": round(row["taker_excess"] * 100, 2),
            }
            for _, row in size_analysis.iterrows()
        ]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="size",
            yKeys=["Maker Excess", "Taker Excess"],
            title="Adverse Selection: Excess Returns by Trade Size",
            yUnit=UnitType.PERCENT,
            xLabel="Trade Size (contracts)",
            yLabel="Excess Return (pp)",
            colors={"Maker Excess": "#10b981", "Taker Excess": "#ef4444"},
        )
