"""Calibration surface analysis across price and time dimensions.

Extends standard 1D calibration (win rate vs price) to a 2D surface C(p,t)
where t is time remaining until market resolution. Reveals how longshot bias
and calibration quality vary as resolution approaches.

Computes the mispricing function M(p,t) = C(p,t) - p/100 across the
full price x time grid, identifying systematic opportunities that vary
with both contract price and temporal proximity to resolution.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType


class CalibrationSurfaceAnalysis(Analysis):
    """2D calibration surface C(p,t) across price and time-to-resolution.

    Analyzes how market calibration varies with both contract price and
    days remaining until close, producing a heatmap of systematic mispricing.
    """

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        price_bins: int = 10,
        time_bins: int = 8,
    ):
        super().__init__(
            name="calibration_surface",
            description="Calibration surface C(p,t) across price and time dimensions",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")
        self.price_bins = price_bins
        self.time_bins = time_bins

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        with self.progress("Computing calibration surface"):
            df = con.execute(
                f"""
                WITH resolved_markets AS (
                    SELECT ticker, result, close_time
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized'
                      AND result IN ('yes', 'no')
                      AND close_time IS NOT NULL
                ),
                trade_positions AS (
                    -- Taker side positions
                    SELECT
                        CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                        CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won,
                        DATEDIFF('day', t.created_time, m.close_time) AS days_to_close
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN resolved_markets m ON t.ticker = m.ticker
                    WHERE t.created_time < m.close_time

                    UNION ALL

                    -- Maker side positions
                    SELECT
                        CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                        CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won,
                        DATEDIFF('day', t.created_time, m.close_time) AS days_to_close
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN resolved_markets m ON t.ticker = m.ticker
                    WHERE t.created_time < m.close_time
                )
                SELECT
                    price,
                    days_to_close,
                    won
                FROM trade_positions
                WHERE price BETWEEN 1 AND 99
                  AND days_to_close >= 0
                """
            ).df()

        if df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No trades with time-to-close data found", transform=ax.transAxes, ha="center")
            return AnalysisOutput(figure=fig)

        # Bin prices and time
        price_edges = np.linspace(1, 99, self.price_bins + 1)
        price_labels = [f"{int(price_edges[i])}-{int(price_edges[i + 1])}" for i in range(self.price_bins)]
        df["price_bin"] = pd.cut(df["price"], bins=price_edges, labels=price_labels, include_lowest=True)

        # Use quantile-based time bins for more even distribution
        time_quantiles = np.linspace(0, 1, self.time_bins + 1)
        time_edges = np.quantile(df["days_to_close"].dropna(), time_quantiles)
        time_edges = np.unique(time_edges)  # Remove duplicates
        actual_time_bins = len(time_edges) - 1
        time_labels = [f"{int(time_edges[i])}-{int(time_edges[i + 1])}d" for i in range(actual_time_bins)]
        df["time_bin"] = pd.cut(df["days_to_close"], bins=time_edges, labels=time_labels, include_lowest=True)

        # Compute calibration surface
        surface = (
            df.groupby(["price_bin", "time_bin"], observed=True)
            .agg(
                win_rate=("won", "mean"),
                n_trades=("won", "count"),
            )
            .reset_index()
        )

        # Compute implied probability as midpoint of each price bin
        price_midpoints = {label: (price_edges[i] + price_edges[i + 1]) / 2 for i, label in enumerate(price_labels)}
        surface["implied_prob"] = surface["price_bin"].astype(str).map(price_midpoints) / 100.0
        surface["mispricing"] = surface["win_rate"] - surface["implied_prob"]
        surface["mispricing_pct"] = surface["mispricing"] * 100

        # Filter to cells with sufficient data
        min_trades = 50
        surface_filtered = surface[surface["n_trades"] >= min_trades].copy()

        fig = self._create_figure(surface_filtered, price_labels, time_labels)
        chart = self._create_chart(surface_filtered)

        return AnalysisOutput(
            figure=fig,
            data=surface,
            chart=chart,
            metadata={
                "price_bins": self.price_bins,
                "time_bins": actual_time_bins,
                "total_trades": len(df),
                "min_trades_per_cell": min_trades,
            },
        )

    def _create_figure(
        self,
        surface: pd.DataFrame,
        price_labels: list[str],
        time_labels: list[str],
    ) -> plt.Figure:
        """Create a 1x2 figure with mispricing heatmap and marginal plots."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("Calibration Surface: Mispricing by Price and Time to Resolution", fontsize=14, fontweight="bold")

        # Left: Mispricing heatmap M(p,t)
        ax = axes[0]
        pivot = surface.pivot_table(
            values="mispricing_pct",
            index="price_bin",
            columns="time_bin",
            aggfunc="mean",
        )
        # Reindex to ensure proper ordering
        pivot = pivot.reindex(index=[lb for lb in price_labels if lb in pivot.index])
        pivot = pivot.reindex(columns=[lb for lb in time_labels if lb in pivot.columns])

        if not pivot.empty:
            vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 1)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(pivot.values, cmap="RdBu_r", norm=norm, aspect="auto", origin="lower")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            ax.set_xlabel("Days to Resolution")
            ax.set_ylabel("Price Range (cents)")
            ax.set_title("M(p,t) = Actual Win Rate - Implied Probability (pp)")

            # Annotate cells
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color="black")

            fig.colorbar(im, ax=ax, label="Mispricing (pp)", shrink=0.8)

        # Right: Mispricing by time period (marginal over price)
        ax = axes[1]
        time_marginal = surface.groupby("time_bin", observed=True).agg(
            mispricing=("mispricing_pct", "mean"),
            n_trades=("n_trades", "sum"),
        )
        time_marginal = time_marginal.reindex([lb for lb in time_labels if lb in time_marginal.index])

        if not time_marginal.empty:
            colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in time_marginal["mispricing"]]
            bars = ax.bar(range(len(time_marginal)), time_marginal["mispricing"], color=colors, alpha=0.8)
            ax.set_xticks(range(len(time_marginal)))
            ax.set_xticklabels(time_marginal.index, rotation=45, ha="right", fontsize=8)
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Days to Resolution")
            ax.set_ylabel("Avg Mispricing (pp)")
            ax.set_title("Mean Mispricing by Time Period")

            # Add trade count labels
            for bar, n in zip(bars, time_marginal["n_trades"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1 if bar.get_height() >= 0 else bar.get_height() - 0.3,
                    f"n={n:,.0f}",
                    ha="center",
                    fontsize=7,
                )

        plt.tight_layout()
        return fig

    def _create_chart(self, surface: pd.DataFrame) -> ChartConfig:
        """Create chart config for web display - heatmap of mispricing surface."""
        chart_data = [
            {
                "price": str(row["price_bin"]),
                "time": str(row["time_bin"]),
                "value": round(row["mispricing_pct"], 2),
            }
            for _, row in surface.iterrows()
            if not np.isnan(row["mispricing_pct"])
        ]

        return ChartConfig(
            type=ChartType.HEATMAP,
            data=chart_data,
            xKey="time",
            yKey="price",
            title="Calibration Surface: Mispricing M(p,t) in Percentage Points",
            xLabel="Days to Resolution",
            yLabel="Price Range (cents)",
        )
