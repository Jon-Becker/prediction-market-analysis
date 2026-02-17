"""Empirical Kelly Criterion with Monte Carlo uncertainty quantification.

Implements position sizing methodology that accounts for uncertainty in edge
estimates. Uses historical trade patterns to construct return distributions,
Monte Carlo resampling to analyze drawdown risk, and uncertainty-adjusted
Kelly fractions for position sizing.

Reference: Kelly (1956), Thorp (2006), and empirical extensions using
the Becker dataset of 72M+ Kalshi trades.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType


class EmpiricalKellySizingAnalysis(Analysis):
    """Empirical Kelly position sizing with Monte Carlo drawdown analysis.

    Evaluates a simple longshot strategy (buy contracts below a price threshold)
    using the full Kalshi dataset, then computes standard Kelly vs empirical
    Kelly sizing adjusted for return distribution uncertainty.
    """

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        price_threshold: int = 20,
        n_simulations: int = 10_000,
        drawdown_percentile: float = 0.95,
    ):
        super().__init__(
            name="empirical_kelly_sizing",
            description="Empirical Kelly position sizing with Monte Carlo uncertainty quantification",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")
        self.price_threshold = price_threshold
        self.n_simulations = n_simulations
        self.drawdown_percentile = drawdown_percentile

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Phase 1 & 2: Extract historical trades matching the longshot strategy
        # and construct return distribution
        with self.progress("Extracting longshot strategy trades"):
            trades_df = con.execute(
                f"""
                WITH resolved_markets AS (
                    SELECT ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized'
                      AND result IN ('yes', 'no')
                )
                SELECT
                    t.ticker,
                    t.yes_price AS price,
                    t.count AS contracts,
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS entry_price,
                    t.taker_side,
                    m.result,
                    CASE
                        WHEN t.taker_side = m.result
                        THEN (100.0 - (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END))
                             / (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END)
                        ELSE -1.0
                    END AS return_pct
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END)
                      <= {self.price_threshold}
                """
            ).df()

        if trades_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No trades found matching strategy criteria", transform=ax.transAxes, ha="center")
            return AnalysisOutput(figure=fig)

        returns = trades_df["return_pct"].values

        # Compute strategy statistics
        win_rate = float((returns > 0).mean())
        avg_return = float(returns.mean())
        std_return = float(returns.std())
        n_trades = len(returns)

        # Standard Kelly calculation
        # f* = (p * b - q) / b where b = avg win / avg loss magnitude
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(np.abs(losses.mean())) if len(losses) > 0 else 1.0
        b = avg_win / avg_loss if avg_loss > 0 else avg_win
        kelly_standard = (win_rate * b - (1 - win_rate)) / b if b > 0 else 0.0
        kelly_standard = max(kelly_standard, 0.0)

        # Phase 3 & 4: Monte Carlo resampling for drawdown distribution
        with self.progress("Running Monte Carlo simulations"):
            max_drawdowns = self._monte_carlo_drawdowns(returns)

        # Phase 5: Uncertainty-adjusted Kelly
        cv_edge = std_return / abs(avg_return) if abs(avg_return) > 0 else float("inf")
        kelly_empirical = kelly_standard * max(0, 1 - cv_edge) if cv_edge < 1 else 0.0

        # Drawdown statistics
        dd_median = float(np.percentile(max_drawdowns, 50))
        dd_95 = float(np.percentile(max_drawdowns, 95))
        dd_99 = float(np.percentile(max_drawdowns, 99))

        # Build summary dataframe
        summary_df = pd.DataFrame(
            {
                "metric": [
                    "strategy",
                    "price_threshold",
                    "n_trades",
                    "win_rate",
                    "avg_return",
                    "std_return",
                    "avg_win",
                    "avg_loss",
                    "payoff_ratio",
                    "kelly_standard",
                    "cv_edge",
                    "kelly_empirical",
                    "drawdown_median",
                    "drawdown_p95",
                    "drawdown_p99",
                    "n_simulations",
                ],
                "value": [
                    f"Buy taker contracts <= {self.price_threshold}c",
                    self.price_threshold,
                    n_trades,
                    round(win_rate, 4),
                    round(avg_return, 4),
                    round(std_return, 4),
                    round(avg_win, 4),
                    round(avg_loss, 4),
                    round(b, 4),
                    round(kelly_standard, 4),
                    round(cv_edge, 4),
                    round(kelly_empirical, 4),
                    round(dd_median, 4),
                    round(dd_95, 4),
                    round(dd_99, 4),
                    self.n_simulations,
                ],
            }
        )

        fig = self._create_figure(returns, max_drawdowns, kelly_standard, kelly_empirical, cv_edge)
        chart = self._create_chart(max_drawdowns, kelly_standard, kelly_empirical)

        return AnalysisOutput(
            figure=fig,
            data=summary_df,
            chart=chart,
            metadata={
                "kelly_standard": kelly_standard,
                "kelly_empirical": kelly_empirical,
                "cv_edge": cv_edge,
                "drawdown_p95": dd_95,
                "n_trades": n_trades,
                "win_rate": win_rate,
            },
        )

    def _monte_carlo_drawdowns(self, returns: np.ndarray) -> np.ndarray:
        """Resample returns and compute max drawdown for each simulated path."""
        n = len(returns)
        max_drawdowns = np.empty(self.n_simulations)

        for i in range(self.n_simulations):
            # Resample with replacement (bootstrap)
            path_returns = np.random.choice(returns, size=n, replace=True)
            # Compute equity curve (starting at 1.0)
            equity = np.cumprod(1 + path_returns)
            # Running max
            running_max = np.maximum.accumulate(equity)
            # Drawdown at each point (guard against division by zero when equity wipes out)
            with np.errstate(invalid="ignore", divide="ignore"):
                drawdowns = np.where(running_max > 0, (running_max - equity) / running_max, 1.0)
            max_dd = np.nanmax(drawdowns)
            max_drawdowns[i] = max_dd if not np.isnan(max_dd) else 1.0

        return max_drawdowns

    def _create_figure(
        self,
        returns: np.ndarray,
        max_drawdowns: np.ndarray,
        kelly_std: float,
        kelly_emp: float,
        cv_edge: float,
    ) -> plt.Figure:
        """Create a 2x2 figure with return distribution, drawdown distribution,
        equity path examples, and Kelly sizing comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Empirical Kelly Sizing: Longshot Strategy (price <= {self.price_threshold}c)",
            fontsize=14,
            fontweight="bold",
        )

        # Top-left: Return distribution
        ax = axes[0, 0]
        ax.hist(returns, bins=50, color="#4C72B0", alpha=0.7, edgecolor="white")
        ax.axvline(returns.mean(), color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean: {returns.mean():.2%}")
        ax.set_xlabel("Trade Return")
        ax.set_ylabel("Frequency")
        ax.set_title("Empirical Return Distribution")
        ax.legend(fontsize=9)

        # Top-right: Drawdown distribution from Monte Carlo
        ax = axes[0, 1]
        ax.hist(max_drawdowns * 100, bins=80, color="#e74c3c", alpha=0.7, edgecolor="white")
        dd_95 = np.percentile(max_drawdowns, 95) * 100
        dd_50 = np.percentile(max_drawdowns, 50) * 100
        ax.axvline(dd_50, color="#2ecc71", linestyle="--", linewidth=1.5, label=f"Median: {dd_50:.1f}%")
        ax.axvline(dd_95, color="#e74c3c", linestyle="-", linewidth=2, label=f"95th pctl: {dd_95:.1f}%")
        ax.set_xlabel("Max Drawdown (%)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Drawdown Distribution ({self.n_simulations:,} paths)")
        ax.legend(fontsize=9)

        # Bottom-left: Sample equity paths
        ax = axes[1, 0]
        n = len(returns)
        for _i in range(min(50, self.n_simulations)):
            path = np.random.choice(returns, size=n, replace=True)
            equity = np.cumprod(1 + path)
            ax.plot(equity, alpha=0.1, color="#4C72B0", linewidth=0.5)
        # Plot median path
        median_path = np.cumprod(1 + returns)
        ax.plot(median_path, color="#e74c3c", linewidth=2, label="Original sequence")
        ax.set_xlabel("Trade Number")
        ax.set_ylabel("Equity Multiple")
        ax.set_title("Simulated Equity Paths")
        ax.legend(fontsize=9)
        ax.set_yscale("log")

        # Bottom-right: Kelly sizing comparison
        ax = axes[1, 1]
        labels = ["Standard\nKelly", "Half\nKelly", "Empirical\nKelly"]
        values = [kelly_std * 100, kelly_std * 50, kelly_emp * 100]
        colors = ["#e74c3c", "#f39c12", "#2ecc71"]
        bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="white", width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{val:.1f}%", ha="center", fontsize=11)
        ax.set_ylabel("Position Size (% of capital)")
        ax.set_title(f"Position Sizing Comparison (CV={cv_edge:.2f})")
        ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 10)

        plt.tight_layout()
        return fig

    def _create_chart(
        self,
        max_drawdowns: np.ndarray,
        kelly_std: float,
        kelly_emp: float,
    ) -> ChartConfig:
        """Create chart config for web display - drawdown distribution histogram."""
        # Build histogram data for the drawdown distribution
        counts, bin_edges = np.histogram(max_drawdowns * 100, bins=50)
        chart_data = [
            {
                "drawdown": round((bin_edges[i] + bin_edges[i + 1]) / 2, 1),
                "count": int(counts[i]),
            }
            for i in range(len(counts))
        ]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="drawdown",
            yKeys=["count"],
            title=f"Max Drawdown Distribution (Std Kelly: {kelly_std:.1%}, Empirical: {kelly_emp:.1%})",
            xLabel="Max Drawdown (%)",
            yLabel="Frequency",
        )
