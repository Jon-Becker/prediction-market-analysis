"""Edge verification and robustness testing.

Validates that discovered signals are real and not artifacts of overfitting,
data snooping, or market microstructure noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.quant.backtest import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class EdgeVerification:
    """Results from edge verification tests."""

    is_significant: bool
    sharpe_pvalue: float  # probability that Sharpe > 0 by chance
    deflated_sharpe: float  # Sharpe adjusted for multiple testing
    return_ttest_pvalue: float
    bootstrap_ci_lower: float  # 95% CI lower bound of mean return
    bootstrap_ci_upper: float
    stability_score: float  # fraction of sub-periods with positive returns
    regime_robustness: float  # min Sharpe across time regimes
    sensitivity_results: pd.DataFrame


class EdgeVerifier:
    """Comprehensive edge verification and robustness testing."""

    def __init__(self, n_bootstrap: int = 1000, significance_level: float = 0.05):
        self.n_bootstrap = n_bootstrap
        self.significance_level = significance_level

    def verify(
        self,
        backtest: BacktestResult,
        n_trials: int = 1,
    ) -> EdgeVerification:
        """Run full edge verification suite."""
        pnls = backtest.trade_log["net_pnl"].values if len(backtest.trade_log) > 0 else np.array([])

        if len(pnls) < 30:
            return self._inconclusive_result()

        # 1. T-test: is mean return significantly different from zero?
        t_stat, t_pval = stats.ttest_1samp(pnls, 0)

        # 2. Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(pnls)

        # 3. Sharpe ratio significance (Lo, 2002)
        sharpe_pval = self._sharpe_pvalue(backtest.sharpe_ratio, len(pnls))

        # 4. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
        deflated = self._deflated_sharpe(
            backtest.sharpe_ratio, len(pnls), n_trials=n_trials
        )

        # 5. Time stability: split into sub-periods
        stability = self._time_stability(pnls)

        # 6. Regime robustness
        regime_rob = self._regime_robustness(pnls)

        # 7. Sensitivity analysis
        sensitivity = self._sensitivity_analysis(backtest)

        is_sig = (
            t_pval < self.significance_level
            and ci_lower > 0
            and stability >= 0.5
        )

        return EdgeVerification(
            is_significant=is_sig,
            sharpe_pvalue=float(sharpe_pval),
            deflated_sharpe=float(deflated),
            return_ttest_pvalue=float(t_pval),
            bootstrap_ci_lower=float(ci_lower),
            bootstrap_ci_upper=float(ci_upper),
            stability_score=float(stability),
            regime_robustness=float(regime_rob),
            sensitivity_results=sensitivity,
        )

    def _bootstrap_ci(
        self, pnls: np.ndarray, alpha: float = 0.05
    ) -> tuple[float, float]:
        """Non-parametric bootstrap 95% CI for mean PnL."""
        rng = np.random.default_rng(42)
        means = np.array([
            rng.choice(pnls, size=len(pnls), replace=True).mean()
            for _ in range(self.n_bootstrap)
        ])
        return float(np.percentile(means, 100 * alpha / 2)), float(
            np.percentile(means, 100 * (1 - alpha / 2))
        )

    def _sharpe_pvalue(self, sharpe: float, n: int) -> float:
        """P-value for Sharpe ratio under null of SR=0 (Lo, 2002).

        SE(SR) ~ 1/sqrt(n) for IID returns (conservative).
        """
        se = 1.0 / np.sqrt(n)
        z = sharpe / max(se, 1e-10)
        return float(2 * (1 - stats.norm.cdf(abs(z))))

    def _deflated_sharpe(
        self, sharpe: float, n: int, n_trials: int = 1
    ) -> float:
        """Deflated Sharpe Ratio accounting for multiple testing.

        Based on Bailey & Lopez de Prado (2014).
        Adjusts for the expected maximum Sharpe from n_trials independent tests.
        """
        if n_trials <= 1:
            return sharpe

        # Expected max Sharpe under null (n_trials independent tests)
        # E[max(Z_1,...,Z_k)] ~ sqrt(2 * ln(k)) for standard normals
        expected_max_sr = np.sqrt(2 * np.log(n_trials)) * (1.0 / np.sqrt(n))

        # Deflated SR: test if observed SR exceeds expected max under null
        se = 1.0 / np.sqrt(n)
        z = (sharpe - expected_max_sr) / max(se, 1e-10)
        return float(stats.norm.cdf(z))  # probability of observing this SR

    def _time_stability(self, pnls: np.ndarray, n_splits: int = 5) -> float:
        """Fraction of time sub-periods with positive mean return."""
        if len(pnls) < n_splits * 10:
            return 0.0

        splits = np.array_split(pnls, n_splits)
        positive = sum(1 for s in splits if s.mean() > 0)
        return positive / n_splits

    def _regime_robustness(self, pnls: np.ndarray, n_regimes: int = 4) -> float:
        """Minimum Sharpe ratio across time regimes (quartiles)."""
        if len(pnls) < n_regimes * 30:
            return 0.0

        splits = np.array_split(pnls, n_regimes)
        sharpes = []
        for s in splits:
            if len(s) < 10 or s.std() < 1e-10:
                sharpes.append(0.0)
            else:
                sharpes.append(s.mean() / s.std())
        return min(sharpes)

    def _sensitivity_analysis(self, backtest: BacktestResult) -> pd.DataFrame:
        """Test sensitivity to transaction costs and position sizing."""
        if len(backtest.trade_log) == 0:
            return pd.DataFrame()

        gross_pnls = backtest.trade_log["gross_pnl"].values
        costs = backtest.trade_log["cost"].values

        results = []

        # Sensitivity to transaction cost multiplier
        for cost_mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            adjusted_pnl = gross_pnls - costs * cost_mult
            mean_ret = adjusted_pnl.mean()
            std_ret = adjusted_pnl.std() if len(adjusted_pnl) > 1 else 1e-6
            sr = (mean_ret / max(std_ret, 1e-6)) * np.sqrt(25000)

            results.append({
                "parameter": "cost_multiplier",
                "value": cost_mult,
                "mean_pnl": float(mean_ret),
                "sharpe": float(sr),
                "total_pnl": float(adjusted_pnl.sum()),
                "win_rate": float((adjusted_pnl > 0).mean()),
            })

        return pd.DataFrame(results)

    def _inconclusive_result(self) -> EdgeVerification:
        return EdgeVerification(
            is_significant=False,
            sharpe_pvalue=1.0,
            deflated_sharpe=0.0,
            return_ttest_pvalue=1.0,
            bootstrap_ci_lower=0.0,
            bootstrap_ci_upper=0.0,
            stability_score=0.0,
            regime_robustness=0.0,
            sensitivity_results=pd.DataFrame(),
        )


def format_backtest_report(
    backtest: BacktestResult, verification: EdgeVerification
) -> str:
    """Format a human-readable report of backtest and verification results."""
    lines = [
        "=" * 60,
        "BACKTEST & EDGE VERIFICATION REPORT",
        "=" * 60,
        "",
        "--- Performance Metrics ---",
        f"Total Return:        ${backtest.total_return:,.2f}",
        f"Annualized Return:   ${backtest.annualized_return:,.2f}",
        f"Sharpe Ratio:        {backtest.sharpe_ratio:.3f}",
        f"Sortino Ratio:       {backtest.sortino_ratio:.3f}",
        f"Max Drawdown:        ${backtest.max_drawdown:,.2f}",
        f"Win Rate:            {backtest.win_rate:.1%}",
        f"Profit Factor:       {backtest.profit_factor:.2f}",
        f"Total Trades:        {backtest.total_trades:,}",
        f"Avg Trade PnL:       ${backtest.avg_trade_return:.4f}",
        "",
        "--- Edge Verification ---",
        f"Significant Edge:    {'YES' if verification.is_significant else 'NO'}",
        f"Return t-test p:     {verification.return_ttest_pvalue:.6f}",
        f"Sharpe p-value:      {verification.sharpe_pvalue:.6f}",
        f"Deflated Sharpe:     {verification.deflated_sharpe:.4f}",
        f"Bootstrap 95% CI:    [${verification.bootstrap_ci_lower:.4f}, ${verification.bootstrap_ci_upper:.4f}]",
        f"Time Stability:      {verification.stability_score:.1%} of periods profitable",
        f"Regime Robustness:   {verification.regime_robustness:.4f} (min sub-period Sharpe)",
        "",
    ]

    if not verification.sensitivity_results.empty:
        lines.append("--- Cost Sensitivity ---")
        for _, row in verification.sensitivity_results.iterrows():
            lines.append(
                f"  Cost x{row['value']:.2f}: "
                f"Sharpe={row['sharpe']:.3f}, "
                f"Total=${row['total_pnl']:,.2f}, "
                f"WR={row['win_rate']:.1%}"
            )

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
