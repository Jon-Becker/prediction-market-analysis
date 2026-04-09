"""Signal identification and statistical testing for alpha discovery.

Screens features for predictive power, tests statistical significance,
and identifies exploitable patterns in prediction market trade data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SignalReport:
    """Results from signal analysis on a single feature."""

    feature: str
    ic: float  # information coefficient (Spearman rank correlation with target)
    ic_pvalue: float
    ic_t_stat: float
    mean_return_long: float  # mean return of top quintile
    mean_return_short: float  # mean return of bottom quintile
    long_short_spread: float
    spread_t_stat: float
    spread_pvalue: float
    monotonicity: float  # rank correlation of quintile returns (ideal = 1.0)
    turnover: float  # mean absolute change in signal rank per period
    n_obs: int


class SignalAnalyzer:
    """Discover and validate alpha signals from featurized trade data."""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def screen_all_features(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "target_return",
    ) -> pd.DataFrame:
        """Screen all features for predictive power.

        Returns a DataFrame of SignalReport rows, sorted by absolute IC.
        """
        reports = []
        for feat in feature_cols:
            report = self._analyze_single_signal(df, feat, target_col)
            if report is not None:
                reports.append(report)

        if not reports:
            return pd.DataFrame()

        result = pd.DataFrame([r.__dict__ for r in reports])
        result["abs_ic"] = result["ic"].abs()
        result = result.sort_values("abs_ic", ascending=False).reset_index(drop=True)
        return result

    def _analyze_single_signal(
        self,
        df: pd.DataFrame,
        feature: str,
        target_col: str,
    ) -> SignalReport | None:
        """Analyze a single feature's predictive power."""
        valid = df[[feature, target_col]].dropna()
        if len(valid) < 100:
            return None

        x = valid[feature].values
        y = valid[target_col].values

        # Skip constant features
        if np.std(x) < 1e-10:
            return None

        # Information Coefficient (Spearman rank correlation)
        ic, ic_pval = stats.spearmanr(x, y)
        n = len(valid)
        ic_t = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-10))

        # Quintile analysis
        try:
            quintiles = pd.qcut(valid[feature], q=5, labels=False, duplicates="drop")
        except ValueError:
            return None

        if quintiles.nunique() < 3:
            return None

        quintile_returns = valid.groupby(quintiles)[target_col].mean()
        top_q = quintile_returns.iloc[-1]
        bottom_q = quintile_returns.iloc[0]
        spread = top_q - bottom_q

        # T-test on long-short spread
        top_mask = quintiles == quintiles.max()
        bottom_mask = quintiles == quintiles.min()
        top_vals = y[top_mask]
        bottom_vals = y[bottom_mask]

        if len(top_vals) < 10 or len(bottom_vals) < 10:
            return None

        t_stat, t_pval = stats.ttest_ind(top_vals, bottom_vals, equal_var=False)

        # Monotonicity: do quintile returns increase monotonically?
        q_means = quintile_returns.values
        mono_corr, _ = stats.spearmanr(range(len(q_means)), q_means)

        # Turnover: how much does the signal rank change between consecutive obs
        ranks = valid[feature].rank(pct=True)
        turnover = ranks.diff().abs().mean()

        return SignalReport(
            feature=feature,
            ic=float(ic),
            ic_pvalue=float(ic_pval),
            ic_t_stat=float(ic_t),
            mean_return_long=float(top_q),
            mean_return_short=float(bottom_q),
            long_short_spread=float(spread),
            spread_t_stat=float(t_stat),
            spread_pvalue=float(t_pval),
            monotonicity=float(mono_corr),
            turnover=float(turnover),
            n_obs=n,
        )

    def find_significant_signals(
        self, report_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter signal report to only statistically significant signals."""
        if report_df.empty:
            return report_df

        mask = (
            (report_df["ic_pvalue"] < self.significance_level)
            & (report_df["spread_pvalue"] < self.significance_level)
            & (report_df["abs_ic"] > 0.01)  # minimum practical IC
        )
        return report_df[mask].reset_index(drop=True)

    def detect_regime_clusters(
        self,
        df: pd.DataFrame,
        feature: str = "volume_imbalance_100",
        n_regimes: int = 3,
    ) -> pd.DataFrame:
        """Detect market regimes via simple quantile-based clustering.

        Splits the feature into n_regimes quantile buckets and computes
        statistics for each regime (mean return, volatility, trade count).
        """
        valid = df[[feature, "target_return", "yes_price"]].dropna()
        if len(valid) < n_regimes * 50:
            return pd.DataFrame()

        valid = valid.copy()
        valid["regime"] = pd.qcut(valid[feature], q=n_regimes, labels=False, duplicates="drop")

        regime_stats = (
            valid.groupby("regime")
            .agg(
                mean_return=("target_return", "mean"),
                std_return=("target_return", "std"),
                mean_price=("yes_price", "mean"),
                mean_signal=(feature, "mean"),
                count=("target_return", "count"),
            )
            .reset_index()
        )

        # Sharpe-like ratio per regime
        regime_stats["sharpe"] = (
            regime_stats["mean_return"] / regime_stats["std_return"].clip(lower=1e-6)
        )

        return regime_stats

    def correlation_matrix(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        method: str = "spearman",
    ) -> pd.DataFrame:
        """Compute pairwise feature correlation matrix.

        Useful for identifying redundant features before model training.
        """
        return df[feature_cols].corr(method=method)

    def test_signal_decay(
        self,
        df: pd.DataFrame,
        feature: str,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Test how a signal's predictive power decays over different horizons.

        Computes IC at each horizon to see if the signal is fast-decaying
        (suitable for short-term) or persistent (suitable for longer holding).
        """
        if horizons is None:
            horizons = [5, 10, 25, 50, 100, 200, 500]

        results = []
        for h in horizons:
            future_ret = df["yes_price"].shift(-h) - df["yes_price"]
            valid = pd.DataFrame({"signal": df[feature], "ret": future_ret}).dropna()
            if len(valid) < 100:
                continue
            ic, pval = stats.spearmanr(valid["signal"], valid["ret"])
            results.append({"horizon": h, "ic": ic, "ic_pvalue": pval, "n_obs": len(valid)})

        return pd.DataFrame(results)
