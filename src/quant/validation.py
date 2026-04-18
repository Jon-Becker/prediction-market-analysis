"""Real-world edge validation for prediction market trading strategies.

Tests whether a discovered edge is genuine and executable by running:
1. Purged walk-forward CV (eliminates feature leakage at train/test boundary)
2. Ticker-level hold-out (tests generalization to unseen markets)
3. Permutation test (tests if signal is distinguishable from noise)
4. Realistic execution model (slippage, latency, liquidity constraints)
5. Stress tests (adverse cost/sizing scenarios)
6. Random baseline comparison (model vs random direction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.quant.backtest import Backtester, BacktestResult
from src.quant.config import PipelineConfig
from src.quant.models import ModelTrainer

logger = logging.getLogger(__name__)


# ── Result dataclasses ─────────────────────────────────────────────────────


@dataclass
class PurgedCVResult:
    """Results from purged walk-forward cross-validation."""

    fold_sharpes: list[float]
    fold_ics: list[float]
    fold_n_trades: list[int]
    mean_sharpe: float
    std_sharpe: float
    mean_ic: float
    embargo_size: int
    n_folds: int


@dataclass
class TickerHoldoutResult:
    """Results from leave-one-ticker-out validation."""

    ticker_sharpes: dict[str, float]
    ticker_ics: dict[str, float]
    ticker_n_trades: dict[str, int]
    mean_sharpe: float
    std_sharpe: float
    mean_ic: float
    n_tickers: int


@dataclass
class PermutationTestResult:
    """Results from permutation test of model edge."""

    observed_sharpe: float
    permuted_sharpes: list[float]
    p_value: float
    n_permutations: int


@dataclass
class ExecutionModelResult:
    """Results from realistic execution simulation."""

    naive_sharpe: float
    realistic_sharpe: float
    sharpe_degradation_pct: float
    trades_filtered_by_liquidity: int
    trades_total: int
    mean_slippage_pct: float
    backtest_result: BacktestResult


@dataclass
class StressTestResult:
    """Results from stress testing under adverse conditions."""

    scenario_results: list[dict]


@dataclass
class RandomBaselineResult:
    """Results comparing model to random trading."""

    model_sharpe: float
    random_sharpes: list[float]
    random_mean_sharpe: float
    random_std_sharpe: float
    p_value: float
    n_baselines: int


@dataclass
class ValidationReport:
    """Consolidated results from the full validation suite."""

    purged_cv: PurgedCVResult
    ticker_holdout: TickerHoldoutResult
    permutation_test: PermutationTestResult
    execution_model: ExecutionModelResult
    stress_tests: StressTestResult
    random_baseline: RandomBaselineResult
    overall_pass: bool
    failure_reasons: list[str]


# ── Validator ──────────────────────────────────────────────────────────────


class RealWorldValidator:
    """Comprehensive real-world validation of trading strategies."""

    def __init__(self, config: PipelineConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def run_validation_suite(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        predictions: np.ndarray,
        model_method: str = "train_ridge",
    ) -> ValidationReport:
        """Run all validation tests and produce a consolidated report."""
        cfg = self.config

        # Subsample for expensive tests
        sample_size = min(cfg.validation_sample_size, len(df))
        df_sample = df.iloc[:sample_size].copy()

        logger.info("Running purged walk-forward CV...")
        purged_cv = self.purged_walk_forward_cv(df_sample, feature_cols, model_method)

        logger.info("Running ticker hold-out CV...")
        ticker_holdout = self.ticker_holdout_cv(df_sample, feature_cols, model_method)

        logger.info("Running permutation test (%d permutations)...", cfg.validation_n_permutations)
        perm_test = self.permutation_test(df_sample, feature_cols, model_method)

        logger.info("Running realistic execution backtest...")
        exec_model = self.realistic_execution_backtest(df, predictions)

        logger.info("Running stress tests...")
        stress = self.stress_test(df, predictions)

        logger.info("Running random baseline comparison (%d baselines)...", cfg.validation_n_random_baselines)
        random_bl = self.random_baseline_comparison(df, predictions)

        # Determine overall pass/fail
        failure_reasons: list[str] = []

        if purged_cv.mean_sharpe < cfg.min_sharpe_ratio:
            failure_reasons.append(
                f"Purged CV Sharpe {purged_cv.mean_sharpe:.2f} < {cfg.min_sharpe_ratio}"
            )
        if ticker_holdout.mean_sharpe < 0:
            failure_reasons.append(
                f"Ticker holdout Sharpe {ticker_holdout.mean_sharpe:.2f} < 0"
            )
        if perm_test.p_value > cfg.significance_level:
            failure_reasons.append(
                f"Permutation p-value {perm_test.p_value:.3f} > {cfg.significance_level}"
            )
        if exec_model.realistic_sharpe < cfg.min_sharpe_ratio:
            failure_reasons.append(
                f"Realistic execution Sharpe {exec_model.realistic_sharpe:.2f} < {cfg.min_sharpe_ratio}"
            )
        if stress.scenario_results:
            worst = min(s["sharpe"] for s in stress.scenario_results)
            if worst < 0:
                failure_reasons.append(f"Stress test worst Sharpe {worst:.2f} < 0")
        if random_bl.p_value > cfg.significance_level:
            failure_reasons.append(
                f"Random baseline p-value {random_bl.p_value:.3f} > {cfg.significance_level}"
            )

        overall_pass = len(failure_reasons) == 0

        return ValidationReport(
            purged_cv=purged_cv,
            ticker_holdout=ticker_holdout,
            permutation_test=perm_test,
            execution_model=exec_model,
            stress_tests=stress,
            random_baseline=random_bl,
            overall_pass=overall_pass,
            failure_reasons=failure_reasons,
        )

    # ── 1. Purged Walk-Forward CV ──────────────────────────────────────

    def purged_walk_forward_cv(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        model_method: str = "train_ridge",
        n_folds: int = 5,
    ) -> PurgedCVResult:
        """Walk-forward CV with embargo gap to prevent feature leakage."""
        cfg = self.config
        embargo = int(
            cfg.validation_embargo_multiplier
            * max(max(cfg.rolling_windows), cfg.target_horizon)
        )

        n = len(df)
        fold_size = n // (n_folds + 1)

        fold_sharpes = []
        fold_ics = []
        fold_n_trades = []

        for k in range(1, n_folds + 1):
            train_end = k * fold_size
            test_start = train_end + embargo
            test_end = min(test_start + fold_size, n)

            if test_start >= n or test_end - test_start < 200:
                continue

            train_slice = df.iloc[:train_end]
            test_slice = df.iloc[test_start:test_end]

            result = self._train_and_backtest(
                train_slice, test_slice, feature_cols, model_method
            )
            if result is not None:
                sharpe, ic, n_trades = result
                fold_sharpes.append(sharpe)
                fold_ics.append(ic)
                fold_n_trades.append(n_trades)

        mean_sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
        std_sharpe = float(np.std(fold_sharpes)) if len(fold_sharpes) > 1 else 0.0
        mean_ic = float(np.mean(fold_ics)) if fold_ics else 0.0

        return PurgedCVResult(
            fold_sharpes=fold_sharpes,
            fold_ics=fold_ics,
            fold_n_trades=fold_n_trades,
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            mean_ic=mean_ic,
            embargo_size=embargo,
            n_folds=len(fold_sharpes),
        )

    # ── 2. Ticker Hold-Out CV ──────────────────────────────────────────

    def ticker_holdout_cv(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        model_method: str = "train_ridge",
    ) -> TickerHoldoutResult:
        """Leave-one-ticker-out: tests generalization to unseen markets."""
        tickers = df["ticker"].unique()

        if len(tickers) < 2:
            logger.warning("Only %d ticker(s) — ticker holdout not meaningful.", len(tickers))
            return TickerHoldoutResult(
                ticker_sharpes={}, ticker_ics={}, ticker_n_trades={},
                mean_sharpe=0.0, std_sharpe=0.0, mean_ic=0.0, n_tickers=0,
            )

        ticker_sharpes: dict[str, float] = {}
        ticker_ics: dict[str, float] = {}
        ticker_n_trades: dict[str, int] = {}

        for t in tickers:
            train_df = df[df["ticker"] != t]
            test_df = df[df["ticker"] == t]

            if len(test_df) < self.config.min_trades_per_ticker:
                continue

            result = self._train_and_backtest(
                train_df, test_df, feature_cols, model_method
            )
            if result is not None:
                sharpe, ic, n_trades = result
                ticker_sharpes[t] = sharpe
                ticker_ics[t] = ic
                ticker_n_trades[t] = n_trades

        sharpe_vals = list(ticker_sharpes.values())
        ic_vals = list(ticker_ics.values())

        return TickerHoldoutResult(
            ticker_sharpes=ticker_sharpes,
            ticker_ics=ticker_ics,
            ticker_n_trades=ticker_n_trades,
            mean_sharpe=float(np.mean(sharpe_vals)) if sharpe_vals else 0.0,
            std_sharpe=float(np.std(sharpe_vals)) if len(sharpe_vals) > 1 else 0.0,
            mean_ic=float(np.mean(ic_vals)) if ic_vals else 0.0,
            n_tickers=len(ticker_sharpes),
        )

    # ── 3. Permutation Test ────────────────────────────────────────────

    def permutation_test(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        model_method: str = "train_ridge",
        n_permutations: int | None = None,
    ) -> PermutationTestResult:
        """Shuffle targets within each ticker, retrain, and compare Sharpe."""
        if n_permutations is None:
            n_permutations = self.config.validation_n_permutations

        # Observed (real) Sharpe
        observed = self._train_and_backtest_full(df, feature_cols, model_method)
        observed_sharpe = observed if observed is not None else 0.0

        permuted_sharpes = []
        target_cols = ["target_return", "future_return", "target_up", "future_price"]

        for i in range(n_permutations):
            df_perm = df.copy()
            # Grouped shuffle within each ticker to preserve per-ticker distributions
            for col in target_cols:
                if col in df_perm.columns:
                    df_perm[col] = df_perm.groupby("ticker")[col].transform(
                        lambda x: self.rng.permutation(x.values)
                    )

            perm_sharpe = self._train_and_backtest_full(df_perm, feature_cols, model_method)
            permuted_sharpes.append(perm_sharpe if perm_sharpe is not None else 0.0)

            if (i + 1) % 20 == 0:
                logger.info("  Permutation %d/%d done", i + 1, n_permutations)

        # p-value: fraction of permuted Sharpes >= observed (with +1 correction)
        n_ge = sum(1 for s in permuted_sharpes if s >= observed_sharpe)
        p_value = (n_ge + 1) / (n_permutations + 1)

        return PermutationTestResult(
            observed_sharpe=observed_sharpe,
            permuted_sharpes=permuted_sharpes,
            p_value=float(p_value),
            n_permutations=n_permutations,
        )

    # ── 4. Realistic Execution Model ───────────────────────────────────

    def realistic_execution_backtest(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
    ) -> ExecutionModelResult:
        """Backtest with slippage, latency, and liquidity constraints."""
        cfg = self.config
        df = df.copy()
        preds = predictions.copy()

        # Naive backtest (baseline)
        naive_bt = Backtester(
            transaction_cost_bps=cfg.transaction_cost_bps,
            max_position_usd=cfg.max_position_size,
            min_edge_threshold=0.5,
        )
        naive_result = naive_bt.run(df, preds)
        naive_sharpe = naive_result.sharpe_ratio

        # --- Liquidity filter ---
        rolling_vol = df["count"].rolling(cfg.validation_liquidity_window, min_periods=1).sum()
        vol_threshold = rolling_vol.quantile(cfg.validation_liquidity_volume_percentile / 100)
        illiquid_mask = rolling_vol < vol_threshold
        trades_filtered = int(illiquid_mask.sum())
        # Zero out predictions for illiquid periods (no trade)
        preds_filtered = preds.copy()
        preds_filtered[illiquid_mask.values[: len(preds_filtered)]] = 0.0

        # --- Latency penalty ---
        # Shift execution price: use price N trades later
        latency = cfg.validation_latency_trades
        df_exec = df.copy()
        if latency > 0 and "yes_price" in df_exec.columns:
            exec_price = df_exec["yes_price"].shift(-latency)
            # Recompute target return based on delayed execution
            future_price = df_exec["yes_price"].shift(-cfg.target_horizon)
            df_exec["target_return"] = (
                (future_price - exec_price) / exec_price.clip(lower=1) * 100
            )

        # --- Slippage via elevated transaction costs ---
        # Average slippage: base + size-dependent (approximate as flat addition)
        avg_position = cfg.max_position_size * 0.5  # rough average
        avg_slippage_pct = (
            cfg.validation_slippage_base_pct
            + cfg.validation_slippage_size_pct * (avg_position / cfg.validation_slippage_size_unit)
        )
        # Convert to bps and add to base cost
        slippage_bps = avg_slippage_pct * 100
        total_cost_bps = cfg.transaction_cost_bps + slippage_bps

        realistic_bt = Backtester(
            transaction_cost_bps=total_cost_bps,
            max_position_usd=cfg.max_position_size,
            min_edge_threshold=0.5,
        )
        realistic_result = realistic_bt.run(df_exec, preds_filtered)
        realistic_sharpe = realistic_result.sharpe_ratio

        degradation = (
            (naive_sharpe - realistic_sharpe) / max(abs(naive_sharpe), 1e-6) * 100
        )

        return ExecutionModelResult(
            naive_sharpe=float(naive_sharpe),
            realistic_sharpe=float(realistic_sharpe),
            sharpe_degradation_pct=float(degradation),
            trades_filtered_by_liquidity=trades_filtered,
            trades_total=len(df),
            mean_slippage_pct=float(avg_slippage_pct),
            backtest_result=realistic_result,
        )

    # ── 5. Stress Tests ────────────────────────────────────────────────

    def stress_test(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
    ) -> StressTestResult:
        """Run backtest under adverse conditions."""
        cfg = self.config
        scenarios: list[dict] = []

        # High cost scenarios
        for mult in cfg.validation_stress_cost_multipliers:
            bt = Backtester(
                transaction_cost_bps=cfg.transaction_cost_bps * mult,
                max_position_usd=cfg.max_position_size,
                min_edge_threshold=0.5,
            )
            result = bt.run(df, predictions)
            scenarios.append({
                "scenario": f"cost_{mult:.0f}x",
                "sharpe": float(result.sharpe_ratio),
                "total_return": float(result.total_return),
                "n_trades": result.total_trades,
                "win_rate": float(result.win_rate),
            })

        # Reduced position size scenarios
        for frac in cfg.validation_stress_max_position_fractions:
            bt = Backtester(
                transaction_cost_bps=cfg.transaction_cost_bps,
                max_position_usd=cfg.max_position_size * frac,
                min_edge_threshold=0.5,
            )
            result = bt.run(df, predictions)
            scenarios.append({
                "scenario": f"position_{frac:.0%}",
                "sharpe": float(result.sharpe_ratio),
                "total_return": float(result.total_return),
                "n_trades": result.total_trades,
                "win_rate": float(result.win_rate),
            })

        # Low-liquidity only
        rolling_vol = df["count"].rolling(cfg.validation_liquidity_window, min_periods=1).sum()
        low_liq_mask = rolling_vol <= rolling_vol.quantile(0.25)
        if low_liq_mask.sum() > 100:
            df_low_liq = df[low_liq_mask].reset_index(drop=True)
            preds_low_liq = predictions[low_liq_mask.values[: len(predictions)]]
            if len(preds_low_liq) > 0:
                bt = Backtester(
                    transaction_cost_bps=cfg.transaction_cost_bps,
                    max_position_usd=cfg.max_position_size,
                    min_edge_threshold=0.5,
                )
                result = bt.run(df_low_liq, preds_low_liq)
                scenarios.append({
                    "scenario": "low_liquidity_only",
                    "sharpe": float(result.sharpe_ratio),
                    "total_return": float(result.total_return),
                    "n_trades": result.total_trades,
                    "win_rate": float(result.win_rate),
                })

        # Combined worst case: 5x cost + 25% position + low liquidity
        if low_liq_mask.sum() > 100:
            bt = Backtester(
                transaction_cost_bps=cfg.transaction_cost_bps * 5,
                max_position_usd=cfg.max_position_size * 0.25,
                min_edge_threshold=0.5,
            )
            result = bt.run(df_low_liq, preds_low_liq)
            scenarios.append({
                "scenario": "worst_case_combined",
                "sharpe": float(result.sharpe_ratio),
                "total_return": float(result.total_return),
                "n_trades": result.total_trades,
                "win_rate": float(result.win_rate),
            })

        return StressTestResult(scenario_results=scenarios)

    # ── 6. Random Baseline ─────────────────────────────────────────────

    def random_baseline_comparison(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        n_baselines: int | None = None,
    ) -> RandomBaselineResult:
        """Compare model to random sign-flipped predictions."""
        if n_baselines is None:
            n_baselines = self.config.validation_n_random_baselines

        bt = Backtester(
            transaction_cost_bps=self.config.transaction_cost_bps,
            max_position_usd=self.config.max_position_size,
            min_edge_threshold=0.5,
        )

        model_result = bt.run(df, predictions)
        model_sharpe = model_result.sharpe_ratio

        random_sharpes = []
        for _ in range(n_baselines):
            # Sign-flip: preserves magnitude distribution, randomizes direction
            signs = self.rng.choice([-1, 1], size=len(predictions))
            random_preds = predictions * signs
            result = bt.run(df, random_preds)
            random_sharpes.append(float(result.sharpe_ratio))

        n_ge = sum(1 for s in random_sharpes if s >= model_sharpe)
        p_value = (n_ge + 1) / (n_baselines + 1)

        return RandomBaselineResult(
            model_sharpe=float(model_sharpe),
            random_sharpes=random_sharpes,
            random_mean_sharpe=float(np.mean(random_sharpes)),
            random_std_sharpe=float(np.std(random_sharpes)),
            p_value=float(p_value),
            n_baselines=n_baselines,
        )

    # ── Helpers ────────────────────────────────────────────────────────

    def _train_and_backtest(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        model_method: str,
    ) -> tuple[float, float, int] | None:
        """Train model on train_df, backtest on test_df. Returns (sharpe, ic, n_trades)."""
        valid_features = [f for f in feature_cols if f in train_df.columns and f in test_df.columns]
        required_cols = valid_features + ["target_return"]

        train_clean = train_df.dropna(subset=required_cols)
        test_clean = test_df.dropna(subset=required_cols)

        if len(train_clean) < 500 or len(test_clean) < 100:
            return None

        trainer = ModelTrainer(valid_features)
        train_split, val_split, _ = trainer.prepare_splits(train_clean, train_frac=0.8, val_frac=0.2)

        if len(train_split) < 200 or len(val_split) < 50:
            return None

        train_fn = getattr(trainer, model_method)
        model_result = train_fn(train_split, val_split)

        preds = trainer.predict(model_result, test_clean)

        bt = Backtester(
            transaction_cost_bps=self.config.transaction_cost_bps,
            max_position_usd=self.config.max_position_size,
            min_edge_threshold=0.5,
        )
        bt_result = bt.run(test_clean, preds)

        # IC on test set
        valid_mask = ~np.isnan(preds) & ~np.isnan(test_clean["target_return"].values)
        if valid_mask.sum() < 30:
            return None
        ic, _ = stats.spearmanr(
            preds[valid_mask], test_clean["target_return"].values[valid_mask]
        )

        return (float(bt_result.sharpe_ratio), float(ic), bt_result.total_trades)

    def _train_and_backtest_full(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        model_method: str,
    ) -> float | None:
        """Train/test on df with standard split, return Sharpe."""
        valid_features = [f for f in feature_cols if f in df.columns]
        required_cols = valid_features + ["target_return"]
        df_clean = df.dropna(subset=required_cols)

        if len(df_clean) < 1000:
            return None

        trainer = ModelTrainer(valid_features)
        train, val, test = trainer.prepare_splits(df_clean)

        if len(train) < 200 or len(val) < 50 or len(test) < 50:
            return None

        train_fn = getattr(trainer, model_method)
        model_result = train_fn(train, val)
        preds = trainer.predict(model_result, test)

        bt = Backtester(
            transaction_cost_bps=self.config.transaction_cost_bps,
            max_position_usd=self.config.max_position_size,
            min_edge_threshold=0.5,
        )
        bt_result = bt.run(test, preds)
        return float(bt_result.sharpe_ratio)


# ── Report Formatting ──────────────────────────────────────────────────────


def format_validation_report(report: ValidationReport) -> str:
    """Format a human-readable validation report."""
    lines = [
        "=" * 65,
        "REAL-WORLD EDGE VALIDATION REPORT",
        "=" * 65,
        "",
    ]

    # 1. Purged CV
    cv = report.purged_cv
    lines += [
        "--- 1. Purged Walk-Forward CV ---",
        f"  Folds:          {cv.n_folds}",
        f"  Embargo:        {cv.embargo_size} trades",
        f"  Mean Sharpe:    {cv.mean_sharpe:.3f} +/- {cv.std_sharpe:.3f}",
        f"  Mean IC:        {cv.mean_ic:.4f}",
        f"  Fold Sharpes:   {[f'{s:.2f}' for s in cv.fold_sharpes]}",
        "",
    ]

    # 2. Ticker Holdout
    th = report.ticker_holdout
    lines += [
        "--- 2. Ticker Hold-Out CV ---",
        f"  Tickers tested: {th.n_tickers}",
        f"  Mean Sharpe:    {th.mean_sharpe:.3f} +/- {th.std_sharpe:.3f}",
        f"  Mean IC:        {th.mean_ic:.4f}",
    ]
    if th.ticker_sharpes:
        sorted_tickers = sorted(th.ticker_sharpes.items(), key=lambda x: x[1], reverse=True)
        lines.append("  Top 5 tickers:")
        for t, s in sorted_tickers[:5]:
            lines.append(f"    {t:30s} Sharpe={s:.3f}  n={th.ticker_n_trades.get(t, 0):,}")
        lines.append("  Bottom 5 tickers:")
        for t, s in sorted_tickers[-5:]:
            lines.append(f"    {t:30s} Sharpe={s:.3f}  n={th.ticker_n_trades.get(t, 0):,}")
    lines.append("")

    # 3. Permutation Test
    pt = report.permutation_test
    lines += [
        "--- 3. Permutation Test ---",
        f"  Observed Sharpe:  {pt.observed_sharpe:.3f}",
        f"  Permuted mean:    {np.mean(pt.permuted_sharpes):.3f} +/- {np.std(pt.permuted_sharpes):.3f}",
        f"  p-value:          {pt.p_value:.4f}",
        f"  Permutations:     {pt.n_permutations}",
        f"  Verdict:          {'SIGNIFICANT' if pt.p_value < 0.05 else 'NOT SIGNIFICANT'}",
        "",
    ]

    # 4. Realistic Execution
    em = report.execution_model
    lines += [
        "--- 4. Realistic Execution Model ---",
        f"  Naive Sharpe:          {em.naive_sharpe:.3f}",
        f"  Realistic Sharpe:      {em.realistic_sharpe:.3f}",
        f"  Degradation:           {em.sharpe_degradation_pct:.1f}%",
        f"  Mean slippage:         {em.mean_slippage_pct:.2f}%",
        f"  Trades filtered (liq): {em.trades_filtered_by_liquidity:,} / {em.trades_total:,}",
        "",
    ]

    # 5. Stress Tests
    lines.append("--- 5. Stress Tests ---")
    for s in report.stress_tests.scenario_results:
        lines.append(
            f"  {s['scenario']:25s} Sharpe={s['sharpe']:7.3f}  "
            f"Return=${s['total_return']:12,.2f}  "
            f"Trades={s['n_trades']:,}  WR={s['win_rate']:.1%}"
        )
    lines.append("")

    # 6. Random Baseline
    rb = report.random_baseline
    lines += [
        "--- 6. Random Baseline Comparison ---",
        f"  Model Sharpe:     {rb.model_sharpe:.3f}",
        f"  Random mean:      {rb.random_mean_sharpe:.3f} +/- {rb.random_std_sharpe:.3f}",
        f"  p-value:          {rb.p_value:.4f}",
        f"  Verdict:          {'BEATS RANDOM' if rb.p_value < 0.05 else 'NO BETTER THAN RANDOM'}",
        "",
    ]

    # Verdict
    lines += [
        "=" * 65,
        f"OVERALL VERDICT: {'PASS' if report.overall_pass else 'FAIL'}",
        "=" * 65,
    ]
    if report.failure_reasons:
        lines.append("Failure reasons:")
        for r in report.failure_reasons:
            lines.append(f"  - {r}")
    else:
        lines.append("All validation checks passed.")
    lines.append("")

    return "\n".join(lines)
