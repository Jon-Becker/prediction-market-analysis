"""Tests for the real-world validation module."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.quant.config import PipelineConfig
from src.quant.features import FeatureEngine
from src.quant.pipeline import DataPipeline
from src.quant.validation import RealWorldValidator


@pytest.fixture
def synthetic_data_dir():
    """Create temp directory with synthetic parquet files (multiple tickers)."""
    tmpdir = Path(tempfile.mkdtemp(prefix="val_test_"))
    trades_dir = tmpdir / "trades"
    markets_dir = tmpdir / "markets"
    trades_dir.mkdir()
    markets_dir.mkdir()

    rng = np.random.default_rng(42)
    tickers = [f"TEST-MKT-{i}" for i in range(10)]
    results = rng.choice(["yes", "no"], size=len(tickers))

    # Markets
    markets_rows = []
    for i, ticker in enumerate(tickers):
        markets_rows.append({
            "ticker": ticker,
            "event_ticker": f"EVENT-{i // 3}",
            "market_type": "binary",
            "title": f"Test Market {i}",
            "yes_sub_title": "Yes",
            "no_sub_title": "No",
            "status": "finalized",
            "yes_bid": None,
            "yes_ask": None,
            "no_bid": None,
            "no_ask": None,
            "last_price": rng.integers(10, 90),
            "volume": rng.integers(500, 10000),
            "volume_24h": rng.integers(10, 500),
            "open_interest": rng.integers(100, 2000),
            "result": results[i],
            "created_time": pd.Timestamp("2024-01-01"),
            "open_time": pd.Timestamp("2024-01-01"),
            "close_time": pd.Timestamp("2024-06-01"),
            "_fetched_at": pd.Timestamp("2024-08-01"),
        })
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(markets_rows)), markets_dir / "markets_0_10.parquet")

    # Trades: 800 per ticker = 8000 total
    all_trades = []
    trade_id = 0
    for _i, ticker in enumerate(tickers):
        n_trades = 800
        base_price = rng.integers(25, 75)
        prices = np.clip(base_price + np.cumsum(rng.normal(0, 1.5, size=n_trades)), 1, 99).astype(int)
        base_time = pd.Timestamp("2024-01-15")

        for j in range(n_trades):
            yes_price = int(prices[j])
            all_trades.append({
                "trade_id": f"trade_{trade_id}",
                "ticker": ticker,
                "count": int(rng.integers(1, 30)),
                "yes_price": yes_price,
                "no_price": 100 - yes_price,
                "taker_side": rng.choice(["yes", "no"]),
                "created_time": base_time + pd.Timedelta(seconds=int(j * rng.integers(10, 120))),
                "_fetched_at": pd.Timestamp("2024-08-01"),
            })
            trade_id += 1

    pq.write_table(pa.Table.from_pandas(pd.DataFrame(all_trades)), trades_dir / "trades_0_8000.parquet")

    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def featurized_df(synthetic_data_dir):
    """Load and featurize synthetic data for validation tests."""
    config = PipelineConfig(
        trades_dir=synthetic_data_dir / "trades",
        markets_dir=synthetic_data_dir / "markets",
        min_trades_per_ticker=50,
        target_horizon=10,
        rolling_windows=[10, 50],
        ema_spans=[10, 50],
        imbalance_windows=[20, 50],
    )
    pipeline = DataPipeline(config)
    try:
        df_raw = pipeline.load_trades_with_outcomes()
    finally:
        pipeline.close()

    engine = FeatureEngine(config)
    chunks = []
    for t in df_raw["ticker"].unique():
        chunk = df_raw[df_raw["ticker"] == t].copy()
        if len(chunk) >= 50:
            chunks.append(engine.build_trade_features(chunk))
    df = pd.concat(chunks, ignore_index=True)
    feature_cols = FeatureEngine.get_feature_names(df)

    return df, feature_cols, config


class TestPurgedWalkForwardCV:
    def test_produces_folds(self, featurized_df):
        df, feature_cols, config = featurized_df
        validator = RealWorldValidator(config)
        result = validator.purged_walk_forward_cv(df, feature_cols, n_folds=3)

        assert result.n_folds > 0
        assert len(result.fold_sharpes) == result.n_folds
        assert result.embargo_size > 0

    def test_embargo_prevents_overlap(self, featurized_df):
        df, feature_cols, config = featurized_df
        validator = RealWorldValidator(config)
        result = validator.purged_walk_forward_cv(df, feature_cols, n_folds=3)

        # Embargo should be at least max(rolling_windows)
        assert result.embargo_size >= max(config.rolling_windows)


class TestTickerHoldoutCV:
    def test_tests_multiple_tickers(self, featurized_df):
        df, feature_cols, config = featurized_df
        validator = RealWorldValidator(config)
        result = validator.ticker_holdout_cv(df, feature_cols)

        assert result.n_tickers > 0
        assert len(result.ticker_sharpes) == result.n_tickers

    def test_single_ticker_handled(self, featurized_df):
        df, feature_cols, config = featurized_df
        # Filter to single ticker
        single = df[df["ticker"] == df["ticker"].unique()[0]].copy()
        validator = RealWorldValidator(config)
        result = validator.ticker_holdout_cv(single, feature_cols)

        # Should return empty result (not crash)
        assert result.n_tickers == 0


class TestPermutationTest:
    def test_produces_pvalue(self, featurized_df):
        df, feature_cols, config = featurized_df
        # Use small sample and few permutations for speed
        sample = df.head(2000)
        validator = RealWorldValidator(config)
        result = validator.permutation_test(sample, feature_cols, n_permutations=5)

        assert 0 <= result.p_value <= 1
        assert len(result.permuted_sharpes) == 5
        assert result.n_permutations == 5


class TestRealisticExecution:
    def test_degrades_or_matches_naive(self, featurized_df):
        df, feature_cols, config = featurized_df

        # Train a model to get predictions
        from src.quant.models import ModelTrainer

        df_clean = df.dropna(subset=feature_cols + ["target_return"])
        trainer = ModelTrainer(feature_cols)
        train, val, test = trainer.prepare_splits(df_clean)
        model = trainer.train_ridge(train, val)
        preds = trainer.predict(model, test)

        validator = RealWorldValidator(config)
        result = validator.realistic_execution_backtest(test, preds)

        # Realistic should be <= naive (adding friction only hurts)
        assert result.realistic_sharpe <= result.naive_sharpe + 0.01  # small tolerance
        assert result.mean_slippage_pct > 0

    def test_liquidity_filter_counts(self, featurized_df):
        df, feature_cols, config = featurized_df
        from src.quant.models import ModelTrainer

        df_clean = df.dropna(subset=feature_cols + ["target_return"])
        trainer = ModelTrainer(feature_cols)
        train, val, test = trainer.prepare_splits(df_clean)
        model = trainer.train_ridge(train, val)
        preds = trainer.predict(model, test)

        validator = RealWorldValidator(config)
        result = validator.realistic_execution_backtest(test, preds)

        assert result.trades_filtered_by_liquidity >= 0
        assert result.trades_total > 0


class TestStressTests:
    def test_all_scenarios_run(self, featurized_df):
        df, feature_cols, config = featurized_df
        from src.quant.models import ModelTrainer

        df_clean = df.dropna(subset=feature_cols + ["target_return"])
        trainer = ModelTrainer(feature_cols)
        train, val, test = trainer.prepare_splits(df_clean)
        model = trainer.train_ridge(train, val)
        preds = trainer.predict(model, test)

        validator = RealWorldValidator(config)
        result = validator.stress_test(test, preds)

        # Should have cost scenarios + position scenarios (+ possibly liquidity)
        assert len(result.scenario_results) >= len(config.validation_stress_cost_multipliers)
        for s in result.scenario_results:
            assert "scenario" in s
            assert "sharpe" in s
            assert isinstance(s["sharpe"], float)


class TestRandomBaseline:
    def test_produces_valid_pvalue(self, featurized_df):
        df, feature_cols, config = featurized_df
        from src.quant.models import ModelTrainer

        df_clean = df.dropna(subset=feature_cols + ["target_return"])
        trainer = ModelTrainer(feature_cols)
        train, val, test = trainer.prepare_splits(df_clean)
        model = trainer.train_ridge(train, val)
        preds = trainer.predict(model, test)

        validator = RealWorldValidator(config)
        result = validator.random_baseline_comparison(test, preds, n_baselines=10)

        assert 0 <= result.p_value <= 1
        assert len(result.random_sharpes) == 10
        assert result.n_baselines == 10
