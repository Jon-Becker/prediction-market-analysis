"""End-to-end test of the quant pipeline using synthetic data.

Generates realistic fake Kalshi trade + market parquet files,
then runs every pipeline stage to catch bugs before the real 36GB run.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def synthetic_data_dir():
    """Create a temp directory with synthetic parquet files."""
    tmpdir = Path(tempfile.mkdtemp(prefix="quant_test_"))

    trades_dir = tmpdir / "trades"
    markets_dir = tmpdir / "markets"
    trades_dir.mkdir()
    markets_dir.mkdir()

    rng = np.random.default_rng(42)
    tickers = [f"TEST-MKT-{i}" for i in range(20)]
    results = rng.choice(["yes", "no"], size=len(tickers))

    # -- Markets parquet --
    markets_rows = []
    for i, ticker in enumerate(tickers):
        markets_rows.append({
            "ticker": ticker,
            "event_ticker": f"EVENT-{i // 5}",
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
            "created_time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 30))),
            "open_time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 30))),
            "close_time": pd.Timestamp("2024-06-01") + pd.Timedelta(days=int(rng.integers(0, 60))),
            "_fetched_at": pd.Timestamp("2024-08-01"),
        })

    markets_df = pd.DataFrame(markets_rows)
    pq.write_table(pa.Table.from_pandas(markets_df), markets_dir / "markets_0_20.parquet")

    # -- Trades parquet --
    # Generate ~500 trades per ticker = 10,000 total
    all_trades = []
    trade_id_counter = 0

    for _i, ticker in enumerate(tickers):
        n_trades = rng.integers(300, 700)
        # Simulate a price path (mean-reverting around a level)
        base_price = rng.integers(20, 80)
        prices = np.clip(
            base_price + np.cumsum(rng.normal(0, 1.5, size=n_trades)),
            1, 99,
        ).astype(int)

        base_time = pd.Timestamp("2024-01-15")
        for j in range(n_trades):
            yes_price = int(prices[j])
            no_price = 100 - yes_price
            taker_side = rng.choice(["yes", "no"])
            count = int(rng.integers(1, 50))

            all_trades.append({
                "trade_id": f"trade_{trade_id_counter}",
                "ticker": ticker,
                "count": count,
                "yes_price": yes_price,
                "no_price": no_price,
                "taker_side": taker_side,
                "created_time": base_time + pd.Timedelta(seconds=int(j * rng.integers(10, 300))),
                "_fetched_at": pd.Timestamp("2024-08-01"),
            })
            trade_id_counter += 1

    trades_df = pd.DataFrame(all_trades)
    pq.write_table(pa.Table.from_pandas(trades_df), trades_dir / "trades_0_10000.parquet")

    yield tmpdir

    shutil.rmtree(tmpdir, ignore_errors=True)


class TestDataPipeline:
    """Test the DuckDB data loading pipeline."""

    def test_load_trades_with_outcomes(self, synthetic_data_dir):
        from src.quant.config import PipelineConfig
        from src.quant.pipeline import DataPipeline

        config = PipelineConfig(
            trades_dir=synthetic_data_dir / "trades",
            markets_dir=synthetic_data_dir / "markets",
            min_trades_per_ticker=50,
        )
        pipeline = DataPipeline(config)
        try:
            df = pipeline.load_trades_with_outcomes()
            assert len(df) > 0
            assert "taker_price" in df.columns
            assert "taker_won" in df.columns
            assert "taker_notional" in df.columns
            assert df["taker_won"].isin([0, 1]).all()
            assert (df["yes_price"] + df["no_price"] == 100).all()
        finally:
            pipeline.close()

    def test_get_active_tickers(self, synthetic_data_dir):
        from src.quant.config import PipelineConfig
        from src.quant.pipeline import DataPipeline

        config = PipelineConfig(
            trades_dir=synthetic_data_dir / "trades",
            markets_dir=synthetic_data_dir / "markets",
            min_trades_per_ticker=50,
        )
        pipeline = DataPipeline(config)
        try:
            tickers = pipeline.get_active_tickers()
            assert len(tickers) > 0
            assert "ticker" in tickers.columns
            assert "trade_count" in tickers.columns
        finally:
            pipeline.close()

    def test_compute_ticker_features_sql(self, synthetic_data_dir):
        from src.quant.config import PipelineConfig
        from src.quant.pipeline import DataPipeline

        config = PipelineConfig(
            trades_dir=synthetic_data_dir / "trades",
            markets_dir=synthetic_data_dir / "markets",
            min_trades_per_ticker=50,
        )
        pipeline = DataPipeline(config)
        try:
            features = pipeline.compute_ticker_features_sql()
            assert len(features) > 0
            assert "vwap" in features.columns
            assert "yes_volume_share" in features.columns
            assert "net_volume_imbalance" in features.columns
        finally:
            pipeline.close()


class TestFeatureEngine:
    """Test feature engineering."""

    def test_build_trade_features(self, synthetic_data_dir):
        from src.quant.config import PipelineConfig
        from src.quant.features import FeatureEngine
        from src.quant.pipeline import DataPipeline

        config = PipelineConfig(
            trades_dir=synthetic_data_dir / "trades",
            markets_dir=synthetic_data_dir / "markets",
            min_trades_per_ticker=50,
            target_horizon=10,
        )
        pipeline = DataPipeline(config)
        try:
            df = pipeline.load_ticker_trades("TEST-MKT-0")
        finally:
            pipeline.close()

        engine = FeatureEngine(config)
        featured = engine.build_trade_features(df)

        # Check key feature groups exist
        assert "price_change" in featured.columns
        assert "signed_volume" in featured.columns
        assert "volume_imbalance_100" in featured.columns
        assert "sma_50" in featured.columns
        assert "volatility_50" in featured.columns
        assert "hour_sin" in featured.columns
        assert "target_return" in featured.columns
        assert "future_price" in featured.columns

        # Check feature values are reasonable
        assert featured["volume_imbalance_100"].between(-1, 1).all()
        assert featured["vpin_100"].between(0, 2).all()

    def test_get_feature_names(self, synthetic_data_dir):
        from src.quant.config import PipelineConfig
        from src.quant.features import FeatureEngine
        from src.quant.pipeline import DataPipeline

        config = PipelineConfig(
            trades_dir=synthetic_data_dir / "trades",
            markets_dir=synthetic_data_dir / "markets",
            min_trades_per_ticker=50,
        )
        pipeline = DataPipeline(config)
        try:
            df = pipeline.load_ticker_trades("TEST-MKT-0")
        finally:
            pipeline.close()

        engine = FeatureEngine(config)
        featured = engine.build_trade_features(df)
        feature_names = FeatureEngine.get_feature_names(featured)

        assert len(feature_names) > 30
        assert "trade_id" not in feature_names
        assert "ticker" not in feature_names
        assert "target_return" not in feature_names


class TestSignalAnalyzer:
    """Test signal screening and statistical tests."""

    def test_screen_features(self, synthetic_data_dir):
        from src.quant.config import PipelineConfig
        from src.quant.features import FeatureEngine
        from src.quant.pipeline import DataPipeline
        from src.quant.signals import SignalAnalyzer

        config = PipelineConfig(
            trades_dir=synthetic_data_dir / "trades",
            markets_dir=synthetic_data_dir / "markets",
            min_trades_per_ticker=50,
            target_horizon=10,
        )
        pipeline = DataPipeline(config)
        try:
            df = pipeline.load_ticker_trades("TEST-MKT-0")
        finally:
            pipeline.close()

        engine = FeatureEngine(config)
        featured = engine.build_trade_features(df)
        feature_names = FeatureEngine.get_feature_names(featured)

        analyzer = SignalAnalyzer()
        report = analyzer.screen_all_features(featured, feature_names)

        assert len(report) > 0
        assert "ic" in report.columns
        assert "long_short_spread" in report.columns
        assert "monotonicity" in report.columns

    def test_signal_decay(self, synthetic_data_dir):
        from src.quant.config import PipelineConfig
        from src.quant.features import FeatureEngine
        from src.quant.pipeline import DataPipeline
        from src.quant.signals import SignalAnalyzer

        config = PipelineConfig(
            trades_dir=synthetic_data_dir / "trades",
            markets_dir=synthetic_data_dir / "markets",
            min_trades_per_ticker=50,
        )
        pipeline = DataPipeline(config)
        try:
            df = pipeline.load_ticker_trades("TEST-MKT-0")
        finally:
            pipeline.close()

        engine = FeatureEngine(config)
        featured = engine.build_trade_features(df)

        analyzer = SignalAnalyzer()
        decay = analyzer.test_signal_decay(featured, "volume_imbalance_100", horizons=[5, 10, 25])

        assert len(decay) > 0
        assert "horizon" in decay.columns
        assert "ic" in decay.columns


class TestModelTrainer:
    """Test model training."""

    def test_ridge_training(self, synthetic_data_dir):
        from src.quant.config import PipelineConfig
        from src.quant.features import FeatureEngine
        from src.quant.models import ModelTrainer
        from src.quant.pipeline import DataPipeline

        config = PipelineConfig(
            trades_dir=synthetic_data_dir / "trades",
            markets_dir=synthetic_data_dir / "markets",
            min_trades_per_ticker=50,
            target_horizon=10,
        )
        pipeline = DataPipeline(config)
        try:
            df = pipeline.load_ticker_trades("TEST-MKT-0")
        finally:
            pipeline.close()

        engine = FeatureEngine(config)
        featured = engine.build_trade_features(df)
        feature_names = FeatureEngine.get_feature_names(featured)

        trainer = ModelTrainer(feature_names)
        train, val, test = trainer.prepare_splits(featured)

        result = trainer.train_ridge(train, val)
        assert result.model_name == "Ridge"
        assert result.val_mse >= 0
        assert len(result.feature_importance) == len(feature_names)

        # Test prediction
        preds = trainer.predict(result, test)
        assert len(preds) == len(test)

    def test_lightgbm_training(self, synthetic_data_dir):
        from src.quant.config import PipelineConfig
        from src.quant.features import FeatureEngine
        from src.quant.models import ModelTrainer
        from src.quant.pipeline import DataPipeline

        config = PipelineConfig(
            trades_dir=synthetic_data_dir / "trades",
            markets_dir=synthetic_data_dir / "markets",
            min_trades_per_ticker=50,
            target_horizon=10,
        )
        pipeline = DataPipeline(config)
        try:
            df = pipeline.load_ticker_trades("TEST-MKT-0")
        finally:
            pipeline.close()

        engine = FeatureEngine(config)
        featured = engine.build_trade_features(df)
        feature_names = FeatureEngine.get_feature_names(featured)

        trainer = ModelTrainer(feature_names)
        train, val, test = trainer.prepare_splits(featured)

        result = trainer.train_lightgbm(train, val)
        assert result.model_name == "LightGBM"
        assert result.val_mse >= 0

        preds = trainer.predict(result, test)
        assert len(preds) == len(test)


class TestBacktester:
    """Test backtesting framework."""

    def test_backtest_run(self, synthetic_data_dir):
        from src.quant.backtest import Backtester
        from src.quant.config import PipelineConfig
        from src.quant.features import FeatureEngine
        from src.quant.models import ModelTrainer
        from src.quant.pipeline import DataPipeline

        config = PipelineConfig(
            trades_dir=synthetic_data_dir / "trades",
            markets_dir=synthetic_data_dir / "markets",
            min_trades_per_ticker=50,
            target_horizon=10,
        )
        pipeline = DataPipeline(config)
        try:
            df = pipeline.load_ticker_trades("TEST-MKT-0")
        finally:
            pipeline.close()

        engine = FeatureEngine(config)
        featured = engine.build_trade_features(df)
        feature_names = FeatureEngine.get_feature_names(featured)

        trainer = ModelTrainer(feature_names)
        train, val, test = trainer.prepare_splits(featured)
        result = trainer.train_ridge(train, val)
        preds = trainer.predict(result, test)

        backtester = Backtester(transaction_cost_bps=100.0, min_edge_threshold=0.0)
        bt = backtester.run(test, preds)

        assert bt.total_trades >= 0
        assert len(bt.equity_curve) > 0
        assert "net_pnl" in bt.trade_log.columns or bt.total_trades == 0

    def test_empty_backtest(self):
        from src.quant.backtest import Backtester

        backtester = Backtester(min_edge_threshold=9999.0)  # threshold so high nothing trades
        df = pd.DataFrame({
            "yes_price": [50, 60],
            "target_return": [1.0, -1.0],
            "prediction": [0.1, -0.1],
            "created_time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        })
        bt = backtester.run(df, np.array([0.1, -0.1]))
        assert bt.total_trades == 0
        assert bt.sharpe_ratio == 0.0


class TestEdgeVerifier:
    """Test edge verification."""

    def test_verify_with_trades(self):
        from src.quant.backtest import BacktestResult
        from src.quant.evaluation import EdgeVerifier

        rng = np.random.default_rng(42)
        n = 500
        pnls = rng.normal(0.01, 0.5, size=n)

        trade_log = pd.DataFrame({
            "net_pnl": pnls,
            "gross_pnl": pnls + 0.01,
            "cost": np.full(n, 0.01),
            "prediction": rng.normal(0, 1, n),
            "direction": rng.choice([1, -1], n),
            "position_size": np.full(n, 10.0),
            "realized_return": pnls * 10,
            "price": rng.integers(20, 80, n),
        })
        equity = np.cumsum(np.concatenate([[0], pnls]))

        bt = BacktestResult(
            total_return=float(pnls.sum()),
            annualized_return=float(pnls.mean() * 25000),
            sharpe_ratio=float(pnls.mean() / pnls.std() * np.sqrt(25000)),
            sortino_ratio=1.0,
            max_drawdown=10.0,
            win_rate=float((pnls > 0).mean()),
            profit_factor=float(pnls[pnls > 0].sum() / abs(pnls[pnls < 0].sum())),
            total_trades=n,
            avg_trade_return=float(pnls.mean()),
            trade_log=trade_log,
            equity_curve=pd.DataFrame({"trade_number": range(len(equity)), "equity": equity}),
            rolling_sharpe=pd.Series([1.0] * n),
        )

        verifier = EdgeVerifier(n_bootstrap=200)
        result = verifier.verify(bt, n_trials=3)

        assert isinstance(result.is_significant, bool)
        assert 0 <= result.sharpe_pvalue <= 1
        assert 0 <= result.stability_score <= 1
        assert not result.sensitivity_results.empty
