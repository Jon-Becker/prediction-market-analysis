"""Main orchestration script for the quant trading pipeline.

Runs the full pipeline from data loading through edge verification:
1. Load & preprocess trade data via DuckDB
2. Engineer features (price, volume, imbalance, volatility, time)
3. Screen signals for predictive power
4. Train models (Ridge, LightGBM, XGBoost)
5. Backtest with realistic transaction costs
6. Verify edge statistical significance

Usage:
    uv run python -m src.quant.run                    # full pipeline
    uv run python -m src.quant.run --step signals      # only signal screening
    uv run python -m src.quant.run --step backtest     # only backtest
    uv run python -m src.quant.run --ticker PRES-2024-DJT  # single ticker
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from src.quant.backtest import Backtester
from src.quant.config import PipelineConfig
from src.quant.evaluation import EdgeVerifier, format_backtest_report
from src.quant.features import FeatureEngine
from src.quant.models import ModelTrainer
from src.quant.pipeline import DataPipeline
from src.quant.signals import SignalAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig, step: str | None = None, ticker: str | None = None):
    """Execute the full quant pipeline or a specific step."""
    output_dir = Path("output/quant")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = DataPipeline(config)
    t0 = time.time()

    try:
        # ── Step 1: Load Data ──────────────────────────────────────────
        if step in (None, "load", "features", "signals", "train", "backtest"):
            logger.info("=" * 60)
            logger.info("STEP 1: Loading trade data")
            logger.info("=" * 60)

            if ticker:
                df_raw = pipeline.load_ticker_trades(ticker)
                logger.info("Loaded %s trades for ticker %s", f"{len(df_raw):,}", ticker)
            else:
                # For the full dataset, use SQL-computed features first for overview
                ticker_features = pipeline.compute_ticker_features_sql()
                ticker_features.to_csv(output_dir / "ticker_features.csv", index=False)
                logger.info("Saved ticker-level features to ticker_features.csv")

                # Then load trade-level data (may be large — consider sampling)
                active_tickers = pipeline.get_active_tickers()
                logger.info("Found %s active tickers", f"{len(active_tickers):,}")

                # For trade-level analysis, sample top tickers by volume
                top_tickers = active_tickers.nlargest(50, "trade_count")["ticker"].tolist()
                logger.info("Using top %d tickers by volume for trade-level analysis", len(top_tickers))

                # Load trades for selected tickers
                chunks = []
                for t in top_tickers:
                    chunk = pipeline.load_ticker_trades(t)
                    if len(chunk) > 0:
                        chunks.append(chunk)
                df_raw = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                logger.info("Loaded %s trades across %d tickers", f"{len(df_raw):,}", len(chunks))

            if len(df_raw) == 0:
                logger.error("No trades loaded. Check data paths.")
                return

            if step == "load":
                logger.info("Step 'load' complete. Elapsed: %.1fs", time.time() - t0)
                return

        # ── Step 2: Feature Engineering ────────────────────────────────
        if step in (None, "features", "signals", "train", "backtest"):
            logger.info("=" * 60)
            logger.info("STEP 2: Feature engineering")
            logger.info("=" * 60)

            engine = FeatureEngine(config)

            if ticker:
                df = engine.build_trade_features(df_raw)
            else:
                # Build features per ticker then concatenate
                feature_chunks = []
                for t in df_raw["ticker"].unique():
                    chunk = df_raw[df_raw["ticker"] == t].copy()
                    if len(chunk) >= config.min_trades_per_ticker:
                        featured = engine.build_trade_features(chunk)
                        feature_chunks.append(featured)

                df = pd.concat(feature_chunks, ignore_index=True) if feature_chunks else pd.DataFrame()

            feature_cols = FeatureEngine.get_feature_names(df)
            logger.info("Generated %d features, %s rows", len(feature_cols), f"{len(df):,}")

            # Drop rows with NaN targets (from look-ahead and rolling warmup)
            df_clean = df.dropna(subset=["target_return"] + feature_cols[:5])
            logger.info("After NaN cleanup: %s rows", f"{len(df_clean):,}")

            if step == "features":
                df_clean.head(1000).to_csv(output_dir / "features_sample.csv", index=False)
                logger.info("Saved feature sample. Elapsed: %.1fs", time.time() - t0)
                return

        # ── Step 3: Signal Screening ───────────────────────────────────
        if step in (None, "signals", "train", "backtest"):
            logger.info("=" * 60)
            logger.info("STEP 3: Signal screening")
            logger.info("=" * 60)

            analyzer = SignalAnalyzer(significance_level=config.significance_level)

            # Screen all features
            signal_report = analyzer.screen_all_features(df_clean, feature_cols)
            signal_report.to_csv(output_dir / "signal_report.csv", index=False)
            logger.info("Signal report saved (%d features screened)", len(signal_report))

            # Filter to significant signals
            significant = analyzer.find_significant_signals(signal_report)
            significant.to_csv(output_dir / "significant_signals.csv", index=False)
            logger.info("Found %d significant signals", len(significant))

            if len(significant) > 0:
                logger.info("\nTop 10 signals by IC:")
                for _, row in significant.head(10).iterrows():
                    logger.info(
                        "  %-30s IC=%.4f  Spread=%.4f  Mono=%.2f  p=%.1e",
                        row["feature"], row["ic"], row["long_short_spread"],
                        row["monotonicity"], row["ic_pvalue"],
                    )

                # Signal decay for top signal
                top_signal = significant.iloc[0]["feature"]
                decay = analyzer.test_signal_decay(df_clean, top_signal)
                decay.to_csv(output_dir / "signal_decay.csv", index=False)
                logger.info("\nSignal decay for '%s':", top_signal)
                for _, row in decay.iterrows():
                    logger.info("  Horizon=%3d  IC=%.4f  p=%.2e", row["horizon"], row["ic"], row["ic_pvalue"])

                # Use significant features for modeling
                model_features = significant["feature"].tolist()[:30]  # cap at 30
            else:
                logger.warning("No significant signals found. Using top features by abs IC.")
                model_features = signal_report["feature"].tolist()[:20]

            if step == "signals":
                logger.info("Step 'signals' complete. Elapsed: %.1fs", time.time() - t0)
                return

        # ── Step 4: Model Training ─────────────────────────────────────
        if step in (None, "train", "backtest"):
            logger.info("=" * 60)
            logger.info("STEP 4: Model training")
            logger.info("=" * 60)

            # Filter to usable features (present in data)
            usable_features = [f for f in model_features if f in df_clean.columns]
            if len(usable_features) < 3:
                logger.error("Too few usable features (%d). Aborting.", len(usable_features))
                return

            trainer = ModelTrainer(usable_features)
            train, val, test = trainer.prepare_splits(
                df_clean, config.train_fraction, config.validation_fraction
            )

            # Train Ridge (always available)
            logger.info("Training Ridge regression...")
            ridge_result = trainer.train_ridge(train, val)
            logger.info(
                "Ridge — Train R2: %.4f, Val R2: %.4f, Val IC: %.4f",
                ridge_result.train_r2, ridge_result.val_r2, ridge_result.val_ic,
            )

            # Train LightGBM
            best_result = ridge_result
            try:
                logger.info("Training LightGBM...")
                lgb_result = trainer.train_lightgbm(train, val)
                logger.info(
                    "LightGBM — Train R2: %.4f, Val R2: %.4f, Val IC: %.4f",
                    lgb_result.train_r2, lgb_result.val_r2, lgb_result.val_ic,
                )
                if abs(lgb_result.val_ic) > abs(best_result.val_ic):
                    best_result = lgb_result
            except ImportError:
                logger.warning("LightGBM not installed, skipping.")

            # Train XGBoost
            try:
                logger.info("Training XGBoost...")
                xgb_result = trainer.train_xgboost(train, val)
                logger.info(
                    "XGBoost — Train R2: %.4f, Val R2: %.4f, Val IC: %.4f",
                    xgb_result.train_r2, xgb_result.val_r2, xgb_result.val_ic,
                )
                if abs(xgb_result.val_ic) > abs(best_result.val_ic):
                    best_result = xgb_result
            except ImportError:
                logger.warning("XGBoost not installed, skipping.")

            logger.info("\nBest model: %s (Val IC: %.4f)", best_result.model_name, best_result.val_ic)

            # Save feature importance
            best_result.feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
            logger.info("Top 10 features:")
            for _, row in best_result.feature_importance.head(10).iterrows():
                logger.info("  %-30s  importance=%.4f", row["feature"], row["importance"])

            if step == "train":
                logger.info("Step 'train' complete. Elapsed: %.1fs", time.time() - t0)
                return

        # ── Step 5: Backtesting ────────────────────────────────────────
        if step in (None, "backtest"):
            logger.info("=" * 60)
            logger.info("STEP 5: Backtesting")
            logger.info("=" * 60)

            backtester = Backtester(
                transaction_cost_bps=config.transaction_cost_bps,
                max_position_usd=config.max_position_size,
            )

            # Generate predictions on test set
            test_predictions = trainer.predict(best_result, test)

            # Run backtest
            bt_result = backtester.run(test, test_predictions)

            # ── Step 6: Edge Verification ──────────────────────────────
            logger.info("=" * 60)
            logger.info("STEP 6: Edge verification")
            logger.info("=" * 60)

            verifier = EdgeVerifier()
            n_models_tested = 3  # Ridge + LightGBM + XGBoost
            verification = verifier.verify(bt_result, n_trials=n_models_tested)

            # Print report
            report = format_backtest_report(bt_result, verification)
            print("\n" + report)

            # Save outputs
            bt_result.trade_log.to_csv(output_dir / "backtest_trades.csv", index=False)
            bt_result.equity_curve.to_csv(output_dir / "equity_curve.csv", index=False)

            with open(output_dir / "report.txt", "w") as f:
                f.write(report)

            logger.info("All outputs saved to %s", output_dir)

    finally:
        pipeline.close()

    elapsed = time.time() - t0
    logger.info("Pipeline complete. Total elapsed: %.1fs (%.1f min)", elapsed, elapsed / 60)


def main():
    parser = argparse.ArgumentParser(description="Quant trading pipeline for prediction markets")
    parser.add_argument(
        "--step",
        choices=["load", "features", "signals", "train", "backtest"],
        default=None,
        help="Run only up to this step (default: full pipeline)",
    )
    parser.add_argument("--ticker", type=str, default=None, help="Analyze a single ticker")
    parser.add_argument("--trades-dir", type=str, default=None, help="Override trades directory")
    parser.add_argument("--markets-dir", type=str, default=None, help="Override markets directory")
    parser.add_argument("--horizon", type=int, default=50, help="Prediction horizon (trades ahead)")
    parser.add_argument("--min-trades", type=int, default=200, help="Minimum trades per ticker")
    parser.add_argument("--cost-bps", type=float, default=100.0, help="Transaction cost in bps")

    args = parser.parse_args()

    config = PipelineConfig(
        target_horizon=args.horizon,
        min_trades_per_ticker=args.min_trades,
        transaction_cost_bps=args.cost_bps,
    )
    if args.trades_dir:
        config.trades_dir = Path(args.trades_dir)
    if args.markets_dir:
        config.markets_dir = Path(args.markets_dir)

    run_pipeline(config, step=args.step, ticker=args.ticker)


if __name__ == "__main__":
    main()
