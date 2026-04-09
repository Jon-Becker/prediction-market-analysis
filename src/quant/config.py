"""Configuration for the quant trading pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Central configuration for the entire quant pipeline."""

    # Data paths
    trades_dir: Path = field(default_factory=lambda: Path("data/kalshi/trades"))
    markets_dir: Path = field(default_factory=lambda: Path("data/kalshi/markets"))

    # Feature engineering
    rolling_windows: list[int] = field(default_factory=lambda: [10, 50, 200, 1000])
    ema_spans: list[int] = field(default_factory=lambda: [10, 50, 200])
    imbalance_windows: list[int] = field(default_factory=lambda: [20, 100, 500])

    # Model training
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    # test_fraction is implicitly 1 - train - validation
    target_horizon: int = 50  # predict price move N trades ahead
    min_trades_per_ticker: int = 200

    # Backtesting
    backtest_rolling_window: int = 50_000
    backtest_step_size: int = 10_000
    transaction_cost_bps: float = 100.0  # 1% round-trip (prediction markets have wide spreads)
    max_position_size: float = 100.0  # max dollars per position

    # Statistical thresholds
    significance_level: float = 0.05
    min_sharpe_ratio: float = 0.5
    min_observations: int = 1000

    @property
    def test_fraction(self) -> float:
        return 1.0 - self.train_fraction - self.validation_fraction
