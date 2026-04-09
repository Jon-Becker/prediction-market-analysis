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

    # Validation
    validation_n_permutations: int = 100
    validation_n_random_baselines: int = 50
    validation_embargo_multiplier: float = 1.0
    validation_slippage_base_pct: float = 0.5  # base slippage as % of notional
    validation_slippage_size_pct: float = 0.1  # additional slippage per $10 notional
    validation_slippage_size_unit: float = 10.0
    validation_latency_trades: int = 3  # execute N trades later than signal
    validation_liquidity_volume_percentile: float = 20.0  # skip bottom 20% by rolling volume
    validation_liquidity_window: int = 100
    validation_stress_cost_multipliers: list[float] = field(default_factory=lambda: [2.0, 3.0, 5.0])
    validation_stress_max_position_fractions: list[float] = field(default_factory=lambda: [0.5, 0.25])
    validation_sample_size: int = 50_000  # subsample for expensive validation steps

    @property
    def test_fraction(self) -> float:
        return 1.0 - self.train_fraction - self.validation_fraction
