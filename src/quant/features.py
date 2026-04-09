"""Feature engineering for prediction market trade data.

Generates alpha factors from price, volume, and order flow signals.
Designed to work on per-ticker trade sequences (time-ordered).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.quant.config import PipelineConfig


class FeatureEngine:
    """Builds features from time-ordered trade sequences."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def build_trade_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all features for a time-ordered trade DataFrame.

        Input must have columns: yes_price, no_price, count, taker_side,
        created_time, taker_price, taker_won.

        Returns DataFrame with original columns plus all computed features.
        """
        df = df.copy()
        df = df.sort_values("created_time").reset_index(drop=True)

        df = self._price_features(df)
        df = self._volume_features(df)
        df = self._imbalance_features(df)
        df = self._rolling_features(df)
        df = self._volatility_features(df)
        df = self._time_features(df)
        df = self._target_variable(df)

        return df

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-based features."""
        # Price changes
        df["price_change"] = df["yes_price"].diff()
        df["price_change_abs"] = df["price_change"].abs()
        df["price_return"] = df["yes_price"].pct_change()

        # Distance from extremes
        df["dist_from_50"] = (df["yes_price"] - 50).abs()
        df["dist_from_boundary"] = df[["yes_price", "no_price"]].min(axis=1)

        # Price level buckets (non-linear zones in prediction markets)
        df["price_zone"] = pd.cut(
            df["yes_price"],
            bins=[0, 10, 25, 40, 60, 75, 90, 100],
            labels=[1, 2, 3, 4, 5, 6, 7],
        ).astype(float)

        # Lagged prices
        for lag in [1, 5, 10, 20]:
            df[f"price_lag_{lag}"] = df["yes_price"].shift(lag)
            df[f"price_diff_{lag}"] = df["yes_price"] - df["yes_price"].shift(lag)

        return df

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume and trade size features."""
        df["log_count"] = np.log1p(df["count"])
        df["notional"] = df["count"] * df["taker_price"] / 100.0

        # Relative size vs recent trades
        for w in self.config.rolling_windows:
            roll = df["count"].rolling(w, min_periods=1)
            df[f"count_zscore_{w}"] = (df["count"] - roll.mean()) / roll.std().clip(lower=1e-6)
            df[f"notional_ma_{w}"] = df["notional"].rolling(w, min_periods=1).mean()

        return df

    def _imbalance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Order flow imbalance signals from taker_side and count."""
        # Binary taker direction
        df["taker_is_yes"] = (df["taker_side"] == "yes").astype(int)

        # Signed volume: positive for yes-taker, negative for no-taker
        df["signed_volume"] = df["count"] * np.where(df["taker_is_yes"] == 1, 1, -1)

        for w in self.config.imbalance_windows:
            # Volume imbalance ratio: net yes volume / total volume
            yes_vol = (df["count"] * df["taker_is_yes"]).rolling(w, min_periods=1).sum()
            total_vol = df["count"].rolling(w, min_periods=1).sum()
            df[f"volume_imbalance_{w}"] = (2 * yes_vol / total_vol.clip(lower=1)) - 1  # [-1, 1]

            # Trade count imbalance
            yes_count = df["taker_is_yes"].rolling(w, min_periods=1).sum()
            df[f"trade_imbalance_{w}"] = (2 * yes_count / w) - 1  # [-1, 1]

            # Cumulative signed volume
            df[f"cum_signed_vol_{w}"] = df["signed_volume"].rolling(w, min_periods=1).sum()

            # Tick imbalance (Kyle/Easley-style): signed trade direction
            df[f"tick_imbalance_{w}"] = df["signed_volume"].rolling(w, min_periods=1).mean()

        # VPIN-style: absolute imbalance over volume
        for w in self.config.imbalance_windows:
            abs_imbalance = df["signed_volume"].abs().rolling(w, min_periods=1).sum()
            total_vol = df["count"].rolling(w, min_periods=1).sum()
            df[f"vpin_{w}"] = abs_imbalance / total_vol.clip(lower=1)

        return df

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling window statistics."""
        for w in self.config.rolling_windows:
            roll = df["yes_price"].rolling(w, min_periods=1)

            # Moving averages
            df[f"sma_{w}"] = roll.mean()
            df[f"price_vs_sma_{w}"] = df["yes_price"] - df[f"sma_{w}"]

            # Min/max
            df[f"roll_min_{w}"] = roll.min()
            df[f"roll_max_{w}"] = roll.max()
            df[f"roll_range_{w}"] = df[f"roll_max_{w}"] - df[f"roll_min_{w}"]

            # Position within range (0 = at min, 1 = at max)
            range_val = df[f"roll_range_{w}"].clip(lower=1)
            df[f"roll_position_{w}"] = (df["yes_price"] - df[f"roll_min_{w}"]) / range_val

        # EMA features
        for span in self.config.ema_spans:
            df[f"ema_{span}"] = df["yes_price"].ewm(span=span, min_periods=1).mean()
            df[f"price_vs_ema_{span}"] = df["yes_price"] - df[f"ema_{span}"]

        # VWAP (volume-weighted average price)
        for w in self.config.rolling_windows:
            pv = (df["yes_price"] * df["count"]).rolling(w, min_periods=1).sum()
            vol = df["count"].rolling(w, min_periods=1).sum()
            df[f"vwap_{w}"] = pv / vol.clip(lower=1)
            df[f"price_vs_vwap_{w}"] = df["yes_price"] - df[f"vwap_{w}"]

        return df

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility and dispersion features."""
        for w in self.config.rolling_windows:
            # Realized volatility (std of price changes)
            df[f"volatility_{w}"] = df["price_change"].rolling(w, min_periods=2).std()

            # Mean absolute deviation
            roll = df["yes_price"].rolling(w, min_periods=2)
            df[f"mad_{w}"] = roll.apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)

            # Parkinson volatility (high-low range estimator)
            roll_range = df[f"roll_range_{w}"] if f"roll_range_{w}" in df.columns else (
                df["yes_price"].rolling(w, min_periods=1).max()
                - df["yes_price"].rolling(w, min_periods=1).min()
            )
            df[f"parkinson_vol_{w}"] = roll_range / (2 * np.sqrt(np.log(2)))

        return df

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Temporal features from trade timestamps."""
        if "created_time" not in df.columns:
            return df

        ts = pd.to_datetime(df["created_time"])

        # Time of day features
        df["hour"] = ts.dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)

        # Inter-trade duration
        df["time_delta_seconds"] = ts.diff().dt.total_seconds()
        df["log_time_delta"] = np.log1p(df["time_delta_seconds"].clip(lower=0))

        # Trade velocity (trades per minute, rolling)
        for w in [20, 100]:
            time_span = ts.diff().dt.total_seconds().rolling(w, min_periods=1).sum()
            df[f"trade_velocity_{w}"] = w / time_span.clip(lower=1) * 60  # trades per minute

        return df

    def _target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the prediction target: future price movement."""
        horizon = self.config.target_horizon

        # Future price (N trades ahead)
        df["future_price"] = df["yes_price"].shift(-horizon)
        df["future_return"] = df["future_price"] - df["yes_price"]

        # Binary target: price goes up
        df["target_up"] = (df["future_return"] > 0).astype(int)

        # Regression target: excess return for a yes-taker
        # If we buy YES at yes_price, we get $1 if outcome is yes, else $0
        # But we don't know the outcome during trading -- so we use future price as proxy
        df["target_return"] = df["future_return"] / df["yes_price"].clip(lower=1) * 100

        return df

    @staticmethod
    def get_feature_names(df: pd.DataFrame) -> list[str]:
        """Extract feature column names from a featurized DataFrame."""
        exclude = {
            "trade_id", "ticker", "count", "yes_price", "no_price", "taker_side",
            "created_time", "result", "taker_price", "taker_won", "taker_notional",
            "_fetched_at", "future_price", "future_return", "target_up",
            "target_return", "notional",
        }
        return [c for c in df.columns if c not in exclude and not c.startswith("target_")]
