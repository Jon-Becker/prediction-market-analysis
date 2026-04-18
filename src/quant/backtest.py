"""Backtesting framework for prediction market trading strategies.

Implements rolling-window walk-forward backtesting with realistic
transaction costs and position sizing for prediction markets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    trade_log: pd.DataFrame
    equity_curve: pd.DataFrame
    rolling_sharpe: pd.Series


class Backtester:
    """Walk-forward backtester for prediction market strategies."""

    def __init__(
        self,
        transaction_cost_bps: float = 100.0,
        max_position_usd: float = 100.0,
        min_edge_threshold: float = 0.5,
    ):
        self.transaction_cost = transaction_cost_bps / 10_000
        self.max_position_usd = max_position_usd
        self.min_edge = min_edge_threshold  # minimum predicted return to trade

    def run(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        target_col: str = "target_return",
    ) -> BacktestResult:
        """Run backtest on predictions vs realized returns.

        Args:
            df: DataFrame with at least yes_price, target_return, created_time.
            predictions: Model predictions aligned with df rows.
            target_col: Column with realized returns.

        Returns:
            BacktestResult with performance metrics.
        """
        df = df.copy()
        df["prediction"] = predictions
        df = df.dropna(subset=[target_col, "prediction"]).reset_index(drop=True)

        trades = []
        equity = [0.0]

        for i in range(len(df)):
            pred = df.loc[i, "prediction"]
            price = df.loc[i, "yes_price"]
            realized = df.loc[i, target_col]

            # Only trade if predicted edge exceeds threshold + costs
            cost_pct = self.transaction_cost * 100  # in same units as prediction
            net_pred = abs(pred) - cost_pct

            if net_pred <= self.min_edge:
                continue

            # Position sizing: proportional to conviction, capped
            position_size = min(
                self.max_position_usd,
                self.max_position_usd * (net_pred / 10.0),  # scale by edge
            )

            # Direction: if prediction > 0, go long (buy YES); if < 0, go short (buy NO)
            direction = 1 if pred > 0 else -1

            # PnL: direction * realized_return * position_size / 100 - costs
            pnl = direction * realized * position_size / 100.0
            cost = position_size * self.transaction_cost
            net_pnl = pnl - cost

            trades.append({
                "index": i,
                "time": df.loc[i, "created_time"] if "created_time" in df.columns else i,
                "price": price,
                "prediction": pred,
                "direction": direction,
                "position_size": position_size,
                "realized_return": realized,
                "gross_pnl": pnl,
                "cost": cost,
                "net_pnl": net_pnl,
            })

            equity.append(equity[-1] + net_pnl)

        if not trades:
            return self._empty_result()

        trade_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame({
            "trade_number": range(len(equity)),
            "equity": equity,
        })

        return self._compute_metrics(trade_df, equity_df)

    def run_rolling(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        trainer_cls,
        model_method: str = "train_lightgbm",
        window_size: int = 50_000,
        step_size: int = 10_000,
    ) -> BacktestResult:
        """Walk-forward rolling window backtest with model retraining.

        Trains a new model on each window, then predicts the next step_size trades.
        This is the most realistic backtest as it avoids look-ahead bias.
        """
        all_trades = []
        equity = [0.0]
        n = len(df)

        for start in range(0, n - window_size - step_size, step_size):
            train_end = start + window_size
            test_end = min(train_end + step_size, n)

            train_slice = df.iloc[start:train_end]
            test_slice = df.iloc[train_end:test_end]

            # Skip if insufficient data
            valid_train = train_slice.dropna(subset=feature_cols + ["target_return"])
            valid_test = test_slice.dropna(subset=feature_cols + ["target_return"])
            if len(valid_train) < 1000 or len(valid_test) < 100:
                continue

            # Train model
            trainer = trainer_cls(feature_cols)
            train_split, val_split, _ = trainer.prepare_splits(
                valid_train, train_frac=0.8, val_frac=0.2
            )

            train_fn = getattr(trainer, model_method)
            result = train_fn(train_split, val_split)

            # Predict on test window
            predictions = trainer.predict(result, valid_test)

            # Execute trades
            for i in range(len(valid_test)):
                pred = predictions[i]
                price = valid_test.iloc[i]["yes_price"]
                realized = valid_test.iloc[i]["target_return"]

                cost_pct = self.transaction_cost * 100
                net_pred = abs(pred) - cost_pct

                if net_pred <= self.min_edge:
                    continue

                position_size = min(
                    self.max_position_usd,
                    self.max_position_usd * (net_pred / 10.0),
                )
                direction = 1 if pred > 0 else -1
                pnl = direction * realized * position_size / 100.0
                cost = position_size * self.transaction_cost
                net_pnl = pnl - cost

                all_trades.append({
                    "window_start": start,
                    "price": price,
                    "prediction": pred,
                    "direction": direction,
                    "position_size": position_size,
                    "realized_return": realized,
                    "gross_pnl": pnl,
                    "cost": cost,
                    "net_pnl": net_pnl,
                })
                equity.append(equity[-1] + net_pnl)

            logger.info(
                "Window %d-%d: %d trades, equity: $%.2f",
                start, train_end, len(all_trades), equity[-1],
            )

        if not all_trades:
            return self._empty_result()

        trade_df = pd.DataFrame(all_trades)
        equity_df = pd.DataFrame({
            "trade_number": range(len(equity)),
            "equity": equity,
        })

        return self._compute_metrics(trade_df, equity_df)

    def _compute_metrics(
        self, trade_df: pd.DataFrame, equity_df: pd.DataFrame
    ) -> BacktestResult:
        """Compute performance metrics from trade log."""
        pnls = trade_df["net_pnl"].values
        total_return = pnls.sum()
        n_trades = len(pnls)

        # Win rate and profit factor
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0
        profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")

        # Drawdown from equity curve
        equity = equity_df["equity"].values
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Sharpe and Sortino (annualized assuming ~250 trading days, ~100 trades/day)
        mean_pnl = pnls.mean()
        std_pnl = pnls.std() if len(pnls) > 1 else 1e-6
        trades_per_year = 250 * 100  # rough estimate
        sharpe = (mean_pnl / max(std_pnl, 1e-6)) * np.sqrt(trades_per_year)

        downside = pnls[pnls < 0]
        downside_std = downside.std() if len(downside) > 1 else 1e-6
        sortino = (mean_pnl / max(downside_std, 1e-6)) * np.sqrt(trades_per_year)

        # Annualized return (rough)
        annualized_return = mean_pnl * trades_per_year

        # Rolling Sharpe (over 1000-trade windows)
        rolling_window = min(1000, len(pnls) // 3)
        if rolling_window > 10:
            rolling_mean = pd.Series(pnls).rolling(rolling_window).mean()
            rolling_std = pd.Series(pnls).rolling(rolling_window).std()
            rolling_sharpe = (rolling_mean / rolling_std.clip(lower=1e-6)) * np.sqrt(trades_per_year)
        else:
            rolling_sharpe = pd.Series([sharpe] * len(pnls))

        return BacktestResult(
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=float(max_drawdown),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            total_trades=n_trades,
            avg_trade_return=float(mean_pnl),
            trade_log=trade_df,
            equity_curve=equity_df,
            rolling_sharpe=rolling_sharpe,
        )

    def _empty_result(self) -> BacktestResult:
        """Return an empty result when no trades were generated."""
        return BacktestResult(
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            avg_trade_return=0.0,
            trade_log=pd.DataFrame(),
            equity_curve=pd.DataFrame({"trade_number": [0], "equity": [0.0]}),
            rolling_sharpe=pd.Series([0.0]),
        )
