"""Model training for alpha prediction.

Supports multiple model types with a unified interface:
- Ridge regression (fast baseline)
- LightGBM (gradient boosting, handles large datasets well)
- XGBoost (alternative gradient boosting)

All models predict future_return (regression) or target_up (classification).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Results from training and evaluating a model."""

    model_name: str
    train_r2: float
    val_r2: float
    val_mse: float
    val_ic: float  # rank correlation on validation set
    val_auc: float | None  # AUC for classification target
    feature_importance: pd.DataFrame
    model: object  # the fitted model
    scaler: StandardScaler | None


class ModelTrainer:
    """Train and evaluate predictive models."""

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols

    def prepare_splits(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data chronologically (no shuffle — time series discipline)."""
        df = df.dropna(subset=self.feature_cols + ["target_return"]).reset_index(drop=True)
        n = len(df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        logger.info("Split sizes — train: %d, val: %d, test: %d", len(train), len(val), len(test))
        return train, val, test

    def train_ridge(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        alpha: float = 1.0,
    ) -> ModelResult:
        """Train a Ridge regression model (fast, interpretable baseline)."""
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[self.feature_cols].values)
        X_val = scaler.transform(val[self.feature_cols].values)
        y_train = train["target_return"].values
        y_val = val["target_return"].values

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        importance = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": np.abs(model.coef_),
        }).sort_values("importance", ascending=False)

        # Classification AUC if binary target available
        val_auc = None
        if "target_up" in val.columns:
            val_auc = _safe_auc(val["target_up"].values, val_pred)

        from scipy.stats import spearmanr
        ic, _ = spearmanr(y_val, val_pred)

        return ModelResult(
            model_name="Ridge",
            train_r2=r2_score(y_train, train_pred),
            val_r2=r2_score(y_val, val_pred),
            val_mse=mean_squared_error(y_val, val_pred),
            val_ic=float(ic),
            val_auc=val_auc,
            feature_importance=importance,
            model=model,
            scaler=scaler,
        )

    def train_lightgbm(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        params: dict | None = None,
    ) -> ModelResult:
        """Train a LightGBM model (best for large datasets with many features)."""
        import lightgbm as lgb

        X_train = train[self.feature_cols].values
        X_val = val[self.feature_cols].values
        y_train = train["target_return"].values
        y_val = val["target_return"].values

        default_params = {
            "objective": "regression",
            "metric": "mse",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "n_jobs": 4,
        }
        if params:
            default_params.update(params)

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_cols)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_cols, reference=dtrain)

        model = lgb.train(
            default_params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        importance = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)

        val_auc = None
        if "target_up" in val.columns:
            val_auc = _safe_auc(val["target_up"].values, val_pred)

        from scipy.stats import spearmanr
        ic, _ = spearmanr(y_val, val_pred)

        return ModelResult(
            model_name="LightGBM",
            train_r2=r2_score(y_train, train_pred),
            val_r2=r2_score(y_val, val_pred),
            val_mse=mean_squared_error(y_val, val_pred),
            val_ic=float(ic),
            val_auc=val_auc,
            feature_importance=importance,
            model=model,
            scaler=None,
        )

    def train_xgboost(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        params: dict | None = None,
    ) -> ModelResult:
        """Train an XGBoost model (alternative to LightGBM)."""
        import xgboost as xgb

        X_train = train[self.feature_cols].values
        X_val = val[self.feature_cols].values
        y_train = train["target_return"].values
        y_val = val["target_return"].values

        default_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": 0.05,
            "max_depth": 8,
            "min_child_weight": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "verbosity": 0,
            "nthread": 4,
        }
        if params:
            default_params.update(params)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_cols)

        model = xgb.train(
            default_params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=100,
        )

        train_pred = model.predict(dtrain)
        val_pred = model.predict(dval)

        importance = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": [model.get_score(importance_type="gain").get(f, 0) for f in self.feature_cols],
        }).sort_values("importance", ascending=False)

        val_auc = None
        if "target_up" in val.columns:
            val_auc = _safe_auc(val["target_up"].values, val_pred)

        from scipy.stats import spearmanr
        ic, _ = spearmanr(y_val, val_pred)

        return ModelResult(
            model_name="XGBoost",
            train_r2=r2_score(y_train, train_pred),
            val_r2=r2_score(y_val, val_pred),
            val_mse=mean_squared_error(y_val, val_pred),
            val_ic=float(ic),
            val_auc=val_auc,
            feature_importance=importance,
            model=model,
            scaler=None,
        )

    def predict(self, result: ModelResult, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions from a trained ModelResult."""
        X = df[self.feature_cols].values

        if result.scaler is not None:
            X = result.scaler.transform(X)

        model = result.model
        if result.model_name == "XGBoost":
            import xgboost as xgb
            X = xgb.DMatrix(X, feature_names=self.feature_cols)

        return model.predict(X)


def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Compute AUC, returning None if it can't be calculated."""
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_pred))
    except ValueError:
        return None
