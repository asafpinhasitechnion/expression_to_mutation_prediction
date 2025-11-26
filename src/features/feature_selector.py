from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.linear_model import Lasso


class FeatureSelector:
    """Utility wrapper around a few common feature-selection strategies."""

    def __init__(self, config_path: str | Path | None = None):
        base_dir = Path(__file__).resolve().parents[2]
        config_path = Path(config_path or (base_dir / "config/config.yaml"))

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        features_cfg = self.config.get("features", {})
        self.n_features = features_cfg.get("n_features", 1000)
        self.method = features_cfg.get("method", "variance")

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame | np.ndarray | None = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select features based on the configured method."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Feature selection requires a pandas DataFrame with column names.")

        if self.method == "variance":
            return self._variance_selection(X)

        if self.method == "mutual_info":
            y_1d = self._ensure_single_target(y, method="mutual_info")
            return self._mutual_info_selection(X, y_1d)

        if self.method == "lasso":
            y_1d = self._ensure_single_target(y, method="lasso")
            return self._lasso_selection(X, y_1d)

        raise ValueError(f"Unknown feature selection method: {self.method}")

    def _variance_selection(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Select features based on a variance threshold."""
        threshold = self.config.get("preprocessing", {}).get("min_variance", 0.0)
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        selected_features = X.columns[selector.get_support()].tolist()
        return X[selected_features].copy(), selected_features

    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        selector = SelectKBest(score_func=mutual_info_classif, k=min(self.n_features, X.shape[1]))
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        return X[selected_features].copy(), selected_features

    def _lasso_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        lasso = Lasso(alpha=0.01, max_iter=5000)
        lasso.fit(X, y)
        mask = lasso.coef_ != 0
        selected_features = X.columns[mask].tolist()
        if not selected_features:
            raise ValueError("Lasso did not keep any features; try lowering alpha.")
        return X[selected_features].copy(), selected_features

    @staticmethod
    def _ensure_single_target(
        y: pd.Series | pd.DataFrame | np.ndarray | None,
        method: str,
    ) -> pd.Series:
        if y is None:
            raise ValueError(f"{method} feature selection requires targets (y).")

        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(f"{method} feature selection only supports a single target column.")
            y = y.iloc[:, 0]

        if isinstance(y, np.ndarray):
            if y.ndim > 1 and y.shape[1] != 1:
                raise ValueError(f"{method} feature selection only supports a single target column.")
            y = pd.Series(y.ravel())

        if not isinstance(y, pd.Series):
            raise TypeError("Targets must be a pandas Series, DataFrame, or numpy array.")

        return y