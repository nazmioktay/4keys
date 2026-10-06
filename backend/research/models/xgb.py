"""XGBoost sarmalayıcıları (regresyon, ikili sınıflandırma). NaN yerel; deterministik: tek iş parçacığı, sabit seed."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from .base import BaseModel
from .registry import register_model

_COMMON = dict(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)


def _xgb_params(seed: int, params: dict) -> dict:
    return {**_COMMON, **params, "random_state": seed, "n_jobs": 1, "tree_method": "hist", "verbosity": 0}


@register_model("xgboost_reg")
class XGBoostReg(BaseModel):
    task = "regression"
    handles_nan = True

    def fit(self, X, y, dates, sample_weight=None):
        keep = self._valid_rows(y)
        w = None if sample_weight is None else np.asarray(sample_weight)[keep]
        self._cols = list(X.columns)
        self._m = xgb.XGBRegressor(**_xgb_params(self.seed, self.params)).fit(X[keep], y[keep].astype(float), sample_weight=w)
        return self

    def predict(self, X):
        return self._m.predict(X[self._cols])


@register_model("xgboost_clf")
class XGBoostClf(BaseModel):
    task = "classification"
    handles_nan = True

    def fit(self, X, y, dates, sample_weight=None):
        keep = self._valid_rows(y)
        w = None if sample_weight is None else np.asarray(sample_weight)[keep]
        self._cols = list(X.columns)
        self._m = xgb.XGBClassifier(**_xgb_params(self.seed, self.params)).fit(X[keep], (y[keep] > 0.5).astype(int), sample_weight=w)
        return self

    def predict(self, X):
        proba = self._m.predict_proba(X[self._cols])
        return proba[:, list(self._m.classes_).index(1)] if 1 in self._m.classes_ else np.zeros(len(X))
