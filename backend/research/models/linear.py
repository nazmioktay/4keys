"""Doğrusal kıyaslar (ZORUNLU): ridge (regresyon) ve lojistik regresyon (ikili sınıflandırma). Eğitim medyanıyla
doldurma + eğitim ortalama/std'siyle standardizasyon."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .base import BaseModel
from .registry import register_model


class _LinearBase(BaseModel):
    def _fit_xy(self, X, y):
        keep = self._valid_rows(y)
        Xp = self._prep_fit(X)
        self._cols = list(Xp.columns)
        self._scaler = StandardScaler().fit(Xp[keep])
        return self._scaler.transform(Xp[keep]), y[keep].to_numpy(dtype=float), keep

    def _transform(self, X):
        return self._scaler.transform(self._prep(X)[self._cols])


@register_model("ridge")
class RidgeModel(_LinearBase):
    task = "regression"

    def __init__(self, seed: int = 0, alpha: float = 10.0, **params):
        super().__init__(seed, alpha=alpha, **params)

    def fit(self, X, y, dates, sample_weight=None):
        Xs, ys, keep = self._fit_xy(X, y)
        w = None if sample_weight is None else np.asarray(sample_weight)[keep]
        self._m = Ridge(alpha=self.params["alpha"]).fit(Xs, ys, sample_weight=w)
        return self

    def predict(self, X):
        return self._m.predict(self._transform(X))


@register_model("logistic")
class LogisticModel(_LinearBase):
    task = "classification"

    def __init__(self, seed: int = 0, C: float = 0.1, max_iter: int = 500, **params):
        super().__init__(seed, C=C, max_iter=max_iter, **params)

    def fit(self, X, y, dates, sample_weight=None):
        Xs, ys, keep = self._fit_xy(X, y)
        w = None if sample_weight is None else np.asarray(sample_weight)[keep]
        self._m = LogisticRegression(C=self.params["C"], max_iter=self.params["max_iter"], random_state=self.seed)
        self._m.fit(Xs, (ys > 0.5).astype(int), sample_weight=w)
        return self

    def predict(self, X):
        proba = self._m.predict_proba(self._transform(X))
        return proba[:, list(self._m.classes_).index(1)] if 1 in self._m.classes_ else np.zeros(len(X))
