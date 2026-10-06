"""LightGBM sarmalayıcıları: regresyon, ikili sınıflandırma ve LambdaRank (kesitsel sıralama). NaN yerel yönetilir.
Deterministik: tek iş parçacığı, sabit seed, `deterministic=True`."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import BaseModel
from .registry import register_model

_COMMON = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8, reg_lambda=1.0)


def _lgb_params(seed: int, params: dict) -> dict:
    p = {**_COMMON, **params}
    return {**p, "random_state": seed, "n_jobs": 1, "verbose": -1, "deterministic": True, "force_row_wise": True}


@register_model("lightgbm_reg")
class LightGBMReg(BaseModel):
    task = "regression"
    handles_nan = True

    def fit(self, X, y, dates, sample_weight=None):
        keep = self._valid_rows(y)
        w = None if sample_weight is None else np.asarray(sample_weight)[keep]
        self._cols = list(X.columns)
        self._m = lgb.LGBMRegressor(**_lgb_params(self.seed, self.params)).fit(X[keep], y[keep].astype(float), sample_weight=w)
        return self

    def predict(self, X):
        return self._m.predict(X[self._cols])


@register_model("lightgbm_clf")
class LightGBMClf(BaseModel):
    task = "classification"
    handles_nan = True

    def fit(self, X, y, dates, sample_weight=None):
        keep = self._valid_rows(y)
        w = None if sample_weight is None else np.asarray(sample_weight)[keep]
        self._cols = list(X.columns)
        self._m = lgb.LGBMClassifier(**_lgb_params(self.seed, self.params)).fit(X[keep], (y[keep] > 0.5).astype(int), sample_weight=w)
        return self

    def predict(self, X):
        proba = self._m.predict_proba(X[self._cols])
        return proba[:, list(self._m.classes_).index(1)] if 1 in self._m.classes_ else np.zeros(len(X))


@register_model("lightgbm_rank")
class LightGBMRank(BaseModel):
    """LambdaRank: grup = tarih (her tarihte evren içi sıralama), hedef 0..1 sıra yüzdesi `n_bins` tamsayı kovaya bölünür."""

    task = "rank"
    handles_nan = True

    def __init__(self, seed: int = 0, n_bins: int = 5, **params):
        super().__init__(seed, n_bins=n_bins, **params)

    def fit(self, X, y, dates, sample_weight=None):
        n_bins = self.params["n_bins"]
        keep = self._valid_rows(y)
        Xk, yk = X[keep], y[keep].astype(float)
        dk = pd.Series(np.asarray(dates)[keep], index=Xk.index)
        order = np.argsort(dk.to_numpy(), kind="stable")  # grupların bitişik olması şart
        Xk, yk, dk = Xk.iloc[order], yk.iloc[order], dk.iloc[order]
        labels = np.minimum((yk.to_numpy() * n_bins).astype(int), n_bins - 1)
        group = dk.groupby(dk.to_numpy(), sort=False).size().to_numpy()
        p = _lgb_params(self.seed, {k: v for k, v in self.params.items() if k != "n_bins"})
        self._cols = list(X.columns)
        self._m = lgb.LGBMRanker(objective="lambdarank", label_gain=list(range(n_bins)), **p)
        self._m.fit(Xk, labels, group=group)
        return self

    def predict(self, X):
        return self._m.predict(X[self._cols])
