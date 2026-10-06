"""Model eklentisi arayüzü.

    fit(X, y, dates, sample_weight=None)   X: (date, symbol) MultiIndex'li DataFrame (özellikler + `__missing` bayrakları)
    predict(X) -> np.ndarray               regresyon: değer; sınıflandırma (ikili): P(y=1); sıralama: skor
    get_params() -> dict                   sabit seed dahil

NaN: `handles_nan=False` modellerde EĞİTİM katmanının medyanıyla doldurulur (`_Imputer`; medyan YALNIZCA eğitimden, aynı
dönüşüm tahminde uygulanır — eğitim/canlı tutarlılığı). Ağaç modelleri (`handles_nan=True`) NaN'ı yerel yönetir.
Her model sabit `seed` ile deterministik olmalıdır (tekrarlanabilirlik kontrolü bunu doğrular)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import pandas as pd


class _Imputer:
    def __init__(self) -> None:
        self.median_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> _Imputer:
        self.median_ = X.median(numeric_only=True).fillna(0.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.fillna(self.median_)


class BaseModel(ABC):
    name: ClassVar[str] = ""
    task: ClassVar[str] = "regression"  # "regression" | "classification" | "rank"
    handles_nan: ClassVar[bool] = False
    supports_sample_weight: ClassVar[bool] = True

    def __init__(self, seed: int = 0, **params) -> None:
        self.seed = int(seed)
        self.params = params
        self._imputer = _Imputer()

    # ---- arayüz -----------------------------------------------------------------
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series | pd.Index, sample_weight: np.ndarray | None = None) -> BaseModel:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def get_params(self) -> dict:
        return {"name": self.name, "task": self.task, "seed": self.seed, **self.params}

    # ---- yardımcılar ------------------------------------------------------------
    def _prep_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.handles_nan:
            return X
        return self._imputer.fit(X).transform(X)

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        return X if self.handles_nan else self._imputer.transform(X)

    @staticmethod
    def _valid_rows(y: pd.Series) -> np.ndarray:
        return y.notna().to_numpy()
