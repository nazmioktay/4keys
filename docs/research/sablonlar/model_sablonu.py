"""MODEL ŞABLONU — yeni bir model eklemek için kopyalayın:

    cp docs/research/sablonlar/model_sablonu.py backend/research/models/benim_modelim.py

Sonra: (1) sınıfı doldurun, (2) `backend/research/models/registry.py::_MODULES` listesine "benim_modelim" ekleyin,
(3) `backend/tests/research/test_models.py` kalıbıyla test yazın (kayıt, deterministiklik, NaN, sample_weight).
Kurallar: docs/research/EKLENTI_REHBERI.md. Bu dosya paketin parçası DEĞİLDİR (içe aktarılmaz)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.models.base import BaseModel
from research.models.registry import register_model


@register_model("benim_modelim")   # config'te `models: [benim_modelim]`
class BenimModelim(BaseModel):
    task = "regression"            # "regression" | "classification" (ikili, y∈{0,1}; predict -> P(1)) | "rank" (grup = tarih)
    handles_nan = False            # False: eğitim medyanıyla doldurulur (`self._prep_fit/_prep`); True: model NaN'ı kendisi yönetir
    supports_sample_weight = True  # desteklemiyorsa False (belgeleyin; runner uyarmaz, ağırlık yok sayılır)

    def __init__(self, seed: int = 0, **params):
        super().__init__(seed, **params)   # seed ZORUNLU: aynı seed -> aynı sonuç (tekrarlanabilirlik kontrolü bunu sınar)

    def fit(self, X: pd.DataFrame, y: pd.Series, dates, sample_weight=None):
        """X: (date, symbol) MultiIndex'li; y NaN satırlar eğitimden atılmalı (`self._valid_rows(y)`).
        Rastgelelik kullanıyorsanız `np.random.default_rng(self.seed)`; global/tohumsuz RNG YASAK."""
        keep = self._valid_rows(y)
        Xp = self._prep_fit(X)[keep]
        self._cols = list(Xp.columns)
        # ... modeli eğitin ...
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Regresyon: değer; sınıflandırma: P(y=1); sıralama: skor. Tahmin edilemeyen satır için NaN döndürün (uydurmayın)."""
        Xp = self._prep(X)[self._cols]
        return np.zeros(len(Xp))
