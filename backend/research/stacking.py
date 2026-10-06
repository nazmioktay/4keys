"""Yığınlama (stacking): birden çok modelin OOF tahminlerini birleştiren sade doğrusal (ridge) birleştirici.

İÇ İÇE walk-forward: fold k'nın birleştiricisi YALNIZCA önceki foldların (0..k-1) OOF'larıyla eğitilir; etiketi fold k'nın ilk
kararından sonra tamamlanan satırlar (son `purge_days` gün) atılır (purge). Fold 0 için tahmin YOKTUR (NaN). Nihai test
penceresi ve fold k'nın kendi etiketleri birleştiriciye ASLA girmez (testle kanıtlanır)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .guard import assert_no_final_test


def stack_oof(
    oof: pd.DataFrame, y: pd.Series, fold_test_dates: list[pd.DatetimeIndex], horizon: int, embargo_days: int | None = None,
    alpha: float = 1.0, allow_final_test: bool = False,
) -> tuple[pd.Series, list[dict]]:
    """`oof`: (date, symbol) × model tahminleri. `fold_test_dates`: her foldun test günleri (kronolojik).
    Döner: (stacked tahmin serisi, fold başına {k, n_train, coef, intercept})."""
    assert_no_final_test(oof.index.get_level_values("date"), allow_final_test)
    purge = max(horizon, embargo_days or 0)
    dates = oof.index.get_level_values("date")
    out = pd.Series(np.nan, index=oof.index, name="stacked")
    info = []
    for k in range(1, len(fold_test_dates)):
        first_test = fold_test_dates[k].min()
        prev = np.zeros(len(oof), dtype=bool)
        for j in range(k):
            prev |= dates.isin(fold_test_dates[j])
        usable = prev & (dates <= first_test - pd.Timedelta(days=purge))  # etiketi k'dan önce tamamlanmış
        P, t = oof[usable], y[usable]
        ok = P.notna().all(axis=1).to_numpy() & t.notna().to_numpy()
        if ok.sum() < 30:
            info.append({"k": k, "n_train": int(ok.sum()), "coef": None, "intercept": None})
            continue
        meta = Ridge(alpha=alpha).fit(P[ok], t[ok])
        test = dates.isin(fold_test_dates[k])
        Pk = oof[test]
        valid = Pk.notna().all(axis=1).to_numpy()
        pred = np.full(len(Pk), np.nan)
        pred[valid] = meta.predict(Pk[valid])
        out[test] = pred
        info.append({"k": k, "n_train": int(ok.sum()), "coef": dict(zip(oof.columns, meta.coef_.round(6))), "intercept": float(meta.intercept_)})
    return out, info
