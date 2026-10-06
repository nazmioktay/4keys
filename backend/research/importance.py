"""Kaynak grubu bazında permütasyon önemi (OOF test katmanlarında).

Her fold'un bellekte tutulan modeli, o fold'un TEST satırlarında kullanılır: bir kaynağın TÜM kolonları (özellikler +
`__missing` bayrağı) birlikte, her tarihte semboller arasında aynı permütasyonla karıştırılır (özellikler arası ilişki
korunur, özellik-etiket bağı koparılır); önem = temel IC − karıştırılmış IC (pozitif = kaynak işe yarıyor). Seed'li ve tekrarlı."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import pred_metrics
from .oof import OofResult


def _permute_group(X: pd.DataFrame, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    out = X.copy()
    dates = X.index.get_level_values("date")
    block = X[cols].to_numpy().copy()
    for d in pd.unique(dates):
        pos = np.flatnonzero(dates == d)
        block[pos] = block[pos[rng.permutation(len(pos))]]
    out[cols] = block
    return out


def permutation_importance(
    X: pd.DataFrame, y: pd.Series, res: OofResult, source_names: list[str], horizon: int = 1, n_repeats: int = 5, seed: int = 0
) -> pd.DataFrame:
    """Dönen tablo: satır = (model, kaynak); kolonlar: importance (ort. IC düşüşü), importance_std, baseline_ic."""
    rng = np.random.default_rng(seed)
    records = []
    labels = list(res.folds[0].models) if res.folds else []
    for model_id in labels:
        base_ics = []
        perm_ics: dict[str, list[float]] = {s: [] for s in source_names}
        for fold in res.folds:
            Xte, yte = X[fold.test_mask], y[fold.test_mask]
            model = fold.models[model_id]
            base = pred_metrics.ic_summary(pd.Series(model.predict(Xte), index=Xte.index), yte, horizon)["ic_mean"]
            base_ics.append(base)
            for s in source_names:
                cols = [c for c in X.columns if c.startswith(f"{s}__")]
                if not cols:
                    continue
                reps = []
                for _ in range(n_repeats):
                    pred = pd.Series(model.predict(_permute_group(Xte, cols, rng)), index=Xte.index)
                    reps.append(pred_metrics.ic_summary(pred, yte, horizon)["ic_mean"])
                perm_ics[s].append(base - float(np.nanmean(reps)))
        for s in source_names:
            if perm_ics[s]:
                records.append({"model": model_id, "source": s, "importance": float(np.nanmean(perm_ics[s])),
                                "importance_std": float(np.nanstd(perm_ics[s])), "baseline_ic": float(np.nanmean(base_ics))})
    return pd.DataFrame(records).set_index(["model", "source"]) if records else pd.DataFrame()
