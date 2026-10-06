"""Walk-forward (purge + embargo) out-of-fold tahmin üretimi. Fold modelleri bellekte tutulur (önem hesabı için)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import stats
from .models.registry import get_model
from .targets import Target


@dataclass
class FoldInfo:
    k: int
    train_mask: np.ndarray  # satır maskesi (X ile aynı sıra)
    test_mask: np.ndarray
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    models: dict = field(default_factory=dict)  # model adı -> fit edilmiş model


@dataclass
class OofResult:
    oof: pd.DataFrame  # (date, symbol) × model adı
    folds: list[FoldInfo]


def compatible(model_task: str, target_kind: str) -> bool:
    return model_task == target_kind or (model_task == "regression" and target_kind in ("rank", "regression"))


def build_folds(dates: pd.Series, cv: dict, target: Target) -> list[tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Tarih bazlı bölmeler: purge = ufuk (h), embargo >= h (zorunlu). Satır maskeleri + tarih kümeleri döner."""
    h = target.horizon
    if cv["embargo_days"] < h:
        raise ValueError(f"embargo_days ({cv['embargo_days']}) hedef ufkundan ({h}) küçük olamaz")
    ud = pd.DatetimeIndex(sorted(pd.unique(dates)))
    out = []
    for train_pos, test_pos in stats.purged_walk_forward_splits(ud, cv["n_splits"], embargo_days=cv["embargo_days"], purge_days=h):
        tr_dates, te_dates = ud[train_pos], ud[test_pos]
        if len(tr_dates) < cv.get("min_train_days", 0):
            continue
        out.append((dates.isin(tr_dates).to_numpy(), dates.isin(te_dates).to_numpy(), tr_dates, te_dates))
    if not out:
        raise ValueError("Walk-forward için yeterli tarih yok (min_train_days / n_splits'i düşürün veya dönemi uzatın)")
    return out


def run_oof(
    X: pd.DataFrame,
    y: pd.Series,
    models_cfg: list[dict],
    target: Target,
    cv: dict,
    seed: int,
    folds=None,
) -> OofResult:
    """Her fold için her modeli eğitim satırlarında fit eder, test satırlarını tahmin eder. `y` NaN satırlar fit'ten atılır."""
    dates = pd.Series(X.index.get_level_values("date"), index=X.index)
    folds = folds or build_folds(dates, cv, target)
    oof = pd.DataFrame(np.nan, index=X.index, columns=[m["name"] if "id" not in m else m["id"] for m in models_cfg])
    infos: list[FoldInfo] = []
    for k, (tr, te, tr_dates, te_dates) in enumerate(folds):
        info = FoldInfo(k, tr, te, tr_dates, te_dates)
        for mc in models_cfg:
            label = mc.get("id", mc["name"])
            model = get_model(mc["name"], seed=seed, **mc.get("params", {}))
            if not compatible(model.task, target.kind):
                raise ValueError(f"Model '{mc['name']}' ({model.task}) hedef '{target.name}' ({target.kind}) ile uyumsuz")
            model.fit(X[tr], y[tr], dates[tr])
            oof.loc[te, label] = model.predict(X[te])
            info.models[label] = model
        infos.append(info)
    return OofResult(oof=oof, folds=infos)
