"""Otomatik sızıntı / sağlamlık testleri. Biri BAŞARISIZSA deney GEÇERSİZ sayılır (uyarılar hariç).

1. shuffled_target   etiketler tarih içinde (semboller arası) karıştırılıp modeller yeniden eğitilir -> IC ≈ 0 olmalı (|t| < eşik)
2. availability      her `<kaynak>__available_at <= karar zamanı`; eğitim satırlarının etiket bitişi test başlangıcından önce
3. extra_lag         tüm özellikler +1 gün geciktirilince beceri TAMAMEN çöküyorsa "zamanlama sızıntısı şüphesi" (UYARI).
                     SINIR: kalıcı hedeflerde (ör. volatilite; y_{t-1} y_t'yi zaten öngörür) güçsüzdür — bu hedeflerde sızıntıyı
                     available_at ve karıştırılmış-hedef testleri yakalar; kesitsel getiri/sıra hedeflerinde güçlüdür
4. reproducibility   aynı config iki kez -> aynı OOF

Her test, sızıntıyı kasten enjekte eden negatif kontrollerle (tests/research/test_checks.py) doğrulanır."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import pred_metrics
from .oof import OofResult, run_oof
from .panel import Panel
from .sources.pit import audit_available_at

SHUFFLE_T_LIMIT = 3.0
LAG_COLLAPSE_RATIO = 0.2
LAG_BASE_T_MIN = 3.0


@dataclass
class CheckReport:
    results: dict = field(default_factory=dict)  # test -> ayrıntı sözlüğü
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.failures

    def add(self, name: str, passed: bool, detail: dict, warning: bool = False) -> None:
        self.results[name] = {"passed": passed, **detail}
        if not passed:
            (self.warnings if warning else self.failures).append(name)


def shuffle_within_date(y: pd.Series, seed: int) -> pd.Series:
    """Her tarihte etiketleri semboller arasında karıştırır (etiket marjinalleri korunur, özellik-etiket bağı yok edilir)."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    dates = y.index.get_level_values("date")
    for d in pd.unique(dates):
        pos = np.flatnonzero(dates == d)
        vals = y.iloc[pos].to_numpy()
        out.iloc[pos] = vals[rng.permutation(len(vals))]
    return out


def _ic_per_model(oof: pd.DataFrame, y: pd.Series, horizon: int) -> dict[str, dict]:
    return {m: pred_metrics.ic_summary(oof[m], y, horizon) for m in oof.columns}


def check_shuffled_target(panel: Panel, models_cfg, cv, seed, horizon, folds, report: CheckReport) -> None:
    y_shuf = shuffle_within_date(panel.y, seed + 12345)
    res = run_oof(panel.X, y_shuf, models_cfg, panel.target, cv, seed, folds)
    stats_ = _ic_per_model(res.oof, y_shuf, horizon)
    bad = {m: s for m, s in stats_.items() if not (abs(s["ic_t"]) < SHUFFLE_T_LIMIT) and s["n_dates"] > 0}
    report.add("shuffled_target", not bad, {"per_model": {m: {"ic_mean": s["ic_mean"], "ic_t": s["ic_t"]} for m, s in stats_.items()}, "limit_t": SHUFFLE_T_LIMIT})


def check_availability(panel: Panel, folds, horizon: int, report: CheckReport) -> None:
    violations = audit_available_at(panel.merged.reset_index(drop=True), panel.sources)
    # etiket sızıntısı: eğitim satırının etiketi, ilk test kararından ÖNCE tamamlanmış olmalı
    label_end_days = horizon + 1
    label_leaks = 0
    for tr_mask, te_mask, tr_dates, te_dates in folds:
        first_test_decision = te_dates.min() + pd.Timedelta(days=1)
        label_leaks += int((tr_dates + pd.Timedelta(days=label_end_days) > first_test_decision).sum())
    ok = sum(violations.values()) == 0 and label_leaks == 0
    report.add("availability", ok, {"available_at_violations": violations, "train_label_overlap_days": label_leaks})


def check_extra_lag(panel: Panel, models_cfg, cv, seed, horizon, folds, base_oof: pd.DataFrame, report: CheckReport) -> None:
    Xlag = panel.X.groupby(level="symbol", sort=False).shift(1)
    res = run_oof(Xlag, panel.y, models_cfg, panel.target, cv, seed, folds)
    base = _ic_per_model(base_oof, panel.y, horizon)
    lagged = _ic_per_model(res.oof, panel.y, horizon)
    suspicious = {}
    for m in base:
        b, l = base[m]["ic_mean"], lagged[m]["ic_mean"]
        ratio = (l / b) if b and b == b and abs(b) > 1e-12 else float("nan")
        if base[m]["ic_t"] == base[m]["ic_t"] and base[m]["ic_t"] > LAG_BASE_T_MIN and ratio == ratio and ratio < LAG_COLLAPSE_RATIO:
            suspicious[m] = ratio
    report.add(
        "extra_lag", not suspicious,
        {"ic_base": {m: base[m]["ic_mean"] for m in base}, "ic_lag1": {m: lagged[m]["ic_mean"] for m in lagged},
         "collapsed": suspicious, "note": "zamanlama sızıntısı şüphesi: +1 gün gecikmede beceri çöktü" if suspicious else ""},
        warning=True,
    )


def check_reproducibility(a: pd.DataFrame, b: pd.DataFrame, report: CheckReport, atol: float = 1e-10) -> None:
    same_shape = a.shape == b.shape and list(a.columns) == list(b.columns)
    max_diff = float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy()))) if same_shape and a.size else float("inf")
    same_nan = same_shape and bool((a.isna().to_numpy() == b.isna().to_numpy()).all())
    report.add("reproducibility", same_shape and same_nan and max_diff <= atol, {"max_abs_diff": max_diff})
