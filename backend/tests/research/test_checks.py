"""Sızıntı/sağlamlık testlerinin NEGATİF KONTROLLERİ: bilerek bozuk bir boru hattını her test yakalamalı;
dürüst boru hattı hepsinden geçmeli."""

import numpy as np
import pandas as pd
import pytest

from research import checks, config, runner
from research.models.base import BaseModel
from research.models.registry import MODEL_REGISTRY, register_model
from research.oof import build_folds, run_oof
from research.panel import build_panel
from research.sources.pit import AVAILABLE_SUFFIX

from tests.research.synth import make_market, write_market


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("chk")
    old = config.CACHE_DIR
    config.CACHE_DIR = tmp / "cache"
    write_market(make_market(n_days=900, seed=4))
    cfg = runner.load_config({
        "id": "chk", "sources": ["ohlcv_core"], "universe": {"n": 10}, "period": {"start": "2021-07-01", "end": "2023-06-30"},
        "target": {"name": "next_rv", "h": 1}, "models": ["har_rv", "lightgbm_reg"], "cv": {"n_splits": 3, "embargo_days": 1, "min_train_days": 200},
        "signal": {"adapter": "vol_target"},
    })
    panel = build_panel(cfg)
    dates = pd.Series(panel.X.index.get_level_values("date"), index=panel.X.index)
    folds = build_folds(dates, cfg["cv"], panel.target)
    yield cfg, panel, folds
    config.CACHE_DIR = old


def test_honest_pipeline_passes_every_check(setup):
    cfg, panel, folds = setup
    rep = checks.CheckReport()
    res = run_oof(panel.X, panel.y, cfg["models"], panel.target, cfg["cv"], 0, folds)
    checks.check_availability(panel, folds, 1, rep)
    checks.check_shuffled_target(panel, cfg["models"], cfg["cv"], 0, 1, folds, rep)
    checks.check_extra_lag(panel, cfg["models"], cfg["cv"], 0, 1, folds, res.oof, rep)
    again = run_oof(panel.X, panel.y, cfg["models"], panel.target, cfg["cv"], 0, folds)
    checks.check_reproducibility(res.oof, again.oof, rep)
    assert rep.valid and not rep.warnings, (rep.failures, rep.warnings, rep.results)


def test_shuffled_target_catches_a_pipeline_that_trains_on_test_labels(setup):
    """Negatif kontrol: eğitim kümesi test satırlarını da içeriyorsa (CV sızıntısı) karıştırılmış etiketlerde bile IC > 0 çıkar."""
    cfg, panel, folds = setup
    leaky = [(np.ones_like(tr), te, tr_d, te_d) for tr, te, tr_d, te_d in folds]  # eğitim = TÜM satırlar (test dahil)
    models = [{"name": "lightgbm_reg", "id": "lightgbm_reg", "params": {"n_estimators": 300, "num_leaves": 63, "min_child_samples": 3, "learning_rate": 0.2}}]
    rep = checks.CheckReport()
    checks.check_shuffled_target(panel, models, cfg["cv"], 0, 1, leaky, rep)
    assert "shuffled_target" in rep.failures
    ok = checks.CheckReport()
    checks.check_shuffled_target(panel, models, cfg["cv"], 0, 1, folds, ok)  # düzgün CV ile aynı model GEÇER
    assert ok.valid


def test_availability_audit_catches_features_published_after_decision_time(setup):
    cfg, panel, folds = setup
    bad = panel.merged.copy()
    col = f"ohlcv_core{AVAILABLE_SUFFIX}"
    bad[col] = bad["decision_time"] + pd.Timedelta(hours=1)  # karar zamanından SONRA yayımlanmış
    panel2 = type(panel)(**{**panel.__dict__, "merged": bad})
    rep = checks.CheckReport()
    checks.check_availability(panel2, folds, 1, rep)
    assert "availability" in rep.failures and rep.results["availability"]["available_at_violations"]["ohlcv_core"] == len(bad)


def test_availability_audit_catches_training_labels_that_overlap_the_test_period(setup):
    cfg, panel, folds = setup
    tr, te, tr_d, te_d = folds[0]
    bad_train_dates = tr_d.append(pd.DatetimeIndex([te_d.min() - pd.Timedelta(hours=0)]))  # test başlangıcının ETİKETİ test içine taşan gün
    rep = checks.CheckReport()
    checks.check_availability(panel, [(tr, te, bad_train_dates, te_d)], 1, rep)
    assert "availability" in rep.failures and rep.results["availability"]["train_label_overlap_days"] > 0


def test_extra_lag_warns_when_skill_comes_from_a_same_day_leak(setup):
    """Negatif kontrol: özellik = gerçek hedefin kendisi (sızıntı). Hedef KALICI olmayan kesitsel getiri sırası olduğundan
    +1 gün gecikmede beceri çöker -> uyarı. (Volatilite gibi KALICI hedeflerde bu test zayıftır: y_{t-1}, y_t'yi zaten
    öngörür — bkz. test_extra_lag_has_low_power_on_persistent_targets.)"""
    cfg, _, _ = setup
    cfg2 = runner.load_config({**{k: cfg[k] for k in ("id", "sources", "universe", "period", "signal")},
                               "target": {"name": "cross_sectional_rank", "h": 1}, "models": ["ridge"], "cv": cfg["cv"]})
    panel = build_panel(cfg2)
    dates = pd.Series(panel.X.index.get_level_values("date"), index=panel.X.index)
    folds = build_folds(dates, cfg2["cv"], panel.target)
    leaky_X = panel.X.copy()
    leaky_X["leak"] = panel.y.to_numpy()
    p2 = type(panel)(**{**panel.__dict__, "X": leaky_X})
    base = run_oof(leaky_X, panel.y, cfg2["models"], panel.target, cfg2["cv"], 0, folds)
    rep = checks.CheckReport()
    checks.check_extra_lag(p2, cfg2["models"], cfg2["cv"], 0, 1, folds, base.oof, rep)
    assert "extra_lag" in rep.warnings and rep.valid  # UYARI (geçersiz değil): şüphe bildirilir
    assert "zamanlama sızıntısı şüphesi" in rep.results["extra_lag"]["note"]
    assert rep.results["extra_lag"]["ic_base"]["ridge"] > 0.9 and abs(rep.results["extra_lag"]["ic_lag1"]["ridge"]) < 0.2


def test_extra_lag_has_low_power_on_persistent_targets(setup):
    """Bilinen SINIR (belgelenmiş): volatilite gibi kalıcı hedefte, sızdırılmış etiket +1 gün gecikince de beceri korunur
    (y_{t-1} ile y_t ilişkili) -> kontrol uyarı VERMEZ. Bu hedeflerde sızıntıyı available_at ve karıştırılmış-hedef
    testleri yakalar; bu test sınırı kayıt altında tutar."""
    cfg, panel, folds = setup
    leaky_X = panel.X.copy()
    leaky_X["leak"] = panel.y.to_numpy()
    p2 = type(panel)(**{**panel.__dict__, "X": leaky_X})
    models = [{"name": "ridge", "id": "ridge", "params": {}}]
    base = run_oof(leaky_X, panel.y, models, panel.target, cfg["cv"], 0, folds)
    rep = checks.CheckReport()
    checks.check_extra_lag(p2, models, cfg["cv"], 0, 1, folds, base.oof, rep)
    assert rep.warnings == [] and rep.results["extra_lag"]["ic_lag1"]["ridge"] > 0.3


def test_reproducibility_catches_a_nondeterministic_model(setup):
    cfg, panel, folds = setup

    @register_model("tohumsuz_test")
    class Unseeded(BaseModel):
        task = "regression"
        handles_nan = True

        def fit(self, X, y, dates, sample_weight=None):
            return self

        def predict(self, X):
            return np.random.default_rng().normal(size=len(X))  # seed YOK

    try:
        models = [{"name": "tohumsuz_test", "id": "tohumsuz_test", "params": {}}]
        a = run_oof(panel.X, panel.y, models, panel.target, cfg["cv"], 0, folds)
        b = run_oof(panel.X, panel.y, models, panel.target, cfg["cv"], 0, folds)
        rep = checks.CheckReport()
        checks.check_reproducibility(a.oof, b.oof, rep)
        assert "reproducibility" in rep.failures
    finally:
        MODEL_REGISTRY.pop("tohumsuz_test", None)


def test_shuffle_within_date_preserves_per_date_values_and_breaks_alignment():
    idx = pd.MultiIndex.from_product([pd.date_range("2024-01-01", periods=20), list("ABCDE")], names=["date", "symbol"])
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    s = checks.shuffle_within_date(y, 1)
    assert all(sorted(s.xs(d, level="date")) == sorted(y.xs(d, level="date")) for d in y.index.get_level_values("date").unique())
    assert not s.equals(y)
    assert checks.shuffle_within_date(y, 1).equals(s)  # tohumlu -> tekrarlanabilir
