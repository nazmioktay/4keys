"""Karşılaştırma araçları: ablasyon, permütasyon önemi, stacking (sentetik piyasa + bilerek gürültü kaynağı)."""

import numpy as np
import pandas as pd
import pytest

from research import ablation, config, importance, registry, runner, stacking
from research.oof import build_folds, run_oof
from research.panel import build_panel
from research.sources.base import Source
from research.sources.registry import SOURCE_REGISTRY, register_source

from tests.research.synth import make_market, write_market


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cmp")
    old = config.CACHE_DIR
    config.CACHE_DIR = tmp / "cache"
    write_market(make_market(n_days=900, seed=7))

    @register_source
    class NoiseSrc(Source):
        name = "noise_src"
        max_staleness = pd.Timedelta(days=3)

        def fetch(self, start, end, symbols=None): ...

        def to_panel(self, universe, dates):
            rng = np.random.default_rng(99)
            idx = pd.MultiIndex.from_product([dates, universe], names=["date", "symbol"]).to_frame(index=False)
            for i in range(3):
                idx[f"noise_src__n{i}"] = rng.normal(size=len(idx))
            idx["available_at"] = idx["date"] + pd.Timedelta(days=1)
            return idx

    cfg = runner.load_config({
        "id": "cmp", "question": "q", "sources": ["ohlcv_core", "noise_src"], "universe": {"n": 10},
        "period": {"start": "2021-07-01", "end": "2023-06-30"}, "target": {"name": "next_rv", "h": 1},
        "models": [{"name": "ridge"}], "cv": {"n_splits": 3, "embargo_days": 1, "min_train_days": 200},
        "signal": {"adapter": "vol_target", "params": {"target_vol_annual": 0.4}},
    })
    panel = build_panel(cfg)
    dates = pd.Series(panel.X.index.get_level_values("date"), index=panel.X.index)
    folds = build_folds(dates, cfg["cv"], panel.target)
    res = run_oof(panel.X, panel.y, cfg["models"], panel.target, cfg["cv"], 0, folds)
    yield cfg, panel, folds, res, tmp
    SOURCE_REGISTRY.pop("noise_src", None)
    config.CACHE_DIR = old


def test_ablation_attributes_skill_to_the_informative_source_not_to_noise(world):
    cfg, panel, folds, res, tmp = world
    table = ablation.run_ablation(cfg, panel, n_boot=200)
    core, noise = table.loc["ohlcv_core"], table.loc["noise_src"]
    assert core["d_ic"] > 0.1 and core["d_ic_lo"] > 0  # bilgi veren kaynak: GA sıfırı dışlar
    assert abs(noise["d_ic"]) < 0.02 and noise["d_ic_lo"] <= 0 <= noise["d_ic_hi"]  # gürültü: GA sıfırı içerir
    md = ablation.ablation_markdown(table)
    assert "ohlcv_core" in md and "ΔIC" in md


def test_ablation_registration_counts_each_ablation_as_a_trial(world, tmp_path):
    cfg, panel, folds, res, _ = world
    paths = dict(results_dir=tmp_path / "res", log_path=tmp_path / "deneyler.md", registry_file=tmp_path / "res" / "_registry.json")
    (tmp_path / "deneyler.md").write_text("# g\n\n**Toplam deneme: 0**\n\n| t |\n|---|\n", encoding="utf-8")
    ablation.run_ablation(cfg, panel, n_boot=50, register=True, **paths)
    assert registry.current_trial_count(paths["registry_file"]) == 2  # kaynak sayısı kadar
    assert (tmp_path / "res" / "cmp__abl_noise_src" / "metrics.json").exists()


def test_permutation_importance_ranks_the_informative_source_first(world):
    cfg, panel, folds, res, _ = world
    imp = importance.permutation_importance(panel.X, panel.y, res, [s.name for s in panel.sources], horizon=1, n_repeats=3)
    core = imp.loc[("ridge", "ohlcv_core"), "importance"]
    noise = imp.loc[("ridge", "noise_src"), "importance"]
    assert core > 0.1 and abs(noise) < 0.02 and core > 10 * abs(noise)
    again = importance.permutation_importance(panel.X, panel.y, res, [s.name for s in panel.sources], horizon=1, n_repeats=3)
    pd.testing.assert_frame_equal(imp, again)  # seed'li -> tekrarlanabilir


# ---------------------------------------------------------------- stacking
def _oof_world(n_dates=400, n_sym=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([pd.date_range("2022-01-01", periods=n_dates), [f"S{i}" for i in range(n_sym)]], names=["date", "symbol"])
    y = pd.Series(rng.normal(size=len(idx)), index=idx)
    good = y + rng.normal(0, 0.5, len(idx))
    bad = rng.normal(size=len(idx))
    oof = pd.DataFrame({"good": good, "bad": bad}, index=idx)
    dates = pd.date_range("2022-01-01", periods=n_dates)
    folds = [dates[i * 100:(i + 1) * 100] for i in range(4)]
    return oof, y, folds


def test_stacking_weights_the_informative_model_and_leaves_fold0_empty():
    oof, y, folds = _oof_world()
    stacked, info = stacking.stack_oof(oof, y, folds, horizon=1)
    assert stacked[oof.index.get_level_values("date").isin(folds[0])].isna().all()  # fold 0 geçmişsiz
    assert info[0]["coef"]["good"] > 5 * abs(info[0]["coef"]["bad"])
    for k in (1, 2, 3):
        mask = oof.index.get_level_values("date").isin(folds[k])
        assert stacked[mask].notna().all()
        assert np.corrcoef(stacked[mask], y[mask])[0, 1] > 0.8


def test_stacking_never_sees_the_test_fold_labels_or_the_purge_window():
    oof, y, folds = _oof_world()
    base, _ = stacking.stack_oof(oof, y, folds, horizon=5)
    dates = oof.index.get_level_values("date")
    # fold k'nın KENDİ etiketlerini bozmak fold k stacked tahminini DEĞİŞTİRMEZ
    y2 = y.copy()
    y2[dates.isin(folds[2])] = 1e6
    s2, _ = stacking.stack_oof(oof, y2, folds, horizon=5)
    k2 = dates.isin(folds[2])
    np.testing.assert_array_equal(base[k2], s2[k2])
    # Purge sınırı (h=5): train günü d ancak d + (h+1) <= ilk_test + 1, yani d <= ilk_test - h ise kullanılır.
    # Fold 1'in SON 4 günü (etiketi fold 2'nin ilk kararından SONRA tamamlanır) birleştiriciye girmez:
    y3 = y.copy()
    y3[dates.isin(folds[1][-4:])] = 1e6
    s3, _ = stacking.stack_oof(oof, y3, folds, horizon=5)
    np.testing.assert_array_equal(base[k2], s3[k2])
    # ... ama sınırdaki gün (ilk_test - h = fold 1'in sondan 5.'si) etiketi tam karar anında tamamlandığı için GİRER:
    y3b = y.copy()
    y3b[dates.isin(folds[1][-5:-4])] = 1e6
    s3b, _ = stacking.stack_oof(oof, y3b, folds, horizon=5)
    assert not np.array_equal(base[k2], s3b[k2])
    # ama fold 1'in daha eski satırları girer: bozmak tahmini DEĞİŞTİRİR (test boş değil)
    y4 = y.copy()
    y4[dates.isin(folds[1][:50])] = 1e6
    s4, _ = stacking.stack_oof(oof, y4, folds, horizon=5)
    assert not np.array_equal(base[k2], s4[k2])


def test_stacking_refuses_the_final_test_window():
    oof, y, folds = _oof_world()
    oof.index = oof.index.set_levels(oof.index.levels[0] + pd.Timedelta(days=1300), level=0)
    shifted = [f + pd.Timedelta(days=1300) for f in folds]
    with pytest.raises(Exception, match="nihai test"):
        stacking.stack_oof(oof, y, shifted, horizon=1)
