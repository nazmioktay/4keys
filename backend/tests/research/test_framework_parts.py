"""Çerçeve parçaları: vektörel evren, tahmin metrikleri, sinyal adaptörleri, deneme bütçesi."""

import numpy as np
import pandas as pd
import pytest

from research import budget, pred_metrics, registry, signals
from research.data import universe


# ---------------------------------------------------------------- vektörel evren == universe_at
def _qv(days=500, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=days, freq="D")
    cols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", "XRPUSDT", "USDCUSDT", "BTCUPUSDT", "BTCDOMUSDT", "LATEUSDT", "GONEUSDT"]
    base = dict(zip(cols, [1e9, 6e8, 3e8, 1.2e8, 9e7, 8e7, 9e9, 8e8, 7e8, 5e8, 4e8]))
    panel = pd.DataFrame({c: base[c] * rng.uniform(0.6, 1.4, days) for c in cols}, index=idx)
    panel.loc[panel.index < "2022-09-01", "LATEUSDT"] = np.nan
    panel.loc[panel.index >= "2023-03-01", "GONEUSDT"] = np.nan  # delist
    panel.iloc[40:43, 2] = np.nan  # eksik günler
    return panel


def test_vectorized_membership_equals_universe_at_on_many_dates():
    qv = _qv()
    mem = universe.universe_membership(qv, 4)
    for d in pd.date_range("2022-05-01", "2023-05-01", freq="17D"):
        expected = set(universe.universe_at(d, 4, qv))
        got = set(mem.columns[mem.loc[d].to_numpy()])
        assert got == expected, (d, got, expected)
    assert not mem[["USDCUSDT", "BTCUPUSDT", "BTCDOMUSDT"]].any().any()
    assert mem.sum(axis=1).max() <= 4


def test_vectorized_membership_does_not_use_the_future():
    qv = _qv()
    mem = universe.universe_membership(qv, 4)
    cut = pd.Timestamp("2022-10-01")
    shuffled = qv.copy()
    shuffled.loc[shuffled.index >= cut] = np.random.default_rng(0).uniform(1e3, 1e12, shuffled.loc[shuffled.index >= cut].shape)
    mem2 = universe.universe_membership(shuffled, 4)
    pd.testing.assert_frame_equal(mem.loc[:cut], mem2.loc[:cut])  # cut günü dahil: cut'ın kendi verisi cut üyeliğini etkilemez


# ---------------------------------------------------------------- tahmin metrikleri
def _pm_frame(n_dates=200, n_sym=10, rho=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([pd.date_range("2022-01-01", periods=n_dates), [f"S{i}" for i in range(n_sym)]], names=["date", "symbol"])
    y = pd.Series(rng.normal(size=len(idx)), index=idx)
    pred = rho * y + np.sqrt(1 - rho**2) * pd.Series(rng.normal(size=len(idx)), index=idx)
    return pred, y


def test_daily_ic_recovers_known_correlation_and_t_stat_is_large():
    pred, y = _pm_frame(rho=0.3)
    s = pred_metrics.ic_summary(pred, y, horizon=1)
    assert s["ic_mean"] == pytest.approx(0.3, abs=0.06) and s["rank_ic_mean"] == pytest.approx(0.29, abs=0.07)
    assert s["ic_t"] > 5 and s["n_dates"] == 200
    noise_pred, y2 = _pm_frame(rho=0.0, seed=2)
    assert abs(pred_metrics.ic_summary(noise_pred, y2)["ic_t"]) < 3


def test_newey_west_t_is_smaller_than_naive_for_autocorrelated_series():
    rng = np.random.default_rng(0)
    e = rng.normal(size=600)
    x = pd.Series(np.convolve(e, np.ones(10) / 10, mode="valid") + 0.05)  # güçlü pozitif otokorelasyon
    naive = x.mean() / (x.std() / np.sqrt(len(x)))
    assert pred_metrics.newey_west_t(x, lag=9) < naive * 0.7
    assert pred_metrics.newey_west_t(x, lag=0) == pytest.approx(naive, rel=0.02)


def test_qlike_is_minimized_at_the_truth_and_zero_when_exact():
    true = np.array([0.02, 0.03, 0.05, 0.04])
    assert pred_metrics.qlike(true, true) == pytest.approx(0.0, abs=1e-9)
    assert pred_metrics.qlike(true * 1.5, true) > 0 and pred_metrics.qlike(true * 0.5, true) > pred_metrics.qlike(true * 1.5, true)  # düşük tahmin daha cezalı


def test_classification_metrics_auc_logloss_and_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(float)  # kalibre edilmiş tahmin
    idx = pd.MultiIndex.from_product([pd.date_range("2022-01-01", periods=500), [f"S{i}" for i in range(10)]], names=["date", "symbol"])
    out = pred_metrics.evaluate(pd.Series(p, index=idx), pd.Series(y, index=idx), "classification", 1)
    assert out["auc"] == pytest.approx(5 / 6, abs=0.02)  # p~U(0,1), y~Bernoulli(p): teorik AUC = 5/6
    assert out["ece"] < 0.03 and len(out["calibration"]) == 10
    assert out["log_loss"] == pytest.approx(0.5, abs=0.02)  # teorik log-loss = 1/2


def test_evaluate_reports_nan_predictions_without_scoring_them():
    pred, y = _pm_frame(n_dates=50)
    pred.iloc[:30] = np.nan
    out = pred_metrics.evaluate(pred, y, "regression", 1, "next_rv")
    assert out["n_pred_nan"] == 30 and out["n_scored"] == len(y) - 30 and "qlike" in out


# ---------------------------------------------------------------- sinyal adaptörleri
def _wide(vals, dates=3):
    idx = pd.date_range("2024-01-01", periods=dates)
    return pd.DataFrame([vals] * dates, index=idx, columns=["A", "B", "C", "D"], dtype=float)


def test_vol_target_scales_inverse_to_forecast_caps_and_ignores_non_members():
    pred = _wide([0.02, 0.04, 0.08, 0.02])
    member = pd.DataFrame(True, index=pred.index, columns=pred.columns)
    member["D"] = False
    w = signals.make_weights("vol_target", pred, member, target_vol_annual=0.40, cap=1.0)
    daily = 0.40 / np.sqrt(365)  # ≈ 0,0209
    assert w.iloc[0]["A"] == pytest.approx(min(1.0, daily / 0.02) / 3)
    assert w.iloc[0]["B"] == pytest.approx(daily / 0.04 / 3) and w.iloc[0]["C"] == pytest.approx(daily / 0.08 / 3)
    assert w.iloc[0]["D"] == 0.0 and w.sum(axis=1).max() <= 1.0 + 1e-12
    nan_pred = pred.copy()
    nan_pred.iloc[0, 0] = np.nan
    assert signals.make_weights("vol_target", nan_pred, member).iloc[0]["A"] == 0.0  # NaN tahmin -> pozisyon yok


def test_rank_long_short_is_dollar_neutral_with_unit_gross():
    pred = _wide([1.0, 2.0, 3.0, 4.0])
    member = pd.DataFrame(True, index=pred.index, columns=pred.columns)
    w = signals.make_weights("rank_long_short", pred, member, q=0.25)
    row = w.iloc[0]
    assert row["D"] == pytest.approx(0.5) and row["A"] == pytest.approx(-0.5) and row[["B", "C"]].abs().sum() == 0
    assert row.sum() == pytest.approx(0.0) and row.abs().sum() == pytest.approx(1.0)
    lo = signals.make_weights("rank_long_short", pred, member, q=0.25, long_only=True).iloc[0]
    assert lo["D"] == pytest.approx(1.0) and lo.abs().sum() == pytest.approx(1.0)


def test_probability_adapters_filter_and_size():
    pred = _wide([0.3, 0.55, 0.8, 0.9])
    member = pd.DataFrame(True, index=pred.index, columns=pred.columns)
    f = signals.make_weights("prob_filter", pred, member, threshold=0.5).iloc[0]
    assert f["A"] == 0 and f[["B", "C", "D"]].eq(0.25).all()
    s = signals.make_weights("prob_size", pred, member).iloc[0]
    assert s["A"] == 0 and s["B"] == pytest.approx(0.1 / 4) and s["D"] == pytest.approx(0.8 / 4)
    with pytest.raises(KeyError):
        signals.make_weights("yok", pred, member)


# ---------------------------------------------------------------- deneme bütçesi
def test_budget_counts_variants_per_question_and_blocks_at_the_limit(tmp_path):
    reg = tmp_path / "_registry.json"
    paths = dict(results_dir=tmp_path / "res", log_path=tmp_path / "deneyler.md", registry_file=reg)
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 300), index=pd.date_range("2021-01-01", periods=300))
    for i in range(3):
        registry.register_experiment(f"e{i}", {}, r, {"sharpe": 1.0, "max_drawdown": -0.1}, question="vol_tahmini", **paths)
    registry.register_experiment("x1", {}, r, {"sharpe": 1.0, "max_drawdown": -0.1}, question="baska", n_variants=5, **paths)
    assert budget.variants_used("vol_tahmini", reg) == 3 and budget.variants_used("baska", reg) == 5  # varyant sayısı, deney değil
    assert budget.assert_budget("vol_tahmini", 1, reg, limit=4) == 0
    with pytest.raises(budget.BudgetExceededError, match="bütçesi dolu"):
        budget.assert_budget("vol_tahmini", 2, reg, limit=4)
    assert budget.MAX_VARIANTS_PER_QUESTION == 40


def test_a_single_experiment_cannot_exceed_the_variant_budget(tmp_path, monkeypatch):
    reg = tmp_path / "_registry.json"
    paths = dict(results_dir=tmp_path / "res", log_path=tmp_path / "deneyler.md", registry_file=reg)
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 300), index=pd.date_range("2021-01-01", periods=300))
    monkeypatch.setattr(registry, "MAX_VARIANTS_PER_QUESTION", 40)
    with pytest.raises(budget.BudgetExceededError, match="41 yeni"):  # 216 varyantlık ızgara gibi: tek deney de bütçeyi aşamaz
        registry.register_experiment("buyuk", {}, r, {"sharpe": 1.0, "max_drawdown": -0.1}, question="q", n_variants=41, **paths)
    assert registry.current_trial_count(reg) == 0


def test_registry_tolerates_legacy_records_and_refuses_to_overwrite_a_corrupt_file(tmp_path):
    import json

    reg = tmp_path / "_registry.json"
    reg.write_text(json.dumps({"total_trials": 2, "experiments": [
        {"id": "eski1", "question": None}, {"id": "eski2", "question": " Garip Soru!  "}]}), encoding="utf-8")  # smoke_runs anahtarı YOK
    assert budget.variants_used("genel", reg) == 1  # question=None -> "genel"; n_variants yok -> 1
    assert budget.variants_used("baska_soru", reg) == 0  # regex'e uymayan ESKİ kayıt (' Garip Soru!  ') sayımı/bütçeyi BOZMAZ (toleranslı)
    assert budget.assert_budget("baska_soru", 1, reg) == 39
    assert registry.record_smoke_run("genel", reg) == 1
    reg.write_text("{bozuk json", encoding="utf-8")
    with pytest.raises(registry.RegistryCorruptError, match="üzerine yazılmadı"):
        registry.record_smoke_run("genel", reg)
    assert reg.read_text(encoding="utf-8") == "{bozuk json"  # bozuk dosya SİLİNMEDİ/ezilmedi
    with pytest.raises(registry.RegistryCorruptError):
        registry.current_trial_count(reg)


def test_registry_write_is_atomic_no_tmp_file_left_behind(tmp_path):
    reg = tmp_path / "_registry.json"
    registry.record_smoke_run("q", reg)
    assert reg.exists() and not (tmp_path / "_registry.json.tmp").exists()
