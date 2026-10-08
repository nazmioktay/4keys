"""ml_kol: edge_threshold adaptörü (maliyet eşiği, işaret, tavanlar, ileri bakış yok, bağlam zorunlu), 18 varyantlık ızgara ve plato
parametreleri, koşucunun yürütme bandı (varsayılan 0 = eski davranış), sentetik piyasada uçtan uca kayıtlı koşu."""

import numpy as np
import pandas as pd
import pytest

from research import config, ml_kol, runner, signals
from tests.research.synth import make_market, write_market


def _ctx(close, slip_bps=5.0):
    return {"close": close, "slip_bps": pd.DataFrame(slip_bps, index=close.index, columns=close.columns), "nav": 250.0,
            "min_notional": None, "amount_step": None}


def _flat_close(n=200, cols=("A", "B", "C"), seed=0):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, len(cols))), axis=0)), index=idx, columns=list(cols))


# ---------------------------------------------------------------- adaptör
def test_edge_threshold_trades_only_above_cost_and_follows_sign():
    close = _flat_close()
    idx, cols = close.index, close.columns
    pred = pd.DataFrame({"A": 1.0, "B": -1.0, "C": 0.001}, index=idx)  # C: maliyetin altında
    member = pd.DataFrame(True, index=idx, columns=cols)
    w = signals.make_weights("edge_threshold", pred, member, _ctx(close), h=7)
    late = w.iloc[-30:]
    assert (late["A"] > 0).all() and (late["B"] < 0).all() and (late["C"] == 0).all()
    assert (w.abs() <= 0.10 + 1e-12).all().all()  # isim tavanı (limitsiz: base_cap)
    assert (w.abs().sum(axis=1) <= 2.0 + 1e-12).all()
    w_hi = signals.make_weights("edge_threshold", pred, member, _ctx(close), h=7, k=1e6)  # eşik çok yüksek -> pozisyon yok
    assert (w_hi == 0).all().all()
    member2 = member.copy()
    member2["A"] = False
    assert (signals.make_weights("edge_threshold", pred, member2, _ctx(close), h=7)["A"] == 0).all()


def test_edge_threshold_cost_scales_with_slippage_and_vol():
    close = _flat_close()
    pred = pd.DataFrame(0.05, index=close.index, columns=close.columns)
    member = pd.DataFrame(True, index=close.index, columns=close.columns)
    cheap = signals.make_weights("edge_threshold", pred, member, _ctx(close, 0.0), h=7)
    dear = signals.make_weights("edge_threshold", pred, member, _ctx(close, 500.0), h=7)
    assert (cheap.iloc[-20:].abs().sum(axis=1) > 0).all() and (dear == 0).all().all()


def test_edge_threshold_uses_no_future_prices():
    close = _flat_close(seed=3)
    rng = np.random.default_rng(4)
    pred = pd.DataFrame(rng.normal(0, 0.5, close.shape), index=close.index, columns=close.columns)
    member = pd.DataFrame(True, index=close.index, columns=close.columns)
    cut = close.index[120]
    changed = close.copy()
    changed.loc[cut + pd.Timedelta(days=1):] *= 1.7
    a = signals.make_weights("edge_threshold", pred, member, _ctx(close), h=7)
    b = signals.make_weights("edge_threshold", pred, member, _ctx(changed), h=7)
    pd.testing.assert_frame_equal(a.loc[:cut], b.loc[:cut])


def test_context_required_for_edge_threshold_and_other_adapters_unchanged():
    close = _flat_close()
    pred = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    member = pd.DataFrame(True, index=close.index, columns=close.columns)
    with pytest.raises(ValueError, match="context"):
        signals.make_weights("edge_threshold", pred, member, h=7)
    w = signals.make_weights("rank_long_short", pred.rank(axis=1), member, q=0.34)  # eski çağrı biçimi çalışır
    assert w.abs().sum(axis=1).iloc[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------- ızgara ve koşucu
def test_grid_has_18_variants_with_preregistered_plateau():
    exps = ml_kol.experiments()
    ids = [m["id"] for e in exps for m in e["models"]]
    assert len(ids) == len(set(ids)) == ml_kol.N_PLANNED == 18
    assert sum(1 for e in exps if e["kind"] == "base") == 4 and len(exps) == 7
    lg = next(e for e in exps if e["id"] == "ml_kol_plato_lgbm")
    params = {m["id"]: m["params"] for m in lg["models"]}
    assert params["plato|num_leaves×0.5"] == {"num_leaves": 8} and params["plato|num_leaves×1.5"] == {"num_leaves": 22}
    assert params["plato|learning_rate×0.5"]["learning_rate"] == pytest.approx(0.015)
    ks = [e["signal"]["k"] for e in exps if e["id"].startswith("ml_kol_plato_k")]
    assert ks == [0.5, 1.5]
    cfg = runner.load_config(ml_kol.make_config(exps[0]))
    assert cfg["execution"]["band"] == 0.25 and cfg["cv"]["embargo_days"] == 7 and cfg["signal"]["params"]["h"] == 7


def test_runner_execution_band_default_and_validation():
    base = ml_kol.make_config(ml_kol.experiments()[0])
    base.pop("execution")
    assert runner.load_config(base)["execution"]["band"] == 0.0
    with pytest.raises(runner.ConfigError, match="execution.band"):
        runner.load_config({**base, "execution": {"band": 1.5}})


# ---------------------------------------------------------------- uçtan uca (sentetik; geçici sayaç)
def test_run_study_end_to_end_registers_18_variants(tmp_path, monkeypatch):
    from research import budget

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    write_market(make_market())
    monkeypatch.setattr(ml_kol, "TIERS", {"a": ["ohlcv_core"], "b": ["ohlcv_core"], "d": ["ohlcv_core"]})
    monkeypatch.setattr(ml_kol, "PERIOD", ("2021-07-01", "2023-08-31"))
    monkeypatch.setattr(ml_kol, "N_SPLITS", 3)
    monkeypatch.setattr(ml_kol, "MIN_TRAIN_DAYS", 200)
    monkeypatch.setattr(ml_kol, "UNIVERSE_N", 10)
    monkeypatch.setattr(ml_kol, "N_BOOT", 50)
    paths = dict(results_dir=tmp_path / "results", log_path=tmp_path / "deneyler.md", registry_file=tmp_path / "results" / "_registry.json")
    paths["log_path"].write_text("# Deney günlüğü\n\n**Toplam deneme: 0**\n\n| t |\n|---|\n", encoding="utf-8")
    tri = pd.date_range("2021-01-01", "2023-09-30", freq="D")
    tr = pd.DataFrame({c: np.random.default_rng(1).normal(0, 0.01, len(tri)) for c in ("main_LF_wf", "main_LS_wf")}, index=tri)
    s = ml_kol.run_study(**paths, configs_dir=tmp_path / "configs", trend_returns=tr, progress=lambda *_: None)
    assert s["trials_after"] == 18 and len(s["table"]) == 12 and len(s["plateau"]["neighbors"]) == 6
    assert [r["model"] for r in s["table"]][:4] == ["ridge"] * 4  # sadelik sırası
    assert all(r["vs_ridge"] is None for r in s["table"] if r["model"] == "ridge")
    assert all(r["vs_ridge"] is not None for r in s["table"] if r["model"] != "ridge")
    assert len(s["tiers"]) == 6
    assert (paths["results_dir"] / "ml_kol" / "sonuc.md").exists() and (paths["results_dir"] / "ml_kol_a_h7" / "metrics.json").exists()
    assert "SONUÇ — `ml_kol`" in paths["log_path"].read_text(encoding="utf-8")
    assert (tmp_path / "configs" / "ml_kol_kazanan.yaml").exists() == s["decision"].startswith("ML KOLU")
    assert budget.variants_used("ml_kol", paths["registry_file"]) == 18
    with pytest.raises(budget.BudgetExceededError):
        budget.assert_budget("ml_kol", 23, paths["registry_file"])  # 18 + 23 > 40
    with pytest.raises(RuntimeError, match="tamamlandı"):
        ml_kol.run_study(**paths, configs_dir=tmp_path / "configs", trend_returns=tr, progress=lambda *_: None)
    # devam modu: kayıtlar yazıldıktan sonra özet aşamasında yarıda kalmış gibi -> yeniden koşu kaydı TEKRARLAMAZ, aynı kararı verir
    (paths["results_dir"] / "ml_kol" / "sonuc.md").unlink()
    s2 = ml_kol.run_study(**paths, configs_dir=tmp_path / "configs", trend_returns=tr, progress=lambda *_: None)
    assert s2["trials_after"] == 18 and s2["decision"] == s["decision"]
    assert [r["sharpe"] for r in s2["table"]] == pytest.approx([r["sharpe"] for r in s["table"]], nan_ok=True)


def test_edge_threshold_warmup_has_no_positions_before_vol_span():
    close = _flat_close(n=300)
    pred = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    pred.iloc[100:] = 1.0  # OOF 100. günde başlar
    member = pd.DataFrame(True, index=close.index, columns=close.columns)
    w = signals.make_weights("edge_threshold", pred, member, _ctx(close), h=7, vol_span=60)
    assert (w.iloc[:100 + 59] == 0).all().all()  # ilk geçerli tahminden sonra 60 gözlem birikene kadar pozisyon yok
    assert (w.iloc[100 + 60:].abs().sum(axis=1) > 0).all()


def test_tier_d_feature_check_stops_before_registration():
    class P:
        def __init__(self, cols):
            self.X = pd.DataFrame({c: [1.0, np.nan, 1.0] for c in cols})

    full = [f"macro__{s}_z" for s in ml_kol.MACRO_SERIES] + ["sentiment_fng__fng"]
    ml_kol.check_tier_features({("d", 7): P(full), ("b", 7): P([])})
    with pytest.raises(ml_kol.TierDataError, match="eksik"):
        ml_kol.check_tier_features({("d", 7): P(full[1:])})
    sparse = P(full)
    sparse.X["macro__dax_z"] = [np.nan, np.nan, 1.0]
    with pytest.raises(ml_kol.TierDataError, match="doluluk"):
        ml_kol.check_tier_features({("d", 7): sparse})
