"""trend deneyi: sinyallerde ileri bakış yok, boyutlandırma tavanları, Donchian durum makinesi, walk-forward yalnızca geçmişle
seçer, motorun göreli bant / dengeleme günleri / gün bazında bant desteği, sentetik veride uçtan uca koşu."""

import numpy as np
import pandas as pd
import pytest

from research import engine, trend

IDX = pd.date_range("2020-01-01", "2025-09-30", freq="D")


def _prices(n_sym=8, seed=0):
    rng = np.random.default_rng(seed)
    cols = ["BTCUSDT", "ETHUSDT"] + [f"S{i:02d}USDT" for i in range(n_sym - 2)]
    drift = rng.normal(0.0004, 0.0004, n_sym)
    ret = rng.normal(drift, 0.03, size=(len(IDX), n_sym))
    close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=IDX, columns=cols)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    return {"open": open_, "high": high, "low": low, "close": close}


def _market(n_sym=8):
    prices = _prices(n_sym)
    cols = prices["close"].columns
    memberships = {}
    for n in trend.UNIVERSES:
        mem = pd.DataFrame(False, index=IDX, columns=cols)
        mem.loc["2020-04-01":, cols[: min(n, len(cols))]] = True
        memberships[n] = mem
    slip = pd.DataFrame(5.0, index=IDX, columns=cols)
    return trend.Market(prices, memberships, slip, None, None, None, [], "synthetic")


# ---------------------------------------------------------------- ileri bakış yok
def test_signals_and_weights_do_not_use_future_prices():
    p = _prices()
    cut = pd.Timestamp("2023-06-30")
    p2 = {k: v.copy() for k, v in p.items()}
    for k in p2:
        p2[k].loc[cut + pd.Timedelta(days=1):] *= np.random.default_rng(1).uniform(0.5, 1.5, p2[k].loc[cut + pd.Timedelta(days=1):].shape)
    s1, s2 = trend.all_signals(p), trend.all_signals(p2)
    mem = pd.DataFrame(True, index=IDX, columns=p["close"].columns)
    for name in s1:
        pd.testing.assert_frame_equal(s1[name].loc[:cut], s2[name].loc[:cut], check_exact=False)
        for d in trend.DIRECTIONS:
            w1 = trend.size_weights(s1[name], p["close"], mem, d).loc[:cut]
            w2 = trend.size_weights(s2[name], p2["close"], mem, d).loc[:cut]
            pd.testing.assert_frame_equal(w1, w2, check_exact=False)


def test_signal_ranges_and_main_blend_uses_available_components():
    p = _prices()
    s = trend.all_signals(p)
    for name, f in s.items():
        v = f.to_numpy()[np.isfinite(f.to_numpy())]
        assert v.min() >= -1 - 1e-12 and v.max() <= 1 + 1e-12, name
    # 250 günlük bileşen henüz yokken ana aday yine tanımlı (mevcut bileşenlerin ortalaması)
    day = pd.Timestamp("2020-05-15")
    assert np.isnan(s["a250"].loc[day]).all() and np.isfinite(s["main"].loc[day]).all()
    comps = trend.main_components(p["close"])
    expected = np.nanmean(np.stack([c.loc[day].to_numpy() for c in comps]), axis=0)
    np.testing.assert_allclose(s["main"].loc[day].to_numpy(), expected)


def test_donchian_state_machine_enters_holds_and_exits():
    idx = pd.date_range("2021-01-01", periods=12, freq="D")
    c = pd.DataFrame({"X": [10, 10, 10, 11, 12, 11.5, 11, 9, 9, 8, 7, 8.5]}, index=idx, dtype=float)
    sig = trend.sig_donchian(c, c, c, n_in=3, n_out=2)["X"]
    assert np.isnan(sig.iloc[:3]).all()  # ısınma
    assert sig.loc["2021-01-04"] == 1.0  # 11 > önceki 3 günün en yükseği (10) -> long
    assert sig.loc["2021-01-06"] == 1.0  # 11.5: önceki 2 günün en düşüğü (11) üstünde -> tut
    assert sig.loc["2021-01-07"] == 0.0  # 11 < min(12, 11.5) -> çıkış; 11, önceki 3 günün en düşüğünün (11) altında değil -> düz
    assert sig.loc["2021-01-08"] == -1.0  # 9 < min(12, 11.5, 11) -> short
    assert sig.loc["2021-01-11"] == -1.0  # 7, önceki 2 günün en yükseğinin (9) altında -> short sürer
    assert sig.loc["2021-01-12"] == 0.0  # 8.5 > max(8, 7) -> short çıkışı; giriş koşulu yok -> düz


def test_size_weights_respect_caps_and_direction():
    p = _prices()
    s = trend.all_signals(p)["main"]
    mem = pd.DataFrame(True, index=IDX, columns=p["close"].columns)
    for d in trend.DIRECTIONS:
        w = trend.size_weights(s, p["close"], mem, d)
        assert (w.abs() <= trend.ASSET_CAP + 1e-9).all().all()
        assert (w.abs().sum(axis=1) <= trend.GROSS_CAP[d] + 1e-9).all()
        if d == "LF":
            assert (w >= 0).all().all()
    # evren dışı sembol ağırlık almaz
    mem2 = mem.copy()
    mem2["ETHUSDT"] = False
    assert (trend.size_weights(s, p["close"], mem2, "LS")["ETHUSDT"] == 0).all()


# ---------------------------------------------------------------- walk-forward
def test_selection_uses_only_returns_before_the_fold():
    rng = np.random.default_rng(3)
    combos = {(n, b): pd.Series(rng.normal(0.001 * i, 0.01, len(IDX)), index=IDX)
              for i, (n, b) in enumerate([(n, b) for n in trend.UNIVERSES for b in trend.BANDS])}
    fold = pd.Timestamp("2022-04-01")
    pick = trend.select_combo(combos, fold)
    tampered = {k: v.copy() for k, v in combos.items()}
    for k in tampered:
        tampered[k].loc[fold - pd.Timedelta(days=1):] = rng.normal(0, 0.05, len(tampered[k].loc[fold - pd.Timedelta(days=1):]))
    assert trend.select_combo(tampered, fold) == pick


def test_stitch_targets_switches_at_decision_day_before_fold():
    a = pd.DataFrame(1.0, index=IDX, columns=["X"])
    b = pd.DataFrame(2.0, index=IDX, columns=["X"])
    f1, f2 = pd.Timestamp("2021-04-01"), pd.Timestamp("2021-10-01")
    tgt, bands = trend.stitch_targets({(10, 0.0): a, (20, 0.1): b}, [(f1, (10, 0.0)), (f2, (20, 0.1))])
    assert tgt.loc["2021-09-29", "X"] == 1.0 and tgt.loc["2021-09-30", "X"] == 2.0  # karar 09-30 kapanışı -> 10-01 açılışı
    assert bands.loc["2021-09-29"] == 0.0 and bands.loc["2021-09-30"] == 0.1


# ---------------------------------------------------------------- motor eklentileri
def _flat_px(n=6):
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.DataFrame({"A": 100.0}, index=idx), idx


def test_engine_relative_band_skips_small_relative_changes_but_always_exits():
    px, idx = _flat_px()
    tgt = pd.DataFrame({"A": [0.10, 0.105, 0.12, 0.0, 0.0, 0.0]}, index=idx)
    res = engine.run(tgt, px, fee_rate=0.0, slippage_bps=0.0, band=0.10, band_relative=True)
    held = res.weights_held["A"].tolist()
    assert held[1] == pytest.approx(0.10)  # ilk giriş
    assert held[2] == pytest.approx(0.10)  # hedef 0.105: |Δ| 0.005 ≤ 0.1·0.105 -> işlem yok
    assert held[3] == pytest.approx(0.12)  # 0.12 vs 0.10: |Δ| 0.02 > 0.012 -> işlem
    assert held[4] == pytest.approx(0.0)  # çıkış her zaman


def test_engine_rebalance_days_and_per_day_band():
    px, idx = _flat_px()
    tgt = pd.DataFrame({"A": [0.1, 0.2, 0.3, 0.4, 0.5, 0.5]}, index=idx)
    res = engine.run(tgt, px, fee_rate=0.0, slippage_bps=0.0, rebalance_days=[idx[1], idx[4]])
    assert res.weights_held["A"].tolist() == pytest.approx([0.0, 0.1, 0.1, 0.1, 0.4])
    bands = pd.Series([0.0, 0.0, 0.5, 0.5, 0.0, 0.0], index=idx)
    res2 = engine.run(tgt, px, fee_rate=0.0, slippage_bps=0.0, band=bands)
    assert res2.weights_held["A"].tolist() == pytest.approx([0.0, 0.1, 0.1, 0.1, 0.4])  # bant 0.5 günlerinde işlem yok


# ---------------------------------------------------------------- uçtan uca (sentetik)
def test_run_study_end_to_end_on_synthetic_market(tmp_path):
    paths = dict(results_dir=tmp_path / "results", log_path=tmp_path / "deneyler.md", registry_file=tmp_path / "results" / "_registry.json")
    paths["log_path"].write_text("# Deney günlüğü\n\n**Toplam deneme: 0**\n\n| t |\n|---|\n", encoding="utf-8")
    s = trend.run_study(market=_market(), progress=lambda *_: None, **paths)
    assert s["n_variants"] == 216 + 32 and s["trials_after"] == 248
    assert len(s["wf"]) == 24 and 0.0 <= s["pbo"] <= 1.0
    assert {r["dir"] for r in s["wf"]} == {"LF", "LS"}
    out = paths["results_dir"] / trend.EXPERIMENT_ID
    for f in ("metrics.json", "sonuc.md", "varyantlar_216.md", "turnover_getiri.png", "main_LF/report.md", "ozet.json"):
        assert (out / f).exists(), f
    log = paths["log_path"].read_text(encoding="utf-8")
    assert "SONUÇ — `trend_001`" in log and "**Toplam deneme: 248**" in log
    assert s["stress"]["main LF (WF)"]["2020-03"]["max_dd"] is None  # pencere öncesi: değerlendirilemez


# ---------------------------------------------------------------- denetim düzeltmeleri: raporlama metrikleri
def test_leg_contrib_assigns_exit_costs_to_their_leg_and_sums_to_total():
    idx = pd.date_range("2022-01-01", periods=8, freq="D")
    px = pd.DataFrame({"A": [100, 101, 99, 102, 100, 103, 101, 100.0], "B": [50, 49, 51, 50, 52, 51, 50, 49.0]}, index=idx)
    tgt = pd.DataFrame({"A": [0.2, 0.2, 0.0, -0.2, -0.2, 0.0, 0.0, 0.0], "B": [-0.1, -0.1, -0.1, 0.0, 0.1, 0.1, 0.0, 0.0]}, index=idx)
    res = engine.run(tgt, px, fee_rate=0.001, slippage_bps=10.0)
    legs = trend.leg_contrib(res)
    assert legs["long"] + legs["short"] == pytest.approx(legs["total"])
    assert legs["total"] == pytest.approx(float(res.returns.sum()))  # katkılar toplamı = günlük getiriler toplamı


def test_skipped_rate_counts_order_attempts_not_position_days():
    idx = pd.date_range("2022-01-01", periods=6, freq="D")
    px = pd.DataFrame({"A": [100, 110, 120, 130, 140, 150.0], "B": [10.0] * 6}, index=idx)
    tgt = pd.DataFrame({"A": [0.5] * 6, "B": [0.001] * 6}, index=idx)  # B: 1.000 NAV'da 1 USDT < 5 min notional -> her gün atlanır
    res = engine.run(tgt, px, fee_rate=0.0, slippage_bps=0.0, account_size=1_000.0, min_notional=pd.Series({"A": 5.0, "B": 5.0}))
    executed = int(res.diagnostics["trades_per_day"].sum())
    skipped = len(res.skipped)
    assert skipped == 4 and executed >= 1  # A drift'le her gün küçük düzeltme alır; B her gün denenip atlanır
    assert trend.skipped_rate(res) == pytest.approx(skipped / (skipped + executed))
