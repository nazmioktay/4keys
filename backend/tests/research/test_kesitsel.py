"""kesitsel deneyi: skorda ileri bakış yok ve son gün hariç, dilimler, portföy yapıları, BTC beta hedge, dengeleme günleri,
zarar limiti + motorun işlem maskesi, walk-forward birleştirme, karışım, sentetik veride uçtan uca kayıtlı koşu (37 varyant)."""

import numpy as np
import pandas as pd
import pytest

from research import engine, kesitsel, trend

IDX = pd.date_range("2020-01-01", "2025-09-30", freq="D")


def _prices(n_sym=14, seed=0):
    rng = np.random.default_rng(seed)
    cols = ["BTCUSDT", "ETHUSDT"] + [f"S{i:02d}USDT" for i in range(n_sym - 2)]
    drift = rng.normal(0.0003, 0.0004, n_sym)
    ret = rng.normal(drift, 0.03, size=(len(IDX), n_sym))
    close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=IDX, columns=cols)
    open_ = close.shift(1).fillna(close.iloc[0])
    return {"open": open_, "high": np.maximum(open_, close) * 1.01, "low": np.minimum(open_, close) * 0.99, "close": close}


def _market(n_sym=14):
    prices = _prices(n_sym)
    cols = prices["close"].columns
    mem = pd.DataFrame(False, index=IDX, columns=cols)
    mem.loc["2020-04-01":, :] = True
    memberships = {n: mem.copy() for n in (10, *kesitsel.UNIVERSES)}
    slip = pd.DataFrame(5.0, index=IDX, columns=cols)
    return trend.Market(prices, memberships, slip, None, None, None, [], "synthetic")


def _mem(close):
    return pd.DataFrame(True, index=close.index, columns=close.columns)


# ---------------------------------------------------------------- skor
def test_score_uses_no_future_and_skips_the_last_day():
    p = _prices()
    close = p["close"]
    t = pd.Timestamp("2023-06-30")
    changed = close.copy()
    changed.loc[t:] *= np.random.default_rng(1).uniform(0.5, 1.5, changed.loc[t:].shape)  # t dahil sonrası değişir
    for vol in (False, True):
        a, b = kesitsel.score(close, _mem(close), vol), kesitsel.score(changed, _mem(close), vol)
        if not vol:  # düz skor t kapanışını kullanmaz (son gün hariç) -> t'de aynı
            pd.testing.assert_frame_equal(a.loc[:t], b.loc[:t])
        else:  # vol-ayarlıda σ t'ye kadar veriyi kullanır (izinli); t−1'e kadar aynı
            pd.testing.assert_frame_equal(a.loc[: t - pd.Timedelta(days=1)], b.loc[: t - pd.Timedelta(days=1)])
    sc = kesitsel.score(close, _mem(close), False).loc["2022-01-01"]
    assert abs(sc.mean()) < 1e-9  # z-skor ortalaması kesitte 0


def test_score_matches_manual_formula_and_respects_membership():
    close = _prices()["close"]
    mem = _mem(close)
    mem["ETHUSDT"] = False
    sc = kesitsel.score(close, mem, False)
    assert sc["ETHUSDT"].isna().all()
    t = pd.Timestamp("2022-03-10")
    zs = []
    for L in kesitsel.LOOKBACKS:
        r = np.log(close.loc[t - pd.Timedelta(days=1)] / close.loc[t - pd.Timedelta(days=1 + L)]).drop("ETHUSDT")
        zs.append((r - r.mean()) / r.std(ddof=0))
    np.testing.assert_allclose(sc.loc[t].drop("ETHUSDT").to_numpy(), (sum(zs) / 3).to_numpy())


def test_legs_quintile_sizes():
    idx = pd.date_range("2022-01-01", periods=3, freq="D")
    sc = pd.DataFrame(np.arange(30, dtype=float)[None, :].repeat(3, 0), index=idx, columns=[f"S{i}" for i in range(30)])
    sc.iloc[1, :26] = np.nan  # 4 üye -> k = 1
    sc.iloc[2, :29] = np.nan  # 1 üye -> dilim yok
    top, bottom = kesitsel.legs(sc)
    assert top.iloc[0].sum() == 6 and bottom.iloc[0].sum() == 6
    assert top.iloc[0, -6:].all() and bottom.iloc[0, :6].all()
    assert top.iloc[1].sum() == 1 and bottom.iloc[1].sum() == 1
    assert top.iloc[2].sum() == 0 and bottom.iloc[2].sum() == 0


# ---------------------------------------------------------------- portföyler
def test_raw_portfolios_structure():
    p = _prices()
    close = p["close"]
    sc = kesitsel.score(close, _mem(close), False)
    a = kesitsel.raw_portfolio(sc, close, "a").loc["2022-01-01":"2022-12-31"]
    np.testing.assert_allclose(a.clip(lower=0).sum(axis=1), 1.0)
    np.testing.assert_allclose(a.clip(upper=0).sum(axis=1), -1.0)
    c = kesitsel.raw_portfolio(sc, close, "c").loc["2022-01-01":"2022-12-31"]
    assert (c >= 0).all().all()
    np.testing.assert_allclose(c.sum(axis=1), 1.0)


def test_btc_beta_recovers_known_beta_and_uses_only_past():
    idx = pd.date_range("2021-01-01", periods=300, freq="D")
    rng = np.random.default_rng(5)
    rb = rng.normal(0, 0.03, len(idx))
    rx = 1.5 * rb + rng.normal(0, 0.002, len(idx))
    close = pd.DataFrame({"BTCUSDT": 100 * np.cumprod(1 + rb), "X": 100 * np.cumprod(1 + rx)}, index=idx)
    long_raw = pd.DataFrame({"BTCUSDT": 0.0, "X": 1.0}, index=idx)
    beta = kesitsel.btc_beta(long_raw, close)
    assert beta.iloc[:60].isna().all()
    assert beta.iloc[-1] == pytest.approx(1.5, abs=0.05)
    changed = close.copy()
    changed.iloc[200:] *= 1.3
    pd.testing.assert_series_equal(beta.iloc[:199], kesitsel.btc_beta(long_raw, changed).iloc[:199])


def test_size_weights_caps_and_btc_hedge_exemption():
    p = _prices()
    close = p["close"]
    caps = pd.DataFrame(kesitsel.NAME_CAP, index=close.index, columns=close.columns)
    sc = kesitsel.score(close, _mem(close), False)
    for pf in kesitsel.PORTFOLIOS:
        w = kesitsel.size_weights(kesitsel.raw_portfolio(sc, close, pf), close, pf, caps)
        assert (w.abs().sum(axis=1) <= kesitsel.GROSS_CAP[pf] + 1e-9).all()
        others = w.drop(columns="BTCUSDT") if pf == "b" else w
        assert (others.abs() <= kesitsel.NAME_CAP + 1e-9).all().all()
    wb = kesitsel.size_weights(kesitsel.raw_portfolio(sc, close, "b"), close, "b", caps)
    assert (wb["BTCUSDT"].loc["2022":] < -kesitsel.NAME_CAP).any()  # hedge isim tavanına takılmaz


# ---------------------------------------------------------------- dengeleme, zarar limiti, motor maskesi
def test_rebalance_days():
    w = kesitsel.rebalance_days(IDX, "W")
    assert (w.dayofweek == 0).all() and w.min() >= trend.POSITION_START
    d3 = kesitsel.rebalance_days(IDX, "3D")
    assert d3[0] == trend.POSITION_START and (pd.Series(d3).diff().dropna() == pd.Timedelta(days=3)).all()


def test_apply_stop_closes_short_after_25pct_rise_until_next_rebalance():
    idx = pd.date_range("2022-01-01", periods=12, freq="D")
    open_ = pd.DataFrame({"X": 100.0, "Y": 100.0}, index=idx)
    close = open_.copy()
    close.loc["2022-01-05":"2022-01-07", "X"] = 126.0  # 4. günde %26 yükseliş (sonra geri döner)
    tgt = pd.DataFrame({"X": -0.1, "Y": -0.1}, index=idx)
    rebal = pd.DatetimeIndex(["2022-01-02", "2022-01-09"])
    out, mask = kesitsel.apply_stop(tgt, open_, close, rebal)
    assert out.loc["2022-01-04", "X"] == -0.1 and out.loc["2022-01-05", "X"] == 0.0 and out.loc["2022-01-07", "X"] == 0.0
    assert out.loc["2022-01-08", "X"] == -0.1  # sonraki dengelemenin kararı: yeni hedef geçerli
    assert (out["Y"] == -0.1).all() and not mask["Y"].any()
    assert not mask.loc["2022-01-05", "X"] and mask.loc["2022-01-06", "X"] and mask.loc["2022-01-08", "X"]
    assert not mask.loc["2022-01-09":, "X"].any()
    # motor: dengeleme dışı günde yalnızca maske hücresinde işlem
    res = engine.run(out, open_, fee_rate=0.0, slippage_bps=0.0, rebalance_days=rebal, trade_mask=mask)
    held = res.weights_held
    assert held.loc["2022-01-05", "X"] == pytest.approx(-0.1) and held.loc["2022-01-06", "X"] == 0.0
    assert held.loc["2022-01-06", "Y"] == pytest.approx(-0.1)
    assert held.loc["2022-01-09", "X"] == pytest.approx(-0.1)  # sonraki dengelemede yeniden açılır


def test_engine_trade_mask_without_rebalance_days():
    idx = pd.date_range("2022-01-01", periods=5, freq="D")
    px = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)
    tgt = pd.DataFrame({"A": 0.1, "B": 0.1}, index=idx)
    mask = pd.DataFrame({"A": True, "B": False}, index=idx)
    res = engine.run(tgt, px, fee_rate=0.0, slippage_bps=0.0, trade_mask=mask)
    assert res.weights_held["A"].iloc[1] == pytest.approx(0.1) and (res.weights_held["B"] == 0).all()


# ---------------------------------------------------------------- walk-forward
def test_stitch_takes_target_rebalance_and_mask_from_the_selected_combo():
    idx = pd.date_range("2021-03-01", "2021-12-31", freq="D")
    cols = ["X"]
    plans = {
        (30, "W"): kesitsel.Plan(pd.DataFrame(1.0, index=idx, columns=cols), kesitsel.rebalance_days(idx, "W")),
        (50, "3D"): kesitsel.Plan(pd.DataFrame(2.0, index=idx, columns=cols), kesitsel.rebalance_days(idx, "3D"),
                                  pd.DataFrame(True, index=idx, columns=cols)),
    }
    f1, f2 = pd.Timestamp("2021-04-01"), pd.Timestamp("2021-10-01")
    p = kesitsel.stitch(plans, [(f1, (30, "W")), (f2, (50, "3D"))])
    assert p.target.loc["2021-09-29", "X"] == 1.0 and p.target.loc["2021-09-30", "X"] == 2.0
    before = p.rebal[(p.rebal >= "2021-06-01") & (p.rebal <= "2021-09-30")]
    after = p.rebal[p.rebal >= "2021-10-01"]
    assert (before.dayofweek == 0).all() and after.isin(plans[(50, "3D")].rebal).all()
    assert not p.mask.loc["2021-09-30", "X"] and p.mask.loc["2021-10-01", "X"]


def test_select_combo_uses_only_past():
    rng = np.random.default_rng(3)
    combos = {c: pd.Series(rng.normal(0.001 * i, 0.01, len(IDX)), index=IDX) for i, c in enumerate(kesitsel.COMBOS)}
    fold = pd.Timestamp("2022-04-01")
    pick = kesitsel.select_combo(combos, fold)
    tampered = {k: v.copy() for k, v in combos.items()}
    for k in tampered:
        tampered[k].loc[fold - pd.Timedelta(days=1):] = rng.normal(0, 0.05, len(tampered[k].loc[fold - pd.Timedelta(days=1):]))
    assert kesitsel.select_combo(tampered, fold) == pick


# ---------------------------------------------------------------- karışım ve korelasyon
def test_equal_risk_mix_weights_inverse_to_past_vol():
    idx = pd.date_range("2022-01-01", periods=200, freq="D")
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    b = pd.Series(rng.normal(0, 0.03, len(idx)), index=idx)
    mix = kesitsel.equal_risk_mix(a, b)
    assert mix.iloc[0] == pytest.approx(0.5 * a.iloc[0] + 0.5 * b.iloc[0])  # vol yokken eşit
    va, vb = a.iloc[100:160].std(), b.iloc[100:160].std()
    wa = (1 / va) / (1 / va + 1 / vb)
    assert mix.iloc[160] == pytest.approx(wa * a.iloc[160] + (1 - wa) * b.iloc[160])


# ---------------------------------------------------------------- uçtan uca (sentetik, kayıtlı: geçici sayaç)
def test_run_study_end_to_end_registers_37_variants(tmp_path):
    from research import budget

    paths = dict(results_dir=tmp_path / "results", log_path=tmp_path / "deneyler.md", registry_file=tmp_path / "results" / "_registry.json")
    paths["log_path"].write_text("# Deney günlüğü\n\n**Toplam deneme: 0**\n\n| t |\n|---|\n", encoding="utf-8")
    m = _market()
    idx = pd.date_range(trend.EVAL_START, "2025-09-29", freq="D")
    rng = np.random.default_rng(9)
    tr = pd.DataFrame({c: rng.normal(0, 0.01, len(idx)) for c in kesitsel.TREND_SERIES}, index=idx)
    s = kesitsel.run_study(market=m, trend_returns=tr, progress=lambda *_: None, register=True, **paths)
    assert s["n_variants"] == kesitsel.N_PLANNED == 37 and s["trials_after"] == 37
    assert len(s["wf"]) == 7 and 0.0 <= s["pbo"] <= 1.0
    assert set(s["corr_trend"]) == set(kesitsel.TREND_SERIES)
    assert len(s["horizon"]) == 60 and len(s["plateau"]["neighbors"]) == 8
    out = paths["results_dir"] / kesitsel.EXPERIMENT_ID
    for f in ("sonuc.md", "varyantlar.md", "getiri_ufuk.png", "ana_aday/report.md", "ozet.json", "metrics.json", "daily_returns.parquet"):
        assert (out / f).exists(), f
    assert "SONUÇ — `kesitsel_001`" in paths["log_path"].read_text(encoding="utf-8")
    assert pd.read_parquet(out / "daily_returns.parquet").shape[1] == 7 + 24 + 4 + 8 + 1  # 7 WF serisi + 37 deneme
    with pytest.raises(budget.BudgetExceededError):
        kesitsel.run_study(market=m, trend_returns=tr, progress=lambda *_: None, register=True, **{**paths, "results_dir": tmp_path / "r2"})


def test_delay_shifts_rebalance_days_and_stop_mask_together():
    """+1 gün senaryosu: hedef, dengeleme günleri ve maske birlikte kayar; aynı karar bir gün geç işlem görür ve maskeli günde
    sıfırlanmamış eski hedefe (ör. ters işaretli) işlem yapılmaz (reviewer bulgusu)."""
    idx = pd.date_range("2022-01-01", periods=14, freq="D")
    open_ = pd.DataFrame({"X": 100.0, "Y": 100.0}, index=idx)
    close = open_.copy()
    close.loc["2022-01-05":"2022-01-07", "X"] = 126.0
    tgt = pd.DataFrame({"X": -0.1, "Y": -0.1}, index=idx)
    tgt.loc["2022-01-04", "X"] = 0.1  # günlük hedefin işareti tetikten önceki gün dönüyor
    rebal = pd.DatetimeIndex(["2022-01-02", "2022-01-09"])
    out, mask = kesitsel.apply_stop(tgt, open_, close, rebal)
    prices = {"open": open_, "high": open_, "low": open_, "close": close}
    m = trend.Market(prices, {}, pd.DataFrame(0.0, index=idx, columns=open_.columns), None, None, None, [], "t")
    plan = kesitsel.Plan(out, rebal, mask)
    h0 = kesitsel.run_engine(m, plan).weights_held["X"]
    h1 = kesitsel.run_engine(m, plan, delay=1).weights_held["X"]
    assert h0.loc["2022-01-06"] == 0.0 and h1.loc["2022-01-06"] != 0.0 and h1.loc["2022-01-07"] == 0.0  # bir gün geç kapanır
    assert (h1 <= 0).all() and (h0 <= 0).all()  # hiçbir senaryoda long açılmaz
    assert h1.loc["2022-01-02"] == 0.0 and h1.loc["2022-01-03"] == pytest.approx(-0.1)  # ilk giriş de bir gün geç
