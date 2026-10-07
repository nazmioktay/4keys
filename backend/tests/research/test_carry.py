"""carry deneyi: bacak riski formülü, A'da funding geliri ve bağlanan sermaye, tasfiye ve yeniden hedge, cüzdan dengeleme,
B'nin giriş/çıkış/K/hacim kuralları ve ileri bakışsızlığı, C'de basis'in vadede kilitlenmesi ve erken geçiş, gecikme, uçtan uca
kayıtlı koşu (36 varyant)."""

import math

import numpy as np
import pandas as pd
import pytest

from research import carry

IDX = pd.date_range("2020-01-01", "2025-09-30", freq="D")


def _data(n_alt=5, seed=0, funding=None, flat=False, idx=IDX):
    rng = np.random.default_rng(seed)
    cols = ["BTCUSDT", "ETHUSDT"] + [f"S{i:02d}USDT" for i in range(n_alt)]
    if flat:
        close = pd.DataFrame(100.0, index=idx, columns=cols)
    else:
        ret = rng.normal(0.0003, 0.02, size=(len(idx), len(cols)))
        close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=idx, columns=cols)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) * 1.005
    qv = pd.DataFrame(1e9, index=idx, columns=cols)
    perp = {"open": open_, "high": high, "close": close, "quote_volume": qv}
    spot = {"open": open_.copy(), "close": close.copy(), "quote_volume": qv.copy()}
    if funding is None:
        funding = pd.DataFrame(0.0003, index=idx, columns=cols)  # günlük %0,03 (yıllık ~%11)
    mem = pd.DataFrame(True, index=idx, columns=cols)
    slip = pd.DataFrame(5.0, index=idx, columns=cols)
    sig = pd.DataFrame(0.005, index=idx, columns=cols)
    d = carry.Data(idx, spot, perp, funding, funding.abs(), slip, sig, mem)
    return d


# ---------------------------------------------------------------- formüller
def test_leg_risk_formula_and_roundtrip_cost():
    assert carry.leg_risk(0.01, 1) == pytest.approx(0.01 * math.sqrt(1 / 60) * math.sqrt(2 / math.pi))
    assert carry.leg_risk(0.01, 5) == pytest.approx(carry.leg_risk(0.01, 1) * math.sqrt(5))
    d = _data()
    rt = carry.roundtrip_cost(d).iloc[10, 0]
    assert rt == pytest.approx(2 * (0.001 + 0.0005 + 2 * 0.0005 + carry.leg_risk(0.005, 1)))


# ---------------------------------------------------------------- A: funding geliri ve sermaye
@pytest.mark.parametrize("lev", [1, 3])
def test_a_flat_prices_earns_funding_on_notional(lev):
    idx = pd.date_range("2021-01-01", periods=60, freq="D")
    d = _data(flat=True, idx=idx)
    res = carry.simulate(d, carry.StratA("BTCUSDT"), lev=lev)
    notional = 250 / (1 + 1 / lev)
    day = res.components["funding"].iloc[5]
    assert day == pytest.approx(notional * 0.0003, rel=0.01)  # lot yok: tüm sermaye kullanılır (maliyet payı kadar eksik)
    assert res.components["basis"].abs().sum() == pytest.approx(0.0, abs=1e-9)
    assert res.bound.iloc[5] == pytest.approx(250, rel=0.01)
    assert res.liquidations == 0 and res.trades == 1


def test_a_liquidation_on_intraday_spike_then_rehedge():
    idx = pd.date_range("2021-01-01", periods=30, freq="D")
    d = _data(flat=True, idx=idx)
    d.perp["high"].loc["2021-01-10", "BTCUSDT"] = 140.0  # gün içi +%40: 3x marjı (%33) yetmez
    res3 = carry.simulate(d, carry.StratA("BTCUSDT"), lev=3)
    res1 = carry.simulate(d, carry.StratA("BTCUSDT"), lev=1)
    assert res3.liquidations == 1 and res1.liquidations == 0
    assert res3.nav.loc["2021-01-10"] < res3.nav.loc["2021-01-09"] * 0.8  # vadeli cüzdan kaybı (hedef marj 250/4 ≈ 62)
    assert res3.trades >= 2  # ertesi gün yeniden hedge (dengeleme)
    assert res3.components["funding"].loc["2021-01-12"] > 0  # hedge yeniden kuruldu


def test_a_margin_band_rebalances_when_price_trends():
    idx = pd.date_range("2021-01-01", periods=80, freq="D")
    d = _data(flat=True, idx=idx)
    path = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    for k in ("open", "close", "high"):
        d.perp[k]["BTCUSDT"] = path
    for k in ("open", "close"):
        d.spot[k]["BTCUSDT"] = path
    res = carry.simulate(d, carry.StratA("BTCUSDT"), lev=2)
    assert res.liquidations == 0 and res.trades > 1  # marj %50 bandının altına düşünce yeniden boyutlanır
    assert abs(res.nav.iloc[-1] / 250 - 1) < 0.05  # hedge'li: fiyat iki katına çıksa da NAV ~sabit (yalnızca funding/maliyet)


# ---------------------------------------------------------------- B
def _b_data():
    idx = pd.date_range("2021-01-01", periods=120, freq="D")
    cols_f = pd.DataFrame(0.0, index=idx, columns=["BTCUSDT", "ETHUSDT"] + [f"S{i:02d}USDT" for i in range(5)])
    cols_f["S00USDT"] = 0.003  # yıllık ~%110: eşiği aşar
    cols_f["S01USDT"] = 0.002
    cols_f["S02USDT"] = 0.0004  # yıllık ~%15: eşiğin altında
    cols_f.loc["2021-03-01":, "S00USDT"] = 0.00005  # yıllık ~%1,8 < %3 -> çıkış
    return _data(flat=True, idx=idx, funding=cols_f)


def test_b_entry_exit_and_k_limit():
    d = _b_data()
    s = carry.StratB(d, K=1, W=3)
    j = 20
    assert set(s.decide(d, j, set(), 250)) == {("spot", "S00USDT", "perp", "S00USDT")}  # K=1: en yüksek
    s2 = carry.StratB(d, K=3, W=3)
    assert {k[1] for k in s2.decide(d, j, set(), 250)} == {"S00USDT", "S01USDT"}  # S02 eşiğin altında
    held = {("spot", "S00USDT", "perp", "S00USDT")}
    j_exit = d.idx.get_loc(pd.Timestamp("2021-03-05"))
    assert ("spot", "S00USDT", "perp", "S00USDT") not in s2.decide(d, j_exit, held, 250)  # histerezis: < %3 -> çık
    res = carry.simulate(d, s2, lev=1)
    assert any(e["symbol"] == "S00USDT" and e["reason"] == "sinyal" for e in res.episodes)


def test_b_volume_cap_and_no_lookahead():
    d = _b_data()
    d.perp["quote_volume"]["S00USDT"] = 20_000.0  # %0,1 -> 20 USDT tavan
    s = carry.StratB(d, K=1, W=3)
    assert s.cap(d, 40, ("spot", "S00USDT", "perp", "S00USDT")) == pytest.approx(20.0)
    res = carry.simulate(d, s, lev=1)
    assert 0 < res.bound.loc["2021-02-01"] <= 41.0  # 1x: notional ≤ 20 + marj ≤ 20
    d2 = _b_data()
    d2.funding.loc["2021-02-15":] = 0.01  # gelecek değişir
    a, b = carry.StratB(d, 3, 7), carry.StratB(d2, 3, 7)
    j = d.idx.get_loc(pd.Timestamp("2021-02-14"))
    pd.testing.assert_series_equal(a.y.iloc[j], b.y.iloc[j])


# ---------------------------------------------------------------- C
def _c_data(basis_ann=0.20, collapse=None):
    idx = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    d = _data(flat=True, idx=idx, funding=pd.DataFrame(0.0, index=idx, columns=["BTCUSDT", "ETHUSDT"] + [f"S{i:02d}USDT" for i in range(5)]))
    for exp in ("2021-06-25", "2021-09-24", "2021-12-31"):
        e = pd.Timestamp(exp)
        dte = pd.Series((e - idx).days, index=idx).clip(lower=0)
        f = 100 * (1 + basis_ann * dte / 365)
        if collapse is not None:
            f = f.where(idx < pd.Timestamp(collapse), 100.0 + 0.01)
        f = f.where(idx <= e)
        c = f"BTCUSDT_{e:%y%m%d}"
        d.fut[c] = pd.DataFrame({"open": f.shift(1).fillna(f.iloc[0]).where(idx <= e), "high": f * 1.001, "close": f})
        d.fut_meta[c] = ("BTCUSDT", e)
    return d


def test_c_hold_to_expiry_locks_basis():
    d = _c_data()
    res = carry.simulate(d, carry.StratC(d, "BTCUSDT", "vade"), lev=1)
    ep = [e for e in res.episodes if e["hedge"] == "BTCUSDT_210625"]
    assert ep and ep[0]["reason"] == "vade" and ep[0]["closed"] == "2021-06-25"
    assert ep[0]["pnl"] > 0  # kilitlenen basis maliyetleri aşar
    assert res.components["funding"].abs().sum() == 0.0


def test_c_roll_exits_when_basis_narrows():
    d = _c_data(collapse="2021-03-01")
    res = carry.simulate(d, carry.StratC(d, "BTCUSDT", "gec"), lev=1)
    first = next(e for e in res.episodes if e["hedge"] == "BTCUSDT_210625")
    assert first["reason"] == "sinyal" and first["closed"] < "2021-03-05"
    held = carry.simulate(d, carry.StratC(d, "BTCUSDT", "vade"), lev=1)
    assert next(e for e in held.episodes if e["hedge"] == "BTCUSDT_210625")["reason"] == "vade"


def test_delay_shifts_decisions_by_one_day():
    d = _b_data()
    r0 = carry.simulate(d, carry.StratB(d, 1, 3), lev=1)
    r1 = carry.simulate(d, carry.StratB(d, 1, 3), lev=1, delay=1)
    o0 = min(e["opened"] for e in r0.episodes)
    o1 = min(e["opened"] for e in r1.episodes)
    assert pd.Timestamp(o1) - pd.Timestamp(o0) == pd.Timedelta(days=1)


# ---------------------------------------------------------------- uçtan uca
def test_run_study_end_to_end_registers_36_variants(tmp_path):
    from research import budget

    d = _data(n_alt=6)
    idx = d.idx
    for exp in pd.date_range("2021-03-26", "2025-09-26", freq="QS-MAR") + pd.Timedelta(days=25):
        for base in carry.A_ASSETS:
            dte = pd.Series((exp - idx).days, index=idx).clip(lower=0)
            f = (d.spot["close"][base] * (1 + 0.15 * dte / 365)).where((idx <= exp) & (idx >= exp - pd.Timedelta(days=180)))
            c = f"{base}_{exp:%y%m%d}"
            d.fut[c] = pd.DataFrame({"open": f.shift(1), "high": f * 1.01, "close": f})
            d.fut_meta[c] = (base, exp)
    paths = dict(results_dir=tmp_path / "results", log_path=tmp_path / "deneyler.md", registry_file=tmp_path / "results" / "_registry.json")
    paths["log_path"].write_text("# Deney günlüğü\n\n**Toplam deneme: 0**\n\n| t |\n|---|\n", encoding="utf-8")
    tri = pd.date_range("2021-04-01", "2025-09-29", freq="D")
    rng = np.random.default_rng(9)
    tr = pd.DataFrame({c: rng.normal(0, 0.01, len(tri)) for c in ("main_LF_wf", "main_LS_wf")}, index=tri)
    s = carry.run_study(data=d, trend_returns=tr, progress=lambda *_: None, register=True, **paths)
    assert s["n_variants"] == carry.N_PLANNED == 36 and s["trials_after"] == 36
    assert len(s["table"]) == 28 and [r["variant"] for r in s["table"]][:2] == ["A|BTC|1x", "A|ETH|1x"]
    assert 0.0 <= s["pbo"] <= 1.0 and len(s["plateau"]["B"]["neighbors"]) == 6 and len(s["plateau"]["C"]["neighbors"]) == 2
    out = paths["results_dir"] / carry.EXPERIMENT_ID
    for f in ("sonuc.md", "varyantlar.csv", "ozet.json", "metrics.json", "daily_returns.parquet", "aday/report.md"):
        assert (out / f).exists(), f
    assert pd.read_parquet(out / "daily_returns.parquet").shape[1] == 36
    assert "SONUÇ — `carry_001`" in paths["log_path"].read_text(encoding="utf-8")
    with pytest.raises(budget.BudgetExceededError):
        carry.run_study(data=d, trend_returns=tr, progress=lambda *_: None, register=True, **{**paths, "results_dir": tmp_path / "r2"})


# ---------------------------------------------------------------- reviewer düzeltmeleri
def _components_match_nav(res, nav0=250.0):
    c = res.components
    pnl = c["funding"] + c["basis"] - c["costs"] - c["leg_risk"] - c["liquidation"]
    np.testing.assert_allclose(pnl.cumsum().to_numpy() + nav0, res.nav.to_numpy(), rtol=0, atol=1e-6)


def test_start_and_components_sum_to_nav_including_liquidation():
    idx = pd.date_range("2021-01-01", periods=40, freq="D")
    d = _data(idx=idx, seed=3)
    d.perp["high"].loc["2021-01-20", "BTCUSDT"] = d.perp["open"].loc["2021-01-20", "BTCUSDT"] * 1.4
    res = carry.simulate(d, carry.StratA("BTCUSDT"), lev=3, start=pd.Timestamp("2021-01-10"))
    assert res.returns.index[0] == pd.Timestamp("2021-01-10") and res.liquidations == 1
    assert res.components["liquidation"].sum() > 0
    _components_match_nav(res)


def test_one_leg_gap_freezes_mark_and_long_gap_closes_at_last_close():
    idx = pd.date_range("2021-01-01", periods=40, freq="D")
    d = _data(idx=idx, seed=4)
    d.perp["open"].loc["2021-01-10":"2021-01-12", "BTCUSDT"] = np.nan  # kısa boşluk: MTM yok, pozisyon kalır
    d.spot["open"].loc["2021-01-20":, "ETHUSDT"] = np.nan  # ETH spotu biter
    d.spot["close"].loc["2021-01-20":, "ETHUSDT"] = np.nan
    r_btc = carry.simulate(d, carry.StratA("BTCUSDT"), lev=1)
    assert not any(e["reason"].startswith("fiyat") for e in r_btc.episodes)
    assert r_btc.components["basis"].loc["2021-01-09":"2021-01-11"].abs().sum() == 0.0
    _components_match_nav(r_btc)
    r_eth = carry.simulate(d, carry.StratA("ETHUSDT"), lev=1)
    ep = next(e for e in r_eth.episodes if e["reason"] == "fiyat bitti (delist)")
    assert ep["closed"] == "2021-01-25"  # son senkron gün 01-19 açılışı; 6. gün (> 5 gün boşluk) kapanır
    _components_match_nav(r_eth)


def test_lot_rounded_noop_rebalance_is_not_a_trade_and_skips_are_reported():
    idx = pd.date_range("2021-01-01", periods=60, freq="D")
    d = _data(flat=True, idx=idx)
    path = pd.Series(np.linspace(100, 160, len(idx)), index=idx)
    for k in ("open", "close", "high"):
        d.perp[k]["BTCUSDT"] = path
    for k in ("open", "close"):
        d.spot[k]["BTCUSDT"] = path
    d.step = pd.Series(1.0, index=d.perp["close"].columns)  # büyük lot: yeniden boyutlama miktarı değiştiremez
    d.min_notional = pd.Series(5.0, index=d.perp["close"].columns)
    res = carry.simulate(d, carry.StratA("BTCUSDT"), lev=3)
    assert res.trades == 1 and res.liquidations == 0  # yalnızca açılış; marj boştaki nakitten tamamlanır
    d2 = _data(flat=True, idx=idx)
    d2.step = pd.Series(10.0, index=d2.perp["close"].columns)  # bir lot 1000 USDT: 250'lik hesapta açılamaz
    r2 = carry.simulate(d2, carry.StratA("BTCUSDT"), lev=1)
    assert r2.trades == 0 and r2.skipped.get("min_lot", 0) > 0


def test_c_no_new_contract_on_expiry_day():
    d = _c_data()
    res = carry.simulate(d, carry.StratC(d, "BTCUSDT", "vade"), lev=1)
    nxt = next(e for e in res.episodes if e["hedge"] == "BTCUSDT_210924")
    assert nxt["opened"] == "2021-06-26"  # 06-25 vade açılışında kapanır; yeni kontrat ertesi karar günü (06-25 kapanışı) -> 06-26


def test_c_expiry_rule_follows_delay():
    d = _c_data()
    res = carry.simulate(d, carry.StratC(d, "BTCUSDT", "vade"), lev=1, delay=1)
    nxt = next(e for e in res.episodes if e["hedge"] == "BTCUSDT_210924")
    assert nxt["opened"] == "2021-06-26"  # karar 06-24 kapanışı (gecikmeli) -> işlem 06-26; vade günü (06-25) açılış yok
