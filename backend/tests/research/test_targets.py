import numpy as np
import pandas as pd
import pytest

from research import config, targets


def _prices(n=60, seed=0, symbols=("A", "B", "C")):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.DataFrame({s: 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))) for s in symbols}, index=idx)
    open_ = close.shift(1).fillna(100.0)
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    return {"open": open_, "high": high, "low": low, "close": close}


def test_next_rv_is_parkinson_vol_of_the_following_days_and_tail_is_nan():
    p = _prices()
    var = np.log(p["high"] / p["low"]) ** 2 / (4 * np.log(2))
    t1, t3 = targets.next_rv(1), targets.next_rv(3)
    y1, y3 = t1.compute(p), t3.compute(p)
    d = pd.Timestamp("2024-02-10")
    assert y1.loc[d, "A"] == pytest.approx(np.sqrt(var.loc[d + pd.Timedelta(days=1), "A"]))
    assert y3.loc[d, "A"] == pytest.approx(np.sqrt(var.loc[d + pd.Timedelta(days=1): d + pd.Timedelta(days=3), "A"].mean()))
    assert y1.iloc[-1].isna().all() and y3.iloc[-3:].isna().all().all() and y3.iloc[-4].notna().all()
    assert targets.next_rv(1, annualize=True).compute(p).loc[d, "A"] == pytest.approx(y1.loc[d, "A"] * np.sqrt(365))


def test_pure_forward_labels_never_use_data_at_or_before_the_decision_day():
    """next_rv / cross_sectional_rank etiketleri yalnızca t+1.. verisinden gelir: t ve öncesini bozmak etiketi DEĞİŞTİRMEZ.
    (vol_adj_return ve strategy_outcome t'ye kadar bilinen veriyi — vol/pozisyon — meşru olarak kullanır.)"""
    p = _prices()
    d = pd.Timestamp("2024-02-10")
    for target in (targets.next_rv(2), targets.cross_sectional_rank(2)):
        base = target.compute(p).loc[d]
        broken = {k: v.copy() for k, v in p.items()}
        for k in broken:
            broken[k].loc[:d] = broken[k].loc[:d] * 5.0 + 3.0  # t ve öncesi bozulur
        # open[t+1] için t+1 bozulmadı; yalnızca <= d bozuldu
        pd.testing.assert_series_equal(target.compute(broken).loc[d], base, rtol=1e-9)  # kayan pencere ortalaması ~1e-16 oynar
    assert targets.next_rv(2).label_end_days == 3


def test_vol_adj_return_scales_forward_open_return_by_trailing_vol():
    p = _prices()
    t = targets.vol_adj_return(2, vol_window=20)
    d = pd.Timestamp("2024-02-10")
    fwd = np.log(p["open"].loc[d + pd.Timedelta(days=3), "B"] / p["open"].loc[d + pd.Timedelta(days=1), "B"])
    vol = np.log(p["close"]["B"]).diff().loc[:d].iloc[-20:].std()
    assert t.compute(p).loc[d, "B"] == pytest.approx(fwd / (vol * np.sqrt(2)))


def test_cross_sectional_rank_is_within_universe_members_only():
    p = _prices()
    membership = pd.DataFrame(True, index=p["close"].index, columns=p["close"].columns)
    membership["C"] = False  # C evren dışı
    y = targets.cross_sectional_rank(1).compute(p, membership)
    d = pd.Timestamp("2024-02-10")
    assert y.loc[d, "C"] != y.loc[d, "C"]  # NaN
    assert sorted(y.loc[d, ["A", "B"]].tolist()) == [0.5, 1.0]  # iki üye arasında sıra
    full = targets.cross_sectional_rank(1).compute(p)
    assert full.loc[d].between(0, 1).all()


def test_strategy_outcome_is_net_of_round_trip_cost_and_zero_when_flat():
    p = _prices()
    t = targets.strategy_outcome("ema_trend", 3, span=10)
    y = t.compute(p)
    pos = targets.STRATEGIES["ema_trend"](p, span=10)
    fwd = targets.forward_open_return(p["open"], 3)
    d = pd.Timestamp("2024-02-10")
    rt = 2 * (config.FUTURES_TAKER_FEE + config.SLIPPAGE_BPS_BTC_ETH / 1e4)
    expected = pos.loc[d, "A"] * fwd.loc[d, "A"] - (rt if pos.loc[d, "A"] != 0 else 0.0)
    assert y.loc[d, "A"] == pytest.approx(expected)
    flat = pos.loc[d] == 0
    assert (y.loc[d][flat] == 0).all()  # pozisyon yok -> maliyet de yok, sonuç 0
    b = targets.strategy_outcome("ema_trend", 3, binary=True, span=10)
    assert b.kind == "classification" and set(b.compute(p).stack().dropna().unique()) <= {0.0, 1.0}


def test_registry_and_horizons():
    assert targets.get_target("next_rv", h=2).horizon == 2
    assert targets.get_target("cross_sectional_rank", h=5).kind == "rank"
    with pytest.raises(KeyError):
        targets.get_target("yok")
    with pytest.raises(KeyError):
        targets.strategy_outcome("yok_strateji", 3)
