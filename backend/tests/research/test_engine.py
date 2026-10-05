import numpy as np
import pandas as pd
import pytest

from research import engine
from research.guard import FinalTestError


def _prices(values, symbol="BTCUSDT", start="2021-01-01"):
    return pd.DataFrame({symbol: values}, index=pd.date_range(start, periods=len(values), freq="D"), dtype=float)


def _target(prices, weight):
    return pd.DataFrame(weight, index=prices.index, columns=prices.columns, dtype=float)


def test_zero_cost_buy_and_hold_equals_open_to_open_price_return():
    px = _prices([100, 110, 99, 120, 130, 125])
    result = engine.run(_target(px, 1.0), px, fee_rate=0.0, slippage_bps=0.0)
    # karar d kapanışı -> d+1 açılışı: ilk tutulan gün index 1; son gün (açılışı yok) atılır
    assert result.nav.iloc[-1] == pytest.approx(px.iloc[-1, 0] / px.iloc[1, 0])
    assert list(result.returns.index) == list(px.index[:-1])
    assert result.returns.iloc[0] == 0.0  # ilk gün hiç pozisyon yok


def test_entry_cost_is_fee_plus_slippage_on_turnover():
    px = _prices([100, 100, 110, 110])
    result = engine.run(_target(px, 1.0), px, fee_rate=0.0005, slippage_bps=2.0)
    cost = 0.0005 + 0.0002
    # d=1: pozisyon açılır (turnover 1) ve fiyat 100 -> 110
    assert result.turnover.iloc[1] == pytest.approx(1.0)
    assert result.returns.iloc[1] == pytest.approx(0.10 - cost)
    assert result.costs["commission"].iloc[1] == pytest.approx(0.0005)
    assert result.costs["slippage"].iloc[1] == pytest.approx(0.0002)
    # sonraki günlerde yeniden dengeleme yok (drift): ağırlık ~1 kalmaz ama band=0 ise hedefe dönülür
    assert result.turnover.iloc[2] < 0.2


def test_funding_long_pays_short_receives_and_flat_pays_nothing():
    px = _prices([100, 100, 100, 100, 100])
    funding = pd.DataFrame(0.0001, index=px.index, columns=px.columns)  # her gün %0,01
    long = engine.run(_target(px, 1.0), px, fee_rate=0.0, slippage_bps=0.0, funding=funding)
    short = engine.run(_target(px, -1.0), px, fee_rate=0.0, slippage_bps=0.0, funding=funding)
    flat = engine.run(_target(px, 0.0), px, fee_rate=0.0, slippage_bps=0.0, funding=funding)
    assert long.costs["funding"].sum() == pytest.approx(3 * 0.0001)  # gün 1..3 tutuluyor
    assert short.costs["funding"].sum() == pytest.approx(-3 * 0.0001)
    assert flat.costs["funding"].sum() == 0.0
    assert long.nav.iloc[-1] < 1.0 < short.nav.iloc[-1]


def test_daily_funding_buckets_payments_into_half_open_day_intervals():
    idx = pd.date_range("2021-01-01", periods=3, freq="D")
    events = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "time": pd.to_datetime(["2021-01-01 00:00", "2021-01-01 08:00", "2021-01-02 00:00", "2021-01-02 16:00"]),
            "rate": [0.0001, 0.0002, 0.0004, 0.0008],
        }
    )
    out = engine.daily_funding(events, idx, ["BTCUSDT"])
    # (d 00:00, d+1 00:00]: 01-01 00:00 ödemesi 12-31 gününe aittir (aralık dışı); 01-02 00:00 -> 01-01 günü
    assert out.loc["2021-01-01", "BTCUSDT"] == pytest.approx(0.0002 + 0.0004)
    assert out.loc["2021-01-02", "BTCUSDT"] == pytest.approx(0.0008)
    assert out.loc["2021-01-03", "BTCUSDT"] == 0.0


def test_no_trade_band_suppresses_small_rebalances():
    px = _prices([100.0] * 8)
    weights = _target(px, 0.50)
    weights.iloc[3:5] = 0.52
    weights.iloc[5:] = 0.49
    banded = engine.run(weights, px, fee_rate=0.001, slippage_bps=0.0, band=0.05)
    unbanded = engine.run(weights, px, fee_rate=0.001, slippage_bps=0.0, band=0.0)
    assert banded.turnover.sum() == pytest.approx(0.50)  # yalnızca ilk giriş
    assert unbanded.turnover.sum() > banded.turnover.sum()


def test_min_notional_blocks_tiny_positions_and_reports_them():
    px = pd.DataFrame({"BTCUSDT": [100.0] * 5, "ETHUSDT": [10.0] * 5}, index=pd.date_range("2021-01-01", periods=5, freq="D"))
    target = pd.DataFrame({"BTCUSDT": 0.5, "ETHUSDT": 0.001}, index=px.index)
    result = engine.run(
        target, px, fee_rate=0.0, slippage_bps=0.0, account_size=1000.0,
        min_notional=pd.Series({"BTCUSDT": 5.0, "ETHUSDT": 5.0}),  # ETH hedefi 1 USDT < 5
    )
    assert result.weights_held["ETHUSDT"].abs().sum() == 0.0
    assert result.weights_held["BTCUSDT"].abs().sum() > 0
    assert result.skipped and all(s["symbol"] == "ETHUSDT" and s["reason"] == "min_notional" for s in result.skipped)
    assert result.diagnostics["skipped_positions"] == len(result.skipped)


def test_amount_step_rounds_quantity_down():
    px = pd.DataFrame({"BTCUSDT": [100.0] * 4}, index=pd.date_range("2021-01-01", periods=4, freq="D"))
    target = pd.DataFrame({"BTCUSDT": 0.555}, index=px.index)
    result = engine.run(target, px, fee_rate=0.0, slippage_bps=0.0, account_size=1000.0,
                        amount_step=pd.Series({"BTCUSDT": 0.1}))
    # 555 USDT / 100 = 5,55 adet -> 5,5 adet -> ağırlık 0,55
    assert result.weights_held["BTCUSDT"].iloc[1] == pytest.approx(0.55)


def test_delay_bars_shifts_execution_by_whole_days():
    px = _prices([100, 100, 110, 121, 121, 121])
    base = engine.run(_target(px, 1.0), px, fee_rate=0.0, slippage_bps=0.0)
    delayed = engine.run(_target(px, 1.0), px, fee_rate=0.0, slippage_bps=0.0, delay_bars=1)
    assert base.turnover.iloc[1] == pytest.approx(1.0)
    assert delayed.turnover.iloc[1] == 0.0 and delayed.turnover.iloc[2] == pytest.approx(1.0)
    assert delayed.nav.iloc[-1] < base.nav.iloc[-1]  # gecikme: giriş 100 yerine 110'dan (ilk %10'luk hareket kaçtı)


def test_missing_prices_are_reported_not_filled_and_positions_liquidated():
    px = _prices([100, 100, 100, np.nan, 100, 100])
    result = engine.run(_target(px, 1.0), px, fee_rate=0.0, slippage_bps=0.0)
    assert result.diagnostics["missing_price_symbol_days"] > 0
    assert result.diagnostics["forced_liquidations"] >= 1  # gün 1'de açılan pozisyon, fiyatsız günde tasfiye
    assert result.weights_held["BTCUSDT"].iloc[1] > 0
    assert result.weights_held["BTCUSDT"].iloc[2] == 0.0  # açılışı NaN olan güne taşınan pozisyon tutulamaz


def test_symbol_contribution_sums_to_daily_return_and_pnl_ranked():
    idx = pd.date_range("2021-01-01", periods=6, freq="D")
    px = pd.DataFrame({"A": [100, 100, 110, 121, 121, 121], "B": [50, 50, 50, 45, 40, 40]}, index=idx, dtype=float)
    target = pd.DataFrame({"A": 0.5, "B": -0.5}, index=idx)
    result = engine.run(target, px, fee_rate=0.0004, slippage_bps=5.0)
    assert result.symbol_contrib.sum(axis=1).to_numpy() == pytest.approx(result.returns.to_numpy())
    assert result.symbol_pnl().index[0] in ("A", "B")


def test_final_test_window_is_refused_without_registered_opening():
    px = _prices([100.0] * 5, start="2025-09-28")  # 09-28 .. 10-02
    with pytest.raises(FinalTestError):
        engine.run(_target(px, 1.0), px)
    ok = _prices([100.0] * 4, start="2025-09-25")  # 09-25 .. 09-28
    engine.run(_target(ok, 1.0), ok)  # sorun yok


def test_cost_summary_reports_cost_to_gross_ratio():
    px = _prices([100, 100, 120, 120])
    result = engine.run(_target(px, 1.0), px, fee_rate=0.001, slippage_bps=0.0, band=0.05)  # drift yeniden dengelemesi yok
    summary = engine.cost_summary(result)
    assert summary["commission"] == pytest.approx(0.001)
    assert summary["cost_to_gross"] == pytest.approx(0.001 / 0.2)


def test_default_slippage_tiers():
    s = engine.default_slippage_bps(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XYZUSDT"], top20={"SOLUSDT"})
    assert s.to_dict() == {"BTCUSDT": 2.0, "ETHUSDT": 2.0, "SOLUSDT": 5.0, "XYZUSDT": 15.0}
