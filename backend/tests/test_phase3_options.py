from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.backtest.system_runner import run_system_backtest
from app.ml.labeling import LONG, NEUTRAL, triple_barrier_labels
from app.ml.multi_timeframe_features import compute_multi_timeframe_features
from app.portfolio.manager import PortfolioManager
from app.portfolio.schemas import RiskRules

from tests.test_portfolio import _engine_with, _FixedModel, _TickerExchange
from tests.test_system_backtest import _scenario_exchange, _scenario_request, _ScriptedModel


# --- Etiketleme (3.1) ---

def _bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"]).assign(volume=1.0)


def test_same_bar_double_hit_is_neutral_with_tie_neutral():
    ohlcv = _bars([[100, 100, 100, 100], [100, 103, 97, 100], [100, 100, 100, 100], [100, 100, 100, 100]])
    legacy = triple_barrier_labels(ohlcv, 2.0, 2.0, max_horizon=2)
    fixed = triple_barrier_labels(ohlcv, 2.0, 2.0, max_horizon=2, tie_neutral=True)
    assert legacy.iloc[0] == LONG  # eski davranış: simetrik bariyerde hep long
    assert fixed.iloc[0] == NEUTRAL


def test_cost_aware_barriers_neutralize_moves_that_do_not_cover_costs():
    ohlcv = _bars([[100, 100, 100, 100], [100, 102.05, 100, 102], [102, 102, 102, 102], [102, 102, 102, 102]])
    assert triple_barrier_labels(ohlcv, 2.0, 2.0, max_horizon=2).iloc[0] == LONG
    assert triple_barrier_labels(ohlcv, 2.0, 2.0, max_horizon=2, cost_pct=0.12).iloc[0] == NEUTRAL


# --- Backtest seçenekleri (3.1 zaman çıkışı, 3.2 başabaş, 3.3 trend filtresi) ---

def test_time_exit_closes_after_max_holding_bars():
    exchange, s, flat = _scenario_exchange()
    model = _ScriptedModel({exchange.full_df.loc[s, "timestamp"]: 1})

    report = run_system_backtest(exchange, model, None, _scenario_request(max_holding_bars=5), persist=False)
    without = run_system_backtest(exchange, model, None, _scenario_request(), persist=False)

    assert report.trades_closed == 1
    assert report.trades[0].exit_reason == "time_exit"
    assert report.trades[0].duration_candles == 4  # giriş mumu dahil 5 mum
    assert without.trades_closed == 0


def test_breakeven_moves_stop_to_entry_after_favourable_move():
    exchange, s, flat = _scenario_exchange()
    df = exchange.full_df
    df.loc[s + 3, "high"] = flat + 1000.0  # giriş ATR'sinin (~300) belirgin üstünde lehe hareket
    model = _ScriptedModel({df.loc[s, "timestamp"]: 1})

    with_be = run_system_backtest(exchange, model, None, _scenario_request(breakeven_atr_mult=1.0), persist=False)
    without = run_system_backtest(exchange, model, None, _scenario_request(), persist=False)

    # Sonraki mumların dibi (flat+5) girişin (+maliyet) altında: başabaş stop vurulur,
    # açılış (flat+10) stop'un altında olduğu için açılıştan dolar.
    assert with_be.trades_closed == 1
    assert with_be.trades[0].exit_price == pytest.approx(flat + 10.0)
    assert without.trades_closed == 0


def test_trend_filter_blocks_trades_against_daily_trend():
    # günlük üst-TF özelliği için >= 60 günlük mum gerekir
    exchange, s, flat = _scenario_exchange(total=2000)
    df = exchange.full_df
    df.loc[s + 5, "low"] = flat - 1000.0  # long'u kapatacak fitil
    df.loc[s + 5, "high"] = flat + 1000.0  # short'u kapatacak fitil
    # backtest'in gördüğü AYNI pencere (son 1900 mum): günlük EMA50 ~80 günlük
    # bar ile tam yakınsamadığından farklı bir pencerede işaret değişebilir
    window = df.iloc[-1900:].reset_index(drop=True)
    gap = compute_multi_timeframe_features(window).loc[s - (len(df) - 1900), "htf_1d_ema_gap"]
    assert gap != 0 and not np.isnan(gap)
    against = -1 if gap > 0 else 1
    model = _ScriptedModel({df.loc[s, "timestamp"]: against})

    blocked = run_system_backtest(exchange, model, None, _scenario_request(trend_filter="block", candles=1900), persist=False)
    allowed = run_system_backtest(exchange, model, None, _scenario_request(candles=1900), persist=False)

    assert blocked.trades_closed == 0
    assert allowed.trades_closed == 1


# --- Canlı karşılıkları ---

def _portfolio(**rules) -> PortfolioManager:
    return PortfolioManager(starting_equity=1000, rules=RiskRules(entry_tranche_weights=[1.0], **rules))


def test_live_breakeven_moves_stop_once_price_runs_in_favour():
    portfolio = _portfolio(breakeven_atr_mult=1.0)
    position = portfolio.open("BTC/USDT", "long", entry_price=100, size_quote=100, stop_loss_price=95)
    position.entry_atr = 2.0
    engine = _engine_with(_TickerExchange(live_price=103.0), _FixedModel("long", 0.9), portfolio)

    engine.evaluate("BTC/USDT")

    assert position.breakeven_done is True
    assert position.stop_loss_price == pytest.approx(100 * (1 + (portfolio.rules.commission_pct + portfolio.rules.slippage_pct) * 2 / 100))


class _ScriptedBarExchange(_TickerExchange):
    """Statik serinin SON mumunu (open, high, low, close) ile değiştirir;
    `shift_bars` serinin zamanını ileri kaydırır (= yeni bir mum kapandı)."""

    def __init__(self, live_price, last_bar, shift_bars=0):
        super().__init__(live_price)
        self.last_bar = last_bar
        self.shift_bars = shift_bars

    def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
        df = super().fetch_ohlcv(symbol, timeframe, limit, since)
        df["timestamp"] = df["timestamp"] + pd.Timedelta(hours=4) * self.shift_bars
        df.loc[df.index[-1], ["open", "high", "low", "close"]] = self.last_bar
        return df


def test_live_breakeven_does_not_apply_new_stop_to_the_bar_that_triggered_it():
    # Mumun başında low 99,5, sonunda high 102: stop girişe (+maliyet, ~100,12) çekilir.
    # Fiyat hiç geri dönmedi — AYNI mumun low'u yeni stop'a karşı kontrol edilmemeli
    # (backtest sırası: mumun kontrolü önceki stop'la, güncelleme yalnızca sonraki mumları etkiler).
    portfolio = _portfolio(breakeven_atr_mult=1.0)
    position = portfolio.open("BTC/USDT", "long", entry_price=100, size_quote=100, stop_loss_price=98)
    position.opened_at = datetime(2024, 1, 1, tzinfo=timezone.utc)  # statik serinin son mumu bundan sonra başlıyor
    position.entry_atr = 1.5
    # Canlı fiyat (101,0) tek başına eşiği (101,5) geçmez: başabaşı KAPANMIŞ mumun high'ı (102) tetikler.
    exchange = _ScriptedBarExchange(live_price=101.0, last_bar=(100.0, 102.0, 99.5, 101.5))
    engine = _engine_with(exchange, _FixedModel("long", 0.9), portfolio)

    breakeven_stop = 100 * (1 + (portfolio.rules.commission_pct + portfolio.rules.slippage_pct) * 2 / 100)

    first = engine.evaluate("BTC/USDT")  # stop'u çeker
    assert position.breakeven_done is True and position.stop_loss_price == pytest.approx(breakeven_stop)
    assert first is None or first.type != "close"

    second = engine.evaluate("BTC/USDT")  # sonraki 5 dk'lık döngü, AYNI mum
    assert second is None or second.type != "close"

    # Yeni bir mum kapanır ve low'u gerçekten stop'un altına iner → kapanmalı.
    exchange.last_bar = (101.0, 101.5, 100.0, 100.5)
    exchange.shift_bars = 1
    third = engine.evaluate("BTC/USDT")
    assert third is not None and third.type == "close"
    assert third.price == pytest.approx(position.stop_loss_price)


def test_live_breakeven_triggered_by_live_price_applies_only_after_the_forming_bar():
    # Oluşan (henüz kapanmamış) mumda önce dip yaşandı, SONRA canlı fiyat başabaşı tetikledi.
    # Mum kapanınca o dip yeni stop'a karşı değerlendirilmemeli: stop mumun ortasında çekildi,
    # dip stop'tan ÖNCEydi (bkz. stop_effective_from_bar = son kapanmış mum + 1 bar).
    portfolio = _portfolio(breakeven_atr_mult=1.0)
    position = portfolio.open("BTC/USDT", "long", entry_price=100, size_quote=100, stop_loss_price=98)
    position.opened_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    position.entry_atr = 1.5
    # Son KAPANMIŞ mum: high 100,8 (başabaşı tetiklemez); canlı fiyat 102 tetikler.
    exchange = _ScriptedBarExchange(live_price=102.0, last_bar=(100.0, 100.8, 99.6, 100.5))
    engine = _engine_with(exchange, _FixedModel("long", 0.9), portfolio)

    first = engine.evaluate("BTC/USDT")

    assert position.breakeven_done is True
    last_closed = engine._last_bar_ts["BTC/USDT"]
    assert pd.Timestamp(position.stop_effective_from_bar) == last_closed + pd.Timedelta(hours=4)  # oluşan mum
    assert first is None or first.type != "close"

    # Oluşan mum kapandı: dip (99,9) yeni stop'un (~100,12) altında ama stop'tan ÖNCEYDİ.
    exchange.last_bar = (100.5, 102.0, 99.9, 101.5)
    exchange.live_price = 101.5
    exchange.shift_bars = 1
    second = engine.evaluate("BTC/USDT")
    assert second is None or second.type != "close"

    # Sonraki mum low'u gerçekten stop'un altına iniyor -> kapanmalı.
    exchange.last_bar = (101.5, 101.8, 100.0, 100.6)
    exchange.shift_bars = 2
    third = engine.evaluate("BTC/USDT")
    assert third is not None and third.type == "close"
    assert third.price == pytest.approx(position.stop_loss_price)


def test_stop_effective_from_bar_survives_state_roundtrip():
    import json

    portfolio = _portfolio(breakeven_atr_mult=1.0)
    position = portfolio.open("BTC/USDT", "long", entry_price=100, size_quote=100, stop_loss_price=98)
    position.stop_effective_from_bar = "2026-10-05T12:00:00"

    restored = PortfolioManager.from_state(json.loads(json.dumps(portfolio.to_state())))

    assert restored.get("BTC/USDT").stop_effective_from_bar == "2026-10-05T12:00:00"

    # Alan eklenmeden ÖNCE kaydedilmiş bir durum da yüklenebilmeli (None).
    legacy = json.loads(json.dumps(portfolio.to_state()))
    del legacy["positions"][0]["stop_effective_from_bar"]
    assert PortfolioManager.from_state(legacy).get("BTC/USDT").stop_effective_from_bar is None


def test_live_time_exit_closes_old_position():
    portfolio = _portfolio(max_holding_bars=3)
    position = portfolio.open("BTC/USDT", "long", entry_price=110, size_quote=100, stop_loss_price=100)
    position.opened_at = datetime.now(timezone.utc) - timedelta(hours=4 * 3 + 1)  # test motoru 4h mum kullanıyor
    engine = _engine_with(_TickerExchange(live_price=111.0), _FixedModel("long", 0.9), portfolio)

    action = engine.evaluate("BTC/USDT")

    assert action.type == "close"
    assert "zaman çıkışı" in action.reason


def test_live_trend_filter_blocks_entry_against_daily_trend(monkeypatch):
    portfolio = _portfolio(trend_filter="block", max_symbol_exposure_pct=100, max_total_exposure_pct=100)
    engine = _engine_with(_TickerExchange(live_price=None), _FixedModel("short", 0.9), portfolio, {})
    original = engine._predict

    def predict_with_uptrend(symbol):
        prediction, price, row = original(symbol)
        row = row.copy()
        row["htf_1d_ema_gap"] = 0.5
        return prediction, price, row

    monkeypatch.setattr(engine, "_predict", predict_with_uptrend)
    action = engine.evaluate("BTC/USDT")

    assert action.type == "hold"
    assert "trend filtresi" in action.reason
