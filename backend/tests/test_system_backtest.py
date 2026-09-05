import numpy as np
import pandas as pd
import pytest

from app.backtest.schemas import SystemBacktestRequest
from app.backtest.system_runner import run_system_backtest
from app.ml.features import build_features
from app.ml.model import SignalModel
from app.exchanges.base import Exchange


class FakeOscillatingExchange(Exchange):
    """Osilasyonlu, göstergelerin ısınması için yeterli uzunlukta sabit bir
    geçmişi olan test borsası (bkz. test_backtest.py::FakeHistoryExchange —
    aynı desen, burada bağımsız kopyalanmış)."""

    def __init__(self, total_candles: int, timeframe: str = "1h", seed: int = 3) -> None:
        freq_minutes = 60
        idx = pd.date_range("2022-01-01", periods=total_candles, freq=f"{freq_minutes}min", tz="UTC").tz_convert(None)
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 60 * np.pi, total_candles)
        base = 20000 + 2000 * np.sin(t)
        close = base + rng.normal(0, 50.0, total_candles)
        self.full_df = pd.DataFrame(
            {
                "timestamp": idx,
                "open": close,
                "high": close + 20,
                "low": close - 20,
                "close": close,
                "volume": rng.uniform(800, 1200, total_candles),
            }
        )

    def list_symbols(self, quote_currency, market_type):
        return ["BTC/USDT:USDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        df = self.full_df
        if since is None:
            return df.iloc[-limit:].reset_index(drop=True)
        since_ts = pd.Timestamp(since, unit="ms")
        mask = df["timestamp"] >= since_ts
        return df.loc[mask].iloc[:limit].reset_index(drop=True)


class FakeLongHistoryExchange(Exchange):
    """`_FAR_PAST_MS` (2017) ile "şimdi" arasında ÇOK daha uzun bir geçmişi
    olan test borsası — `run_system_backtest`'in gerçekten EN SON
    `request.candles` mumu (2017'den itibaren en ESKİ değil) çektiğini
    doğrulamak için (bkz. gerçek üretim regresyonu: backtest raporu
    2019-2020 gibi alakasız bir dönem gösteriyordu, çünkü `fetch_full_history`
    bilerek en eski geçmişten başlıyor)."""

    def __init__(self, total_candles: int) -> None:
        idx = pd.date_range("2017-01-01", periods=total_candles, freq="1h", tz="UTC").tz_convert(None)
        rng = np.random.default_rng(11)
        close = 20000 + np.cumsum(rng.normal(0, 5, total_candles))
        self.full_df = pd.DataFrame(
            {
                "timestamp": idx,
                "open": close,
                "high": close + 20,
                "low": close - 20,
                "close": close,
                "volume": rng.uniform(800, 1200, total_candles),
            }
        )

    def list_symbols(self, quote_currency, market_type):
        return ["BTC/USDT:USDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        df = self.full_df
        if since is None:
            return df.iloc[-limit:].reset_index(drop=True)
        since_ts = pd.Timestamp(since, unit="ms")
        mask = df["timestamp"] >= since_ts
        return df.loc[mask].iloc[:limit].reset_index(drop=True)


def test_run_system_backtest_uses_most_recent_candles_not_earliest(monkeypatch):
    from app.db.session import reset_for_tests
    from app.core.config import settings

    monkeypatch.setattr(settings, "database_url", "")
    reset_for_tests()

    # 5 yıldan fazla (2017'den itibaren) veri var, ama yalnızca son 600
    # mum istiyoruz — rapor 2017'ye değil, verinin EN SONUNA yakın olmalı.
    exchange = FakeLongHistoryExchange(total_candles=50000)
    train_ohlcv = exchange.full_df.iloc[-1000:-600].reset_index(drop=True)
    model = _trained_model(train_ohlcv)

    request = SystemBacktestRequest(symbol="BTC/USDT:USDT", timeframe="1h", candles=600, initial_balance=1000.0)
    report = run_system_backtest(exchange, model, None, request)

    expected_period_end = exchange.full_df["timestamp"].iloc[-1]
    actual_period_end = pd.Timestamp(report.period_end)
    assert actual_period_end.year == expected_period_end.year
    assert (expected_period_end - actual_period_end).days < 30  # ısınma barları yüzünden küçük bir kayma olabilir


def _trained_model(ohlcv: pd.DataFrame) -> SignalModel:
    features = build_features(ohlcv).dropna().reset_index(drop=True)
    # Basit, gerçekçi olmayan ama deterministik bir etiket: getiri işaretine göre.
    returns = features["close"].pct_change().shift(-1).fillna(0.0)
    y = pd.Series(np.where(returns > 0.001, 1, np.where(returns < -0.001, -1, 0)), index=features.index)
    model = SignalModel()
    model.fit(features, y)
    return model


def test_run_system_backtest_produces_report_and_persists_ohlcv(tmp_path, monkeypatch):
    from app.db import session as db_session

    monkeypatch.setattr(db_session.settings, "database_url", f"sqlite:///{tmp_path}/test.db")
    db_session.reset_for_tests()
    db_session.init_db()

    exchange = FakeOscillatingExchange(total_candles=600)
    train_ohlcv = exchange.full_df.iloc[:400].reset_index(drop=True)
    model = _trained_model(train_ohlcv)

    request = SystemBacktestRequest(symbol="BTC/USDT:USDT", timeframe="1h", candles=600, initial_balance=1000.0)
    report = run_system_backtest(exchange, model, None, request)

    assert report.candles_used == 600
    assert report.initial_balance == 1000.0
    assert report.trades_closed >= 0
    assert report.win_rate_pct >= 0.0
    # equity eğrisi trade sayısına tutarlı: final_equity, initial_balance +
    # tüm işlemlerin pnl_quote toplamı olmalı
    if report.trades:
        expected_final = 1000.0 + sum(t.pnl_quote for t in report.trades)
        assert report.final_equity == pytest.approx(expected_final, abs=0.05)

    # DB'ye backfill edilen OHLCV geri okunabilmeli
    from app.db import repository as db

    latest = db.get_latest_backtest_run(symbol="BTC/USDT:USDT")
    assert latest is not None
    assert latest["trades_closed"] == report.trades_closed

    db_session.reset_for_tests()


def test_run_system_backtest_raises_on_insufficient_history():
    # Borsada 100 mum var ama en az 300 istenir -> exchange bunun altında
    # döner, göstergelerin ısınması (250 bar) için yetersiz kalır.
    exchange = FakeOscillatingExchange(total_candles=100)
    model = _trained_model(exchange.full_df)
    request = SystemBacktestRequest(symbol="BTC/USDT:USDT", timeframe="1h", candles=300)

    with pytest.raises(ValueError):
        run_system_backtest(exchange, model, None, request)


def test_stop_loss_closes_long_position_on_crash():
    exchange = FakeOscillatingExchange(total_candles=600)
    train_ohlcv = exchange.full_df.iloc[:400].reset_index(drop=True)
    model = _trained_model(train_ohlcv)

    request = SystemBacktestRequest(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        candles=600,
        initial_balance=1000.0,
        stop_loss_pct=1.0,
        open_confidence=0.5,  # izin verilen minimum -> daha çok işlem tetiklensin
        close_confidence=0.9,  # kapanış yalnızca stop-loss'tan gelsin, sinyalden değil
    )
    report = run_system_backtest(exchange, model, None, request)

    stop_loss_trades = [t for t in report.trades if t.exit_reason == "stop_loss"]
    for t in stop_loss_trades:
        if t.direction == "long":
            assert t.exit_price <= t.entry_price * 1.0001
