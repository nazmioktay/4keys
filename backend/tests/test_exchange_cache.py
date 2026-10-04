import pandas as pd
import pytest

from app.core.config import settings
from app.db.session import init_db, reset_for_tests
from app.exchanges.base import Exchange
from app.exchanges.cache import fetch_ohlcv_cached


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "sqlite:///:memory:")
    reset_for_tests()
    init_db()
    yield
    reset_for_tests()


class _CountingExchange(Exchange):
    """Sabit bir mum serisi döner, kaç kez `fetch_ohlcv` çağırdığını sayar
    ve her çağrıdaki `since`'i kaydeder — önbelleğin gerçekten yalnızca
    EKSİK kuyruğu istediğini doğrulamak için."""

    def __init__(self, total_candles: int = 500) -> None:
        # Son mum "şimdi"ye yakın olmalı, yoksa tazelik kontrolü (bkz.
        # `_FRESHNESS_TOLERANCE_BARS`) test ortamında her zaman "bayat" bulur.
        end = pd.Timestamp.now(tz="UTC").floor("h")
        start = end - pd.Timedelta(hours=total_candles - 1)
        idx = pd.date_range(start, periods=total_candles, freq="1h", tz="UTC").tz_convert(None)
        close = list(range(total_candles))
        self.full_df = pd.DataFrame(
            {
                "timestamp": idx,
                "open": close,
                "high": [c + 1 for c in close],
                "low": [c - 1 for c in close],
                "close": close,
                "volume": [100.0] * total_candles,
            }
        )
        self.call_count = 0
        self.since_calls: list[int | None] = []

    def list_symbols(self, quote_currency, market_type):
        return ["BTC/USDT:USDT"]

    def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
        self.call_count += 1
        self.since_calls.append(since)
        df = self.full_df
        if since is None:
            return df.iloc[-limit:].reset_index(drop=True)
        since_ts = pd.Timestamp(since, unit="ms")
        mask = df["timestamp"] >= since_ts
        return df.loc[mask].iloc[:limit].reset_index(drop=True)


def _forming_bar_ts() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("h").tz_convert(None)


def test_fetch_ohlcv_cached_hits_exchange_once_when_db_cold():
    exchange = _CountingExchange()
    result = fetch_ohlcv_cached(exchange, "BTC/USDT:USDT", "1h", 100)

    assert len(result) == 100
    assert exchange.call_count == 1
    assert exchange.since_calls == [None]


def test_fetch_ohlcv_cached_drops_the_still_forming_bar():
    exchange = _CountingExchange()
    result = fetch_ohlcv_cached(exchange, "BTC/USDT:USDT", "1h", 100)

    assert result["timestamp"].iloc[-1] < _forming_bar_ts()
    assert result["timestamp"].iloc[-1] == _forming_bar_ts() - pd.Timedelta(hours=1)


def test_fetch_ohlcv_cached_refetches_only_a_small_tail_when_warm():
    exchange = _CountingExchange()
    first = fetch_ohlcv_cached(exchange, "BTC/USDT:USDT", "1h", 100)
    assert exchange.call_count == 1

    second = fetch_ohlcv_cached(exchange, "BTC/USDT:USDT", "1h", 100)
    assert exchange.call_count == 2
    # Tam geçmişi baştan değil, son birkaç barı (`since` ile) istemeli.
    assert exchange.since_calls[1] is not None
    since_ts = pd.Timestamp(exchange.since_calls[1], unit="ms")
    assert since_ts >= first["timestamp"].iloc[-3]
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True), check_dtype=False)


def test_fetch_ohlcv_cached_overwrites_a_previously_stored_partial_bar():
    from app.db import repository as db

    exchange = _CountingExchange()
    fetch_ohlcv_cached(exchange, "BTC/USDT:USDT", "1h", 100)

    last_closed = exchange.full_df.iloc[-2]
    partial = pd.DataFrame([{**last_closed.to_dict(), "close": -999.0, "high": -999.0}])
    db.save_ohlcv_bulk("BTC/USDT:USDT", "1h", partial)
    assert db.get_ohlcv("BTC/USDT:USDT", "1h", 1)["close"].iloc[-1] == -999.0

    result = fetch_ohlcv_cached(exchange, "BTC/USDT:USDT", "1h", 100)

    assert result["close"].iloc[-1] == last_closed["close"]
    assert db.get_ohlcv("BTC/USDT:USDT", "1h", 1)["close"].iloc[-1] == last_closed["close"]


def test_fetch_ohlcv_cached_falls_back_to_cache_when_exchange_fails():
    exchange = _CountingExchange()
    first = fetch_ohlcv_cached(exchange, "BTC/USDT:USDT", "1h", 100)

    def _boom(*args, **kwargs):
        raise RuntimeError("network down")

    exchange.fetch_ohlcv = _boom
    result = fetch_ohlcv_cached(exchange, "BTC/USDT:USDT", "1h", 100)
    pd.testing.assert_frame_equal(first.reset_index(drop=True), result.reset_index(drop=True), check_dtype=False)


def test_fetch_ohlcv_cached_falls_back_to_full_fetch_when_db_disabled(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "")
    from app.db.session import reset_for_tests as _reset

    _reset()
    exchange = _CountingExchange()
    result = fetch_ohlcv_cached(exchange, "BTC/USDT:USDT", "1h", 50)
    assert len(result) == 50
    assert exchange.call_count == 1
    assert result["timestamp"].iloc[-1] < _forming_bar_ts()
