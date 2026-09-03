from app.exchanges.binance import BinanceExchange


def test_fetch_ohlcv_single_call_when_under_limit(monkeypatch):
    exchange = BinanceExchange()
    calls = []

    def fake_fetch_ohlcv(symbol, timeframe, limit, since=None):
        calls.append({"limit": limit, "since": since})
        base = since or 0
        return [[base + i * 3_600_000, 1.0, 2.0, 0.5, 1.5, 100.0] for i in range(limit)]

    monkeypatch.setattr(exchange._futures, "fetch_ohlcv", fake_fetch_ohlcv)

    df = exchange.fetch_ohlcv("BTC/USDT:USDT", "1h", 500)

    assert len(calls) == 1
    assert len(df) == 500


def test_fetch_ohlcv_paginates_when_over_limit(monkeypatch):
    exchange = BinanceExchange()
    calls = []
    hour_ms = 3_600_000

    def fake_fetch_ohlcv(symbol, timeframe, limit, since=None):
        calls.append({"limit": limit, "since": since})
        start = since
        return [[start + i * hour_ms, 1.0, 2.0, 0.5, 1.5, 100.0] for i in range(limit)]

    monkeypatch.setattr(exchange._futures, "parse_timeframe", lambda tf: 3600)
    monkeypatch.setattr(exchange._futures, "milliseconds", lambda: 2_500 * hour_ms)
    monkeypatch.setattr(exchange._futures, "fetch_ohlcv", fake_fetch_ohlcv)

    df = exchange.fetch_ohlcv("BTC/USDT:USDT", "1h", 2500)

    assert len(calls) > 1  # tek istekle Binance'in 1000 limitini aşamaz, sayfalama yaptı
    assert len(df) == 2500
    assert df["timestamp"].is_monotonic_increasing


def test_fetch_ohlcv_stops_when_exchange_returns_no_more_data(monkeypatch):
    exchange = BinanceExchange()
    hour_ms = 3_600_000

    def fake_fetch_ohlcv(symbol, timeframe, limit, since=None):
        # yalnızca 1500 mumluk gerçek veri var, ondan sonrası boş döner
        if since and since >= 1500 * hour_ms:
            return []
        n = min(limit, 1500 - (since or 0) // hour_ms)
        start = since or 0
        return [[start + i * hour_ms, 1.0, 2.0, 0.5, 1.5, 100.0] for i in range(max(n, 0))]

    monkeypatch.setattr(exchange._futures, "parse_timeframe", lambda tf: 3600)
    monkeypatch.setattr(exchange._futures, "milliseconds", lambda: 5000 * hour_ms)
    monkeypatch.setattr(exchange._futures, "fetch_ohlcv", fake_fetch_ohlcv)

    df = exchange.fetch_ohlcv("BTC/USDT:USDT", "1h", 3000)

    assert len(df) <= 1500
