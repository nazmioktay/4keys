from app.exchanges import get_exchange
from app.exchanges.demo import DemoExchange


def test_get_exchange_returns_demo_exchange():
    exchange = get_exchange("demo")
    assert isinstance(exchange, DemoExchange)


def test_demo_exchange_lists_symbols():
    exchange = DemoExchange()
    symbols = exchange.list_symbols("USDT", "swap")
    assert len(symbols) > 0
    assert all("/" in s for s in symbols)


def test_demo_exchange_ohlcv_shape_and_columns():
    exchange = DemoExchange()
    df = exchange.fetch_ohlcv("BTC/USDT:USDT", "1h", 200)
    assert len(df) == 200
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert (df["high"] >= df["low"]).all()
    assert (df["close"] > 0).all()


def test_demo_exchange_is_deterministic_per_symbol():
    exchange = DemoExchange()
    df1 = exchange.fetch_ohlcv("ETH/USDT:USDT", "1h", 50, since=0)
    df2 = exchange.fetch_ohlcv("ETH/USDT:USDT", "1h", 50, since=0)
    assert (df1["close"] == df2["close"]).all()


def test_trading_executor_never_uses_demo_exchange():
    from app.trading import executor
    import inspect

    source = inspect.getsource(executor.get_trading_exchange)
    assert "demo" not in source.lower()
