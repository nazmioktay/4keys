import ccxt
import pandas as pd

from .base import Exchange


class BinanceExchange(Exchange):
    def __init__(self) -> None:
        self._spot = ccxt.binance()
        self._futures = ccxt.binance({"options": {"defaultType": "future"}})

    def _client(self, market_type: str) -> ccxt.binance:
        return self._futures if market_type == "future" else self._spot

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        client = self._client(market_type)
        markets = client.load_markets()
        return [
            m["symbol"]
            for m in markets.values()
            if m.get("quote") == quote_currency
            and m.get("active", True)
            and (market_type != "future" or m.get("swap"))
        ]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        client = self._client("future")
        raw = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
