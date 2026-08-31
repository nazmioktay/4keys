import ccxt
import pandas as pd

from .base import Exchange


class BinanceExchange(Exchange):
    """Binance adapter'ı.

    İki kullanım modu vardır:
    - **Kimlik doğrulamasız (varsayılan)**: Sadece herkese açık piyasa verisi
      (semboller, OHLCV). Screener, ML ve strateji modülleri bu modu kullanır
      ve API anahtarınıza asla dokunmaz.
    - **Kimlik doğrulamalı** (`api_key`/`api_secret` verildiğinde): Bakiye
      okuma, pozisyon okuma ve gerçek emir gönderme. Bu mod yalnızca
      `app/trading/executor.py` üzerinden, açık güvenlik kapılarından
      geçirilerek kullanılmalıdır — doğrudan burada değil.
    """

    def __init__(self, api_key: str | None = None, api_secret: str | None = None, testnet: bool = True) -> None:
        auth = {"apiKey": api_key, "secret": api_secret} if api_key and api_secret else {}
        self._spot = ccxt.binance(auth)
        self._futures = ccxt.binance({**auth, "options": {"defaultType": "future"}})
        if testnet:
            self._spot.set_sandbox_mode(True)
            self._futures.set_sandbox_mode(True)
        self._authenticated = bool(api_key and api_secret)
        self.testnet = testnet

    def _client(self, market_type: str) -> ccxt.binance:
        return self._futures if market_type == "future" else self._spot

    def _require_auth(self) -> None:
        if not self._authenticated:
            raise PermissionError(
                "Bu işlem Binance API anahtarı gerektirir. BinanceExchange'i "
                "api_key/api_secret ile oluşturun (bkz. app/trading/executor.py)."
            )

    # ---- Herkese açık piyasa verisi (Exchange arayüzü) ----

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

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        client = self._client("future")
        raw = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    # ---- Kimlik doğrulamalı hesap/emir işlemleri ----

    def fetch_balance(self, market_type: str = "future") -> dict:
        self._require_auth()
        return self._client(market_type).fetch_balance()

    def fetch_positions(self) -> list[dict]:
        self._require_auth()
        return self._futures.fetch_positions()

    def fetch_open_orders(self, symbol: str | None = None, market_type: str = "future") -> list[dict]:
        self._require_auth()
        return self._client(market_type).fetch_open_orders(symbol)

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        market_type: str = "future",
    ) -> dict:
        self._require_auth()
        client = self._client(market_type)
        return client.create_order(symbol, order_type, side, amount, price)

    def cancel_order(self, order_id: str, symbol: str, market_type: str = "future") -> dict:
        self._require_auth()
        return self._client(market_type).cancel_order(order_id, symbol)
