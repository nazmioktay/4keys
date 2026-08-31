from .base import Exchange
from .binance import BinanceExchange

__all__ = ["Exchange", "BinanceExchange", "get_exchange"]


def get_exchange(exchange_id: str) -> Exchange:
    """Yalnızca herkese açık piyasa verisi için borsa istemcisi döner.

    Kimlik doğrulamasızdır ve her zaman gerçek (testnet olmayan) piyasa
    verisini kullanır — screener, ML ve strateji modülleri bunu kullanır.
    Gerçek hesap/emir işlemleri için `app.trading.executor.get_trading_exchange`'i
    kullanın.
    """
    if exchange_id == "binance":
        return BinanceExchange(testnet=False)
    raise ValueError(f"Unsupported exchange: {exchange_id}")
