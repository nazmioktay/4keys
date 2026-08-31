from .base import Exchange
from .binance import BinanceExchange
from .demo import DemoExchange

__all__ = ["Exchange", "BinanceExchange", "DemoExchange", "get_exchange"]


def get_exchange(exchange_id: str) -> Exchange:
    """Yalnızca herkese açık piyasa verisi için borsa istemcisi döner.

    Kimlik doğrulamasızdır ve her zaman gerçek (testnet olmayan) piyasa
    verisini kullanır — screener, ML ve strateji modülleri bunu kullanır.
    Gerçek hesap/emir işlemleri için `app.trading.executor.get_trading_exchange`'i
    kullanın.

    `exchange_id="demo"`, ağ erişimi olmadan sentetik veri üreten
    `DemoExchange`'i döner (bkz. `app.exchanges.demo`) — yalnızca bu
    salt-okunur piyasa verisi arayüzünde geçerlidir; gerçek emir verme
    yolu (`app.trading.executor.get_trading_exchange`) "demo"yu kabul
    etmez.
    """
    if exchange_id == "binance":
        return BinanceExchange(testnet=False)
    if exchange_id == "demo":
        return DemoExchange()
    raise ValueError(f"Unsupported exchange: {exchange_id}")
