from .base import Exchange
from .binance import BinanceExchange

__all__ = ["Exchange", "BinanceExchange", "get_exchange"]


def get_exchange(exchange_id: str) -> Exchange:
    if exchange_id == "binance":
        return BinanceExchange()
    raise ValueError(f"Unsupported exchange: {exchange_id}")
