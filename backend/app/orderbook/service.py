from app.db import repository as db
from app.exchanges.binance import BinanceExchange


def fetch_and_record_orderbook_snapshot(exchange: BinanceExchange, symbol: str) -> dict | None:
    """Bir sembolün emir defteri özetini çekip (DB açıksa) kalıcı hale
    getirir. Borsadan erişilemezse (ağ hatası, sembol yok vb.) None döner
    ve hiçbir şey kaydedilmez — çağıran taraf bunu normal karşılamalı."""
    metrics = exchange.fetch_order_book_metrics(symbol)
    if metrics is None:
        return None
    db.record_orderbook_snapshot(symbol, metrics)
    return metrics


def refresh_all_configured_symbols(exchange: BinanceExchange, symbols: list[str]) -> dict[str, dict | None]:
    """Birden çok sembol için emir defteri anlık görüntüsünü toplar; bir
    sembolün başarısız olması diğerlerini etkilemez."""
    return {symbol: fetch_and_record_orderbook_snapshot(exchange, symbol) for symbol in symbols}
