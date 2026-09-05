from app.db import repository as db
from app.exchanges.binance import BinanceExchange


def fetch_and_record_open_interest_snapshot(exchange: BinanceExchange, symbol: str) -> dict | None:
    """Bir sembolün açık pozisyonunu (open interest) çekip (DB açıksa)
    kalıcı hale getirir. Borsadan erişilemezse (ağ hatası, spot sembol vb.)
    None döner ve hiçbir şey kaydedilmez — çağıran taraf bunu normal
    karşılamalı (bkz. `app.orderbook.service` AYNI desen)."""
    metrics = exchange.fetch_open_interest(symbol)
    if metrics is None:
        return None
    db.record_open_interest_snapshot(symbol, metrics)
    return metrics


def refresh_all_configured_symbols(exchange: BinanceExchange, symbols: list[str]) -> dict[str, dict | None]:
    """Birden çok sembol için open interest anlık görüntüsünü toplar; bir
    sembolün başarısız olması diğerlerini etkilemez."""
    return {symbol: fetch_and_record_open_interest_snapshot(exchange, symbol) for symbol in symbols}
