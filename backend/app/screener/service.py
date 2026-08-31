import logging
import time

from app.core.config import settings
from app.exchanges import get_exchange

from .scanner import scan_market
from .schemas import ScreenerResult

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 60
_cache: dict[str, tuple[float, list[ScreenerResult]]] = {}


def get_scan_results(force_refresh: bool = False) -> list[ScreenerResult]:
    """Önbelleğe alınmış tarama sonuçlarını döner; süresi geçmişse veya
    `force_refresh=True` ise borsayı yeniden tarar.

    Bu önbellek hem `/screener/top` API'si hem de periyodik zamanlayıcı
    (`app.scheduler`) tarafından paylaşılır — zamanlayıcı düzenli aralıklarla
    `refresh()` çağırarak önbelleği taze tutar, böylece API isteği gelen
    kullanıcı borsa taramasının bitmesini beklemek zorunda kalmaz.
    """
    now = time.time()
    cached = _cache.get(settings.exchange_id)
    if not force_refresh and cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    return refresh()


def refresh() -> list[ScreenerResult]:
    exchange = get_exchange(settings.exchange_id)
    results = scan_market(exchange)
    _cache[settings.exchange_id] = (time.time(), results)
    logger.info("screener refreshed: %d symbols scanned", len(results))
    return results
