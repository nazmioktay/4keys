import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.exchanges import get_exchange
from app.screener.scanner import scan_market, top_long, top_short
from app.screener.schemas import ScreenerResult

router = APIRouter(prefix="/screener", tags=["screener"])

_CACHE_TTL_SECONDS = 60
_cache: dict[str, tuple[float, list[ScreenerResult]]] = {}


def _get_scan_results() -> list[ScreenerResult]:
    now = time.time()
    cached = _cache.get(settings.exchange_id)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    exchange = get_exchange(settings.exchange_id)
    results = scan_market(exchange)
    _cache[settings.exchange_id] = (now, results)
    return results


@router.get("/top", response_model=list[ScreenerResult])
def get_top(
    direction: Literal["long", "short"] = Query(..., description="long veya short"),
    limit: int = Query(default=settings.screener_top_n, ge=1, le=50),
) -> list[ScreenerResult]:
    results = _get_scan_results()
    if not results:
        raise HTTPException(status_code=503, detail="Tarama sonucu alınamadı, tekrar deneyin.")

    return top_long(results, limit) if direction == "long" else top_short(results, limit)
