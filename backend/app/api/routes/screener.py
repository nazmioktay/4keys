from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.screener.schemas import ScreenerResult
from app.screener.scanner import top_long, top_short
from app.screener.service import get_scan_results

router = APIRouter(prefix="/screener", tags=["screener"])


@router.get("/top", response_model=list[ScreenerResult])
def get_top(
    direction: Literal["long", "short"] = Query(..., description="long veya short"),
    limit: int = Query(default=settings.screener_top_n, ge=1, le=50),
) -> list[ScreenerResult]:
    results = get_scan_results()
    if not results:
        raise HTTPException(status_code=503, detail="Tarama sonucu alınamadı, tekrar deneyin.")

    return top_long(results, limit) if direction == "long" else top_short(results, limit)
