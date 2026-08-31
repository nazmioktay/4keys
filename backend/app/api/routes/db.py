from fastapi import APIRouter, Query

from app.db import repository as db
from app.db.session import check_connection, is_enabled

router = APIRouter(prefix="/db", tags=["database"])


@router.get("/status")
def status() -> dict:
    return {"enabled": is_enabled(), "connected": check_connection()}


@router.get("/trades")
def trades(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    """Kalıcı veritabanından kapanmış işlem geçmişi (süreç yeniden başlasa
    da kaybolmaz) — bkz. `app.portfolio.manager.PortfolioManager.close()`."""
    return db.get_recent_trades(limit)


@router.get("/signals")
def signals(
    limit: int = Query(50, ge=1, le=500),
    symbol: str | None = None,
    source: str | None = Query(None, description="'screener' | 'ml' | 'meta'"),
) -> list[dict]:
    return db.get_recent_signals(limit, symbol, source)
