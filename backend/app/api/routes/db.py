from fastapi import APIRouter, Query

from app.core.config import settings
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


@router.get("/features")
def features(
    symbol: str = Query("BTC/USDT:USDT", description="Bkz. FOURKEYS_FEATURE_SNAPSHOT_SYMBOLS"),
    timeframe: str | None = None,
    limit: int = Query(5000, ge=1, le=50000),
) -> dict:
    """Zamanla biriken ML özellik vektörlerini (bkz. `app.ml.features.FEATURE_COLUMNS`)
    döner — ileride LSTM/RL eğitiminde kullanılacak zaman serisi veri setinin
    şu ana kadar ne kadar biriktiğini gösterir."""
    df = db.get_feature_snapshots(symbol, timeframe or settings.candle_timeframe, limit)
    return {"symbol": symbol, "rows": len(df), "data": df.to_dict(orient="records")}
