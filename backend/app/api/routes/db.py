from datetime import datetime

from fastapi import APIRouter, Query

from app.core.config import settings
from app.db import repository as db
from app.db.session import check_connection, is_enabled

router = APIRouter(prefix="/db", tags=["database"])


@router.get("/status")
def status() -> dict:
    return {"enabled": is_enabled(), "connected": check_connection()}


@router.get("/trades")
def trades(
    limit: int = Query(50, ge=1, le=500),
    symbol: str | None = Query(None, description="Yalnızca bu sembolün işlemleri"),
    since: datetime | None = Query(None, description="Bu zamandan (dahil, ISO 8601) sonra kapanan işlemler"),
    until: datetime | None = Query(None, description="Bu zamana (dahil, ISO 8601) kadar kapanan işlemler"),
) -> list[dict]:
    """Kalıcı veritabanından kapanmış işlem geçmişi (süreç yeniden başlasa
    da kaybolmaz) — bkz. `app.portfolio.manager.PortfolioManager.close_tranche()`.
    Her satır artık kapanış NEDENİNİ de taşır (`reason` — ör. "stop-loss
    tetiklendi", "model kapanış/ters sinyali") — bkz. `GET /db/trades/summary`
    sembol bazlı toplu özet için."""
    return db.get_recent_trades(limit, symbol=symbol, since=since, until=until)


@router.get("/trades/summary")
def trades_summary(
    since: datetime | None = Query(None, description="Bu zamandan (dahil) sonra kapanan işlemler; boşsa TÜM geçmiş"),
    until: datetime | None = Query(None, description="Bu zamana (dahil) kadar kapanan işlemler"),
) -> list[dict]:
    """Kapanmış işlemleri sembole göre gruplayıp işlem sayısı/kazanma oranı/
    toplam PnL döner — frontend'deki işlem geçmişi sayfasının özet paneli."""
    return db.get_trade_pnl_summary(since=since, until=until)


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
