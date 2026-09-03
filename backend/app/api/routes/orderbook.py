from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.db import repository as db
from app.exchanges import get_exchange
from app.orderbook.service import fetch_and_record_orderbook_snapshot

router = APIRouter(prefix="/orderbook", tags=["orderbook"])


@router.get("/latest")
def latest(symbol: str = Query(..., description="Örn: BTC/USDT:USDT")) -> dict:
    """DB'de kayıtlı en son emir defteri anlık görüntüsünü döner. Geçmişe
    dönük emir defteri verisi yoktur — DB kapalıysa veya bu sembol için hiç
    kayıt yoksa `null` alanlar döner."""
    snapshot = db.get_latest_orderbook_snapshot(symbol)
    return snapshot or {"time": None, "symbol": symbol}


@router.get("/history")
def history(symbol: str = Query(..., description="Örn: BTC/USDT:USDT"), limit: int = Query(500, ge=1, le=5000)) -> dict:
    df = db.get_orderbook_snapshots(symbol, limit)
    return {"rows": len(df), "data": df.to_dict(orient="records")}


@router.post("/refresh")
def refresh(symbol: str = Query(..., description="Örn: BTC/USDT:USDT")) -> dict:
    """Bir sembolün emir defteri özetini şimdi çeker ve (DB açıksa) kaydeder.
    Normalde bu, zamanlayıcı tarafından `feature_snapshot_symbols`
    ayarındaki semboller için periyodik olarak (bkz.
    `FOURKEYS_ORDERBOOK_REFRESH_SECONDS`) otomatik yapılır."""
    exchange = get_exchange(settings.exchange_id)
    metrics = fetch_and_record_orderbook_snapshot(exchange, symbol)
    if metrics is None:
        raise HTTPException(status_code=502, detail="Emir defteri borsadan alınamadı.")
    return {"symbol": symbol, **metrics}
