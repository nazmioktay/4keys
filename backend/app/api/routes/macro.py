from fastapi import APIRouter, Query

from app.core.config import settings
from app.db import repository as db
from app.exchanges import get_exchange
from app.macro.service import refresh_and_record_macro_snapshot

router = APIRouter(prefix="/macro", tags=["macro"])


@router.get("/latest")
def latest() -> dict:
    """DB'de kayıtlı en son makro anlık görüntüsünü döner (bkz. `app.macro.data`).
    DB kapalıysa veya hiç kayıt yoksa `null` alanlar/boş sonuç döner."""
    snapshot = db.get_latest_macro_snapshot()
    return snapshot or {"time": None}


@router.get("/history")
def history(limit: int = Query(500, ge=1, le=5000)) -> dict:
    df = db.get_macro_snapshots(limit)
    return {"rows": len(df), "data": df.to_dict(orient="records")}


@router.post("/refresh")
def refresh() -> dict:
    """Tüm ücretsiz makro kaynakları (TOTAL, BTC dominansı, funding rate,
    VIX, altın, dünya endeksleri, Fed/ECB faiz oranları) şimdi çeker ve
    (DB açıksa) kaydeder. Normalde bu, zamanlayıcı tarafından periyodik
    olarak (bkz. `FOURKEYS_MACRO_REFRESH_SECONDS`) otomatik yapılır."""
    exchange = get_exchange(settings.exchange_id)
    snapshot = refresh_and_record_macro_snapshot(exchange)
    return snapshot
