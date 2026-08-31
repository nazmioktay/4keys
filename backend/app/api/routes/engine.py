from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.engine.decision import DecisionEngine
from app.engine.positions import PaperPositionStore
from app.exchanges import get_exchange
from app.ml.model import DEFAULT_MODEL_PATH, SignalModel
from app.portfolio.schemas import PortfolioStatus
from app.portfolio.shared import get_portfolio
from app.screener.scanner import scan_market, top_long, top_short

router = APIRouter(prefix="/engine", tags=["engine"])

# Geriye dönük uyumluluk / portföy yöneticisi olmadan tek başına kullanım için.
_positions = PaperPositionStore()


class CycleResponse(BaseModel):
    actions: list[dict]


@router.post("/run-cycle", response_model=CycleResponse)
def run_cycle() -> CycleResponse:
    if not DEFAULT_MODEL_PATH.exists():
        raise HTTPException(status_code=409, detail="Model henüz eğitilmedi. Önce /ml/train çağırın.")

    exchange = get_exchange(settings.exchange_id)
    model = SignalModel.load_from()

    results = scan_market(exchange)
    picks = top_long(results, settings.screener_top_n) + top_short(results, settings.screener_top_n)
    symbols = [r.symbol for r in picks]

    engine = DecisionEngine(
        exchange=exchange,
        model=model,
        positions=_positions,
        timeframe=settings.candle_timeframe,
        lookback=settings.candle_lookback,
        portfolio=get_portfolio(),
    )
    actions = engine.run_cycle(symbols)
    return CycleResponse(
        actions=[
            {
                "symbol": a.symbol,
                "type": a.type,
                "reason": a.reason,
                "price": a.price,
                "confidence": a.confidence,
            }
            for a in actions
        ]
    )


@router.get("/status", response_model=PortfolioStatus)
def status() -> PortfolioStatus:
    """Karar motorunun açtığı/kapattığı pozisyonları portföy durumu olarak döner.

    Detaylı risk kuralları ve manuel işlemler için bkz. /portfolio/* uçları —
    ikisi de aynı paylaşılan PortfolioManager örneğini kullanır.
    """
    return PortfolioStatus(**get_portfolio().status())
