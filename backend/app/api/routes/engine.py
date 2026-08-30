from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.engine.decision import DecisionEngine
from app.engine.positions import PaperPositionStore
from app.exchanges import get_exchange
from app.ml.model import DEFAULT_MODEL_PATH, SignalModel
from app.screener.scanner import scan_market, top_long, top_short

router = APIRouter(prefix="/engine", tags=["engine"])

# Süreç ömrü boyunca paylaşılan tek paper-trading defteri.
# NOT: Bu bellek içi bir simülasyondur, gerçek borsa emri göndermez.
_positions = PaperPositionStore()


class CycleResponse(BaseModel):
    actions: list[dict]


class StatusResponse(BaseModel):
    open_positions: list[dict]
    closed_history: list[dict]


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


@router.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(
        open_positions=[
            {
                "symbol": p.symbol,
                "direction": p.direction,
                "entry_price": p.entry_price,
                "opened_at": p.opened_at.isoformat(),
            }
            for p in _positions.list_open()
        ],
        closed_history=_positions.closed_history,
    )
