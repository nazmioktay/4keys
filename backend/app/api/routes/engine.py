from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.engine.service import ModelNotTrained, run_cycle_once
from app.portfolio.schemas import PortfolioStatus
from app.portfolio.shared import get_portfolio

router = APIRouter(prefix="/engine", tags=["engine"])


class CycleResponse(BaseModel):
    actions: list[dict]


@router.post("/run-cycle", response_model=CycleResponse)
def run_cycle() -> CycleResponse:
    try:
        actions = run_cycle_once()
    except ModelNotTrained as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

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
