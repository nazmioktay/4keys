from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.portfolio.risk_manager import calculate_kelly_position_size, calculate_position_size, evaluate_risk, resolve_kelly_multiplier
from app.portfolio.schemas import (
    KellySizeRequest,
    KellySizeResponse,
    PortfolioStatus,
    PositionSizeRequest,
    PositionSizeResponse,
    RiskCheckRequest,
    RiskDecision,
    RiskRules,
    TradeStats,
)
from app.portfolio.shared import get_portfolio, reset_portfolio

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


class ResetRequest(BaseModel):
    starting_equity: float | None = None
    rules: RiskRules | None = None


@router.get("/status", response_model=PortfolioStatus)
def status() -> PortfolioStatus:
    return PortfolioStatus(**get_portfolio().status())


@router.put("/rules", response_model=RiskRules)
def update_rules(rules: RiskRules) -> RiskRules:
    """Ana para yönetimi kurallarını günceller (mevcut pozisyonlar/equity korunur)."""
    get_portfolio().rules = rules
    return rules


@router.post("/reset", response_model=PortfolioStatus)
def reset(payload: ResetRequest) -> PortfolioStatus:
    """Portföyü verilen sermaye ve kurallarla sıfırdan başlatır (açık pozisyonlar silinir)."""
    portfolio = reset_portfolio(payload.starting_equity, payload.rules)
    return PortfolioStatus(**portfolio.status())


@router.post("/position-size", response_model=PositionSizeResponse)
def position_size(payload: PositionSizeRequest) -> PositionSizeResponse:
    """İşlem başına risk yüzdesi ve stop-loss mesafesinden pozisyon boyutu hesaplar."""
    try:
        size_quote, risk_amount_quote, stop_distance_pct = calculate_position_size(
            payload.equity,
            payload.entry_price,
            payload.stop_loss_price,
            payload.risk_per_trade_pct,
            payload.direction,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PositionSizeResponse(
        size_quote=round(size_quote, 6),
        risk_amount_quote=round(risk_amount_quote, 6),
        stop_distance_pct=round(stop_distance_pct, 4),
    )


@router.get("/trade-stats", response_model=TradeStats)
def trade_stats() -> TradeStats:
    """Paylaşılan portföyün kendi kapanmış işlem geçmişinden hesaplanan
    kazanma oranı ve ortalama kazanç/kayıp — `position_sizing_method='kelly'`
    iken otomatik olarak bu istatistikler kullanılır."""
    return get_portfolio().trade_stats()


@router.post("/kelly-size", response_model=KellySizeResponse)
def kelly_size(payload: KellySizeRequest) -> KellySizeResponse:
    """Çeyrek/yarım/tam Kelly kriterine göre bağımsız pozisyon boyutu hesaplar.

    Durumsuzdur — paylaşılan portföyü etkilemez. Bir backtest raporundan
    (`/backtest/run`) gelen `win_rate_pct`/`avg_win_pct`/`avg_loss_pct`
    değerlerini burada deneyip Kelly'nin ne önerdiğini görebilirsiniz.
    """
    try:
        multiplier = resolve_kelly_multiplier(payload.variant, payload.custom_multiplier)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    size_quote, applied_pct, full_kelly_pct = calculate_kelly_position_size(
        payload.equity,
        payload.win_rate_pct,
        payload.avg_win_pct,
        payload.avg_loss_pct,
        multiplier,
        payload.max_kelly_fraction_pct,
    )
    return KellySizeResponse(
        full_kelly_pct=round(full_kelly_pct, 4),
        applied_kelly_pct=round(applied_pct, 4),
        kelly_multiplier_used=multiplier,
        size_quote=round(size_quote, 6),
    )


@router.post("/risk-check", response_model=RiskDecision)
def risk_check(payload: RiskCheckRequest) -> RiskDecision:
    """Verilen portföy durumuna göre önerilen bir pozisyonun kurallara uyup
    uymadığını (ve gerekirse küçültülmüş boyutunu) döner. Durumsuzdur —
    paylaşılan portföyü etkilemez, "ne olurdu" sorgusu için kullanılır."""
    return evaluate_risk(
        payload.equity,
        payload.open_positions,
        payload.realized_pnl_session,
        payload.proposed_symbol,
        payload.proposed_size_quote,
        payload.rules,
    )
