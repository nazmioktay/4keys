from pydantic import BaseModel

from fastapi import APIRouter

from app.core.config import settings
from app.security import kill_switch
from app.security.safety import MAX_LEVERAGE

router = APIRouter(prefix="/security", tags=["security"])


class KillSwitchStatus(BaseModel):
    active: bool
    reason: str
    triggered_by: str
    triggered_at: str | None


class ActivateRequest(BaseModel):
    reason: str


class SecurityStatus(BaseModel):
    kill_switch: KillSwitchStatus
    live_trading_enabled: bool
    binance_testnet: bool
    max_leverage: int
    api_key_permission_check_required: bool
    kill_switch_daily_drawdown_pct: float


@router.get("/status", response_model=SecurityStatus)
def status() -> SecurityStatus:
    """Güvenlik Protokolü (Bölüm 9) kontrol listesine karşı anlık durum özeti."""
    ks = kill_switch.status()
    return SecurityStatus(
        kill_switch=KillSwitchStatus(
            active=ks.active, reason=ks.reason, triggered_by=ks.triggered_by, triggered_at=ks.triggered_at
        ),
        live_trading_enabled=settings.enable_live_trading,
        binance_testnet=settings.binance_testnet,
        max_leverage=MAX_LEVERAGE,
        api_key_permission_check_required=settings.require_api_key_permission_check,
        kill_switch_daily_drawdown_pct=settings.kill_switch_daily_drawdown_pct,
    )


@router.post("/kill-switch/activate", response_model=KillSwitchStatus)
def activate(payload: ActivateRequest) -> KillSwitchStatus:
    """Kill switch'i manuel olarak devreye alır — bundan sonra hiçbir yeni
    pozisyon (paper veya gerçek) açılmaz. Mevcut açık pozisyonlar otomatik
    kapatılmaz; kapatma kararı kullanıcıya aittir."""
    state = kill_switch.activate(reason=payload.reason, triggered_by="manual")
    return KillSwitchStatus(active=state.active, reason=state.reason, triggered_by=state.triggered_by, triggered_at=state.triggered_at)


@router.post("/kill-switch/deactivate", response_model=KillSwitchStatus)
def deactivate() -> KillSwitchStatus:
    state = kill_switch.deactivate()
    return KillSwitchStatus(active=state.active, reason=state.reason, triggered_by=state.triggered_by, triggered_at=state.triggered_at)
