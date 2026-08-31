from dataclasses import dataclass
from datetime import datetime, timezone


class KillSwitchActive(Exception):
    """Kill switch aktifken otomatik işlem yollarının fırlattığı istisna."""


@dataclass
class KillSwitchState:
    active: bool = False
    reason: str = ""
    triggered_by: str = ""  # "manual" | "auto_drawdown"
    triggered_at: str | None = None


_state = KillSwitchState()


def is_active() -> bool:
    return _state.active


def status() -> KillSwitchState:
    return _state


def activate(reason: str, triggered_by: str = "manual") -> KillSwitchState:
    """Kill switch'i devreye alır — bu aktifken hiçbir yeni pozisyon açılmaz
    (bkz. `app.engine.service.run_cycle_once`, `app.engine.decision.DecisionEngine`,
    `app.trading.executor.place_live_order`). Mevcut açık pozisyonlar
    otomatik kapatılmaz; bu bilinçli bir tasarım kararıdır — panik halinde
    piyasa fiyatından zorla kapatmak bazen durumu daha da kötüleştirebilir,
    karar Proje Sahibi'ne bırakılır (bkz. Kripto Bot Rehberi Bölüm 0.1)."""
    global _state
    _state = KillSwitchState(
        active=True, reason=reason, triggered_by=triggered_by, triggered_at=datetime.now(timezone.utc).isoformat()
    )
    return _state


def deactivate() -> KillSwitchState:
    global _state
    _state = KillSwitchState()
    return _state


def reset_for_tests() -> None:
    deactivate()
