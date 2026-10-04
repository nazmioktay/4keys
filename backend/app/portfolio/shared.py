import logging

from app.core.config import settings
from app.db import repository as db

from .manager import PortfolioManager
from .schemas import RiskRules

logger = logging.getLogger(__name__)

# Süreç ömrü boyunca paylaşılan tek portföy/risk yöneticisi.
# Karar motoru (app.engine.decision.DecisionEngine) ve /portfolio API'si
# aynı örneği kullanır, böylece tüm modüller ortak risk bütçesini paylaşır.
# İlk erişimde DB'deki son anlık görüntüden (bkz. `PortfolioManager.to_state`)
# geri yüklenir — restart'ta açık pozisyonlar ve Kelly geçmişi kaybolmaz.
_portfolio: PortfolioManager | None = None


def _new_portfolio(starting_equity: float | None = None, rules: RiskRules | None = None) -> PortfolioManager:
    portfolio = PortfolioManager(
        starting_equity=starting_equity or settings.default_starting_equity,
        rules=rules or RiskRules(),
    )
    portfolio.persist_enabled = True
    return portfolio


def _restore() -> PortfolioManager:
    state = db.load_portfolio_state()
    if state:
        try:
            portfolio = PortfolioManager.from_state(state)
            portfolio.persist_enabled = True
            logger.info(
                "portföy DB'den geri yüklendi: equity=%.2f, %d açık pozisyon, %d kapanmış işlem",
                portfolio.equity,
                len(portfolio.positions),
                len(portfolio.closed_history),
            )
            return portfolio
        except Exception:  # noqa: BLE001 - bozuk/eski şemalı durum yeni bir portföyle başlamayı engellememeli
            logger.exception("kayıtlı portföy durumu okunamadı, yeni portföyle başlanıyor")
    return _new_portfolio()


def get_portfolio() -> PortfolioManager:
    global _portfolio
    if _portfolio is None:
        _portfolio = _restore()
    return _portfolio


def reset_portfolio(starting_equity: float | None = None, rules: RiskRules | None = None) -> PortfolioManager:
    global _portfolio
    _portfolio = _new_portfolio(starting_equity, rules)
    _portfolio.persist()
    return _portfolio
