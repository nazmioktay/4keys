from app.core.config import settings

from .manager import PortfolioManager
from .schemas import RiskRules

# Süreç ömrü boyunca paylaşılan tek portföy/risk yöneticisi.
# Karar motoru (app.engine.decision.DecisionEngine) ve /portfolio API'si
# aynı örneği kullanır, böylece tüm modüller ortak risk bütçesini paylaşır.
_portfolio = PortfolioManager(starting_equity=settings.default_starting_equity, rules=RiskRules())


def get_portfolio() -> PortfolioManager:
    return _portfolio


def reset_portfolio(starting_equity: float | None = None, rules: RiskRules | None = None) -> PortfolioManager:
    global _portfolio
    _portfolio = PortfolioManager(
        starting_equity=starting_equity or settings.default_starting_equity,
        rules=rules or RiskRules(),
    )
    return _portfolio
