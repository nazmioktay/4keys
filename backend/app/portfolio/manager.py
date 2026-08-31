from dataclasses import dataclass, field
from datetime import datetime, timezone

from .risk_manager import calculate_position_size, evaluate_risk
from .schemas import PositionExposure, RiskDecision, RiskRules


@dataclass
class PortfolioPosition:
    symbol: str
    direction: str  # "long" | "short"
    entry_price: float
    size_quote: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def pnl_pct(self, current_price: float) -> float:
        change = (current_price / self.entry_price - 1) * 100
        return change if self.direction == "long" else -change


class PortfolioManager:
    """Ana para/risk yönetimi katmanı.

    Karar/strateji motorlarının ürettiği açma sinyallerini burada tanımlı
    kurallara (risk/işlem, toplam maruziyet, sembol maruziyeti, eşzamanlı
    pozisyon sayısı, günlük zarar limiti) göre boyutlandırır ve gerekirse
    reddeder. Tüm modüller (ML karar motoru, DCA botları, manuel stratejiler)
    gerçek/paper emir öncesi bu katmandan geçmelidir.
    """

    def __init__(self, starting_equity: float, rules: RiskRules | None = None) -> None:
        self.starting_equity = starting_equity
        self.equity = starting_equity
        self.rules = rules or RiskRules()
        self.realized_pnl_session = 0.0
        self.positions: dict[str, PortfolioPosition] = {}
        self.closed_history: list[dict] = []

    def get(self, symbol: str) -> PortfolioPosition | None:
        return self.positions.get(symbol)

    def propose_open(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss_price: float,
    ) -> RiskDecision:
        size_quote, _, _ = calculate_position_size(
            self.equity, entry_price, stop_loss_price, self.rules.max_risk_per_trade_pct, direction
        )
        exposures = [PositionExposure(symbol=p.symbol, size_quote=p.size_quote) for p in self.positions.values()]
        return evaluate_risk(self.equity, exposures, self.realized_pnl_session, symbol, size_quote, self.rules)

    def open(self, symbol: str, direction: str, entry_price: float, size_quote: float) -> PortfolioPosition:
        position = PortfolioPosition(symbol=symbol, direction=direction, entry_price=entry_price, size_quote=size_quote)
        self.positions[symbol] = position
        return position

    def close(self, symbol: str, exit_price: float) -> dict | None:
        position = self.positions.pop(symbol, None)
        if position is None:
            return None

        pnl_pct = position.pnl_pct(exit_price)
        pnl_quote = position.size_quote * pnl_pct / 100
        self.equity += pnl_quote
        self.realized_pnl_session += pnl_quote

        record = {
            "symbol": position.symbol,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "size_quote": round(position.size_quote, 6),
            "pnl_pct": round(pnl_pct, 3),
            "pnl_quote": round(pnl_quote, 6),
            "opened_at": position.opened_at.isoformat(),
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.closed_history.append(record)
        return record

    def status(self) -> dict:
        return {
            "equity": round(self.equity, 6),
            "starting_equity": self.starting_equity,
            "realized_pnl_session": round(self.realized_pnl_session, 6),
            "open_positions": [
                {
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "entry_price": p.entry_price,
                    "size_quote": round(p.size_quote, 6),
                    "opened_at": p.opened_at.isoformat(),
                }
                for p in self.positions.values()
            ],
            "closed_history": self.closed_history,
            "rules": self.rules,
        }
