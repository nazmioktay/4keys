from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Position:
    symbol: str
    direction: str  # "long" | "short"
    entry_price: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def unrealized_pct(self, current_price: float) -> float:
        change = (current_price / self.entry_price - 1) * 100
        return change if self.direction == "long" else -change


class PaperPositionStore:
    """Bellek içi paper-trading (simülasyon) pozisyon defteri.

    Gerçek borsaya emir göndermez; karar motorunun ürettiği aksiyonları
    simüle edip izlemeye yarar. Gerçek emir yürütme ayrı, açıkça
    etkinleştirilmesi gereken bir katmanda olmalı (bkz. README "Canlı
    işlem" notu) — bu sınıf o katmanı bilerek içermez.
    """

    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}
        self.closed_history: list[dict] = []

    def get(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def list_open(self) -> list[Position]:
        return list(self._positions.values())

    def open(self, symbol: str, direction: str, price: float) -> Position:
        position = Position(symbol=symbol, direction=direction, entry_price=price)
        self._positions[symbol] = position
        return position

    def close(self, symbol: str, price: float) -> dict | None:
        position = self._positions.pop(symbol, None)
        if position is None:
            return None
        record = {
            "symbol": position.symbol,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": price,
            "pnl_pct": round(position.unrealized_pct(price), 3),
            "opened_at": position.opened_at.isoformat(),
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.closed_history.append(record)
        return record
