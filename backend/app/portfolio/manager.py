from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.config import settings
from app.db import repository as db
from app.security import kill_switch

from .risk_manager import calculate_kelly_position_size, calculate_position_size, evaluate_risk
from .schemas import PositionExposure, RiskDecision, RiskRules, TradeStats


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

    `rules.position_sizing_method` iki modu destekler:
    - `"fixed_risk"`: stop-loss mesafesine göre sabit risk yüzdesi (klasik).
    - `"kelly"`: çeyrek/yarım/tam Kelly kriteri. İstatistikler (kazanma
      oranı, ortalama kazanç/kayıp) varsayılan olarak bu portföyün KENDİ
      kapanmış işlem geçmişinden otomatik hesaplanır — yani sistem canlı
      performansına göre kendi kendini ayarlar. Yeterli geçmiş
      (`rules.kelly_min_trades`) birikene kadar güvenli tarafta kalmak için
      otomatik olarak `fixed_risk`'e düşer. Bir backtest raporundan gelen
      istatistikleri "önsel" (prior) olarak kullanmak isterseniz
      `kelly_stats_override` ile geçebilirsiniz.
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

    def trade_stats(self) -> TradeStats:
        """Kapanmış işlem geçmişinden kazanma oranı ve ortalama kazanç/kayıp hesaplar."""
        pnls = [record["pnl_pct"] for record in self.closed_history]
        if not pnls:
            return TradeStats(num_trades=0, win_rate_pct=0.0, avg_win_pct=0.0, avg_loss_pct=0.0)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        return TradeStats(
            num_trades=len(pnls),
            win_rate_pct=round(len(wins) / len(pnls) * 100, 2),
            avg_win_pct=round(sum(wins) / len(wins), 3) if wins else 0.0,
            avg_loss_pct=round(sum(losses) / len(losses), 3) if losses else 0.0,
        )

    def propose_open(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss_price: float,
        kelly_stats_override: TradeStats | None = None,
    ) -> RiskDecision:
        size_quote = self._size_new_position(symbol, direction, entry_price, stop_loss_price, kelly_stats_override)
        exposures = [PositionExposure(symbol=p.symbol, size_quote=p.size_quote) for p in self.positions.values()]
        return evaluate_risk(self.equity, exposures, self.realized_pnl_session, symbol, size_quote, self.rules)

    def _size_new_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss_price: float,
        kelly_stats_override: TradeStats | None,
    ) -> float:
        if self.rules.position_sizing_method == "kelly":
            stats = kelly_stats_override or self.trade_stats()
            if stats.num_trades >= self.rules.kelly_min_trades and stats.avg_loss_pct < 0:
                size_quote, _, _ = calculate_kelly_position_size(
                    self.equity,
                    stats.win_rate_pct,
                    stats.avg_win_pct,
                    stats.avg_loss_pct,
                    self.rules.kelly_multiplier,
                    self.rules.max_kelly_fraction_pct,
                )
                return size_quote
            # Yeterli/geçerli Kelly istatistiği yok -> güvenli tarafta kal, fixed_risk kullan.

        size_quote, _, _ = calculate_position_size(
            self.equity, entry_price, stop_loss_price, self.rules.max_risk_per_trade_pct, direction
        )
        return size_quote

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
        db.record_trade(record)
        self._maybe_trip_kill_switch()
        return record

    def _maybe_trip_kill_switch(self) -> None:
        """Güvenlik Protokolü Bölüm 9.3: günlük/oturum drawdown limiti
        aşılırsa kill switch'i otomatik devreye alır. Zaten aktifse tekrar
        tetiklemez (ilk tetikleyen sebep ve zaman korunur)."""
        if kill_switch.is_active():
            return
        if self.starting_equity <= 0:
            return
        drawdown_pct = max(0.0, (self.starting_equity - self.equity) / self.starting_equity * 100)
        if drawdown_pct >= settings.kill_switch_daily_drawdown_pct:
            kill_switch.activate(
                reason=(
                    f"Oturum drawdown'u %{drawdown_pct:.2f}, limiti (%{settings.kill_switch_daily_drawdown_pct}) aştı."
                ),
                triggered_by="auto_drawdown",
            )

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
            "trade_stats": self.trade_stats(),
        }
