from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.config import settings
from app.db import repository as db
from app.monitoring.metrics import portfolio_equity, portfolio_open_positions, portfolio_realized_pnl_session, record_trade_closed
from app.security import kill_switch

from .risk_manager import calculate_kelly_position_size, calculate_position_size, evaluate_risk
from .schemas import PnlSummary, PnlWindow, PositionExposure, RiskDecision, RiskRules, TradeStats


@dataclass
class PortfolioPosition:
    symbol: str
    direction: str  # "long" | "short"
    entry_price: float
    size_quote: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # --- Kademeli alım/satım durumu ---
    # Pozisyon açıldığı andaki kural ağırlıkları burada DONDURULUR — rules
    # sonradan değişse bile bu pozisyonun planı tutarlı kalır.
    target_size_quote: float = 0.0  # tam hesaplanan (Kelly/fixed_risk) boyut
    entry_tranche_weights: list[float] = field(default_factory=lambda: [1.0])
    entry_fill_index: int = 1  # kaç dilim doldu (açılışta ilk dilim zaten dolu)
    exit_tranche_weights: list[float] = field(default_factory=lambda: [1.0])
    exit_fill_index: int = 0
    exit_base_size_quote: float | None = None  # ilk satış diliminde DONDURULUR

    def pnl_pct(self, current_price: float) -> float:
        change = (current_price / self.entry_price - 1) * 100
        return change if self.direction == "long" else -change

    def entry_fully_filled(self) -> bool:
        return self.entry_fill_index >= len(self.entry_tranche_weights)


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

    def _update_gauges(self) -> None:
        portfolio_equity.set(self.equity)
        portfolio_open_positions.set(len(self.positions))
        portfolio_realized_pnl_session.set(self.realized_pnl_session)

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
        confidence: float | None = None,
        vix_zscore: float | None = None,
    ) -> RiskDecision:
        size_quote = self._size_new_position(symbol, direction, entry_price, stop_loss_price, kelly_stats_override)
        size_quote *= self._confidence_scale(confidence)

        regime_scale, regime_reason = self._regime_scale(vix_zscore)
        size_quote *= regime_scale

        exposures = [PositionExposure(symbol=p.symbol, size_quote=p.size_quote) for p in self.positions.values()]
        decision = evaluate_risk(self.equity, exposures, self.realized_pnl_session, symbol, size_quote, self.rules)
        if regime_reason:
            decision = RiskDecision(allowed=decision.allowed and regime_scale > 0, size_quote=decision.size_quote, reasons=[*decision.reasons, regime_reason])
        return decision

    def _confidence_scale(self, confidence: float | None) -> float:
        """Boyutu tahminin güvenine göre ölçekler: `confidence_scaling_min_confidence`
        (veya altı) -> `confidence_scaling_min_scale`; `1.0` confidence -> `1.0`.
        Aradaki değerler doğrusal enterpole edilir."""
        if not self.rules.confidence_scaling_enabled or confidence is None:
            return 1.0
        min_conf = self.rules.confidence_scaling_min_confidence
        min_scale = self.rules.confidence_scaling_min_scale
        if confidence <= min_conf:
            return min_scale
        if confidence >= 1.0:
            return 1.0
        span = 1.0 - min_conf
        return min_scale + (confidence - min_conf) / span * (1.0 - min_scale) if span > 0 else 1.0

    def _regime_scale(self, vix_zscore: float | None) -> tuple[float, str | None]:
        """VIX rejim filtresi: aşırı stres anlarında boyutu küçültür veya
        tamamen engeller (bkz. `RiskRules` alan açıklamaları)."""
        if not self.rules.vix_regime_filter_enabled or vix_zscore is None:
            return 1.0, None
        if vix_zscore >= self.rules.vix_zscore_block_threshold:
            return 0.0, f"VIX rejim filtresi: z-skoru {vix_zscore:.2f} >= engelleme eşiği {self.rules.vix_zscore_block_threshold}"
        if vix_zscore >= self.rules.vix_zscore_reduce_threshold:
            return 0.5, f"VIX rejim filtresi: z-skoru {vix_zscore:.2f} >= küçültme eşiği {self.rules.vix_zscore_reduce_threshold}, boyut yarıya indirildi"
        return 1.0, None

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
        """`size_quote`, kurallara göre hesaplanan TAM (hedef) pozisyon
        boyutudur — gerçekte ilk anda yalnızca `entry_tranche_weights[0]`
        kesri kadarı açılır; kalanı, sinyal sonraki döngü(ler)de de
        kalıcıysa `add_entry_tranche` ile eklenir (bkz. `DecisionEngine`)."""
        weights = list(self.rules.entry_tranche_weights)
        first_fill = size_quote * weights[0]
        position = PortfolioPosition(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            size_quote=first_fill,
            target_size_quote=size_quote,
            entry_tranche_weights=weights,
            entry_fill_index=1,
            exit_tranche_weights=list(self.rules.exit_tranche_weights),
        )
        self.positions[symbol] = position
        self._update_gauges()
        return position

    def add_entry_tranche(self, symbol: str, price: float) -> PortfolioPosition | None:
        """Bir sonraki alım dilimini mevcut fiyattan ekler; ortalama giriş
        fiyatını yeniden hesaplar. Pozisyon yoksa veya tüm dilimler zaten
        doluysa None döner (no-op)."""
        position = self.positions.get(symbol)
        if position is None or position.entry_fully_filled():
            return None

        weight = position.entry_tranche_weights[position.entry_fill_index]
        add_size = position.target_size_quote * weight
        total_size = position.size_quote + add_size
        position.entry_price = (position.entry_price * position.size_quote + price * add_size) / total_size
        position.size_quote = total_size
        position.entry_fill_index += 1
        return position

    def close_tranche(self, symbol: str, exit_price: float) -> dict | None:
        """Bir satış dilimini kapatır. Pozisyonun tamamı henüz kapanmadıysa
        (`partial: True`) pozisyon açık kalır (küçültülmüş boyutla);
        son dilimde YUVARLAMA ARTIĞI kalmaması için kalan tüm boyut kapatılır."""
        position = self.positions.get(symbol)
        if position is None:
            return None

        if position.exit_base_size_quote is None:
            position.exit_base_size_quote = position.size_quote

        weights = position.exit_tranche_weights
        is_last_tranche = position.exit_fill_index >= len(weights) - 1
        close_size = (
            position.size_quote
            if is_last_tranche
            else min(position.exit_base_size_quote * weights[position.exit_fill_index], position.size_quote)
        )

        pnl_pct = position.pnl_pct(exit_price)
        pnl_quote = close_size * pnl_pct / 100
        self.equity += pnl_quote
        self.realized_pnl_session += pnl_quote
        position.size_quote -= close_size
        position.exit_fill_index += 1

        fully_closed = is_last_tranche or position.size_quote <= 1e-9
        record = {
            "symbol": position.symbol,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "size_quote": round(close_size, 6),
            "pnl_pct": round(pnl_pct, 3),
            "pnl_quote": round(pnl_quote, 6),
            "partial": not fully_closed,
            "tranche": position.exit_fill_index,
            "opened_at": position.opened_at.isoformat(),
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.closed_history.append(record)
        db.record_trade({k: v for k, v in record.items() if k not in ("partial", "tranche")})
        record_trade_closed(position.direction, pnl_pct)

        if fully_closed:
            self.positions.pop(symbol, None)
        self._maybe_trip_kill_switch()
        self._update_gauges()
        return record

    def close(self, symbol: str, exit_price: float) -> dict | None:
        """Pozisyonu TEK seferde, tamamen kapatır (kademeli değil) —
        geriye dönük uyumluluk ve "acil/tam kapat" ihtiyaçları için.
        Modelin normal kapanış sinyalleri `close_tranche` kullanmalı."""
        position = self.positions.get(symbol)
        if position is None:
            return None
        position.exit_tranche_weights = [1.0]
        position.exit_fill_index = 0
        position.exit_base_size_quote = None
        return self.close_tranche(symbol, exit_price)

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
                    "entry_fill_index": p.entry_fill_index,
                    "entry_tranche_count": len(p.entry_tranche_weights),
                    "exit_fill_index": p.exit_fill_index,
                    "exit_tranche_count": len(p.exit_tranche_weights),
                }
                for p in self.positions.values()
            ],
            "closed_history": self.closed_history,
            "rules": self.rules,
            "trade_stats": self.trade_stats(),
        }

    def pnl_summary(self, now: datetime | None = None) -> PnlSummary:
        """Kayan pencereli (rolling — takvim sınırı değil) PNL özeti.
        `now` yalnızca testler için verilir; production'da gerçek UTC an."""
        now = now or datetime.now(timezone.utc)

        def _window(hours: float | None) -> PnlWindow:
            records = self.closed_history
            if hours is not None:
                cutoff = now.timestamp() - hours * 3600
                records = [r for r in records if datetime.fromisoformat(r["closed_at"]).timestamp() >= cutoff]
            pnl_quote = sum(r["pnl_quote"] for r in records)
            wins = [r for r in records if r["pnl_quote"] > 0]
            win_rate = round(len(wins) / len(records) * 100, 2) if records else 0.0
            return PnlWindow(pnl_quote=round(pnl_quote, 6), trade_count=len(records), win_rate_pct=win_rate)

        return PnlSummary(
            total=_window(None),
            daily=_window(24),
            weekly=_window(24 * 7),
            monthly=_window(24 * 30),
        )
