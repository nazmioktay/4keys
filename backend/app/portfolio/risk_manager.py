from .schemas import PositionExposure, RiskDecision, RiskRules


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_loss_price: float,
    risk_per_trade_pct: float,
    direction: str = "long",
) -> tuple[float, float, float]:
    """Sermayenin belirli bir yüzdesini riske ederek pozisyon boyutu hesaplar.

    Mantık: stop-loss'a çarpılırsa kaybedilecek miktar tam olarak
    `equity * risk_per_trade_pct / 100` olacak şekilde pozisyon büyüklüğü
    (quote para biriminde) geriye hesaplanır.

    Döner: (size_quote, risk_amount_quote, stop_distance_pct)
    """
    if direction == "long" and stop_loss_price >= entry_price:
        raise ValueError("Long pozisyon için stop-loss giriş fiyatının altında olmalı.")
    if direction == "short" and stop_loss_price <= entry_price:
        raise ValueError("Short pozisyon için stop-loss giriş fiyatının üstünde olmalı.")

    stop_distance_pct = abs(entry_price - stop_loss_price) / entry_price * 100
    risk_amount_quote = equity * risk_per_trade_pct / 100
    size_quote = risk_amount_quote / (stop_distance_pct / 100)
    return size_quote, risk_amount_quote, stop_distance_pct


def evaluate_risk(
    equity: float,
    open_positions: list[PositionExposure],
    realized_pnl_session: float,
    proposed_symbol: str,
    proposed_size_quote: float,
    rules: RiskRules,
) -> RiskDecision:
    """Önerilen bir pozisyonu ana para yönetimi kurallarına karşı denetler.

    Reddetmek yerine mümkün olduğunda pozisyonu izin verilen üst sınıra
    küçültür (adjusted size); tamamen imkansızsa allowed=False döner.
    """
    reasons: list[str] = []

    if realized_pnl_session <= -equity * rules.daily_loss_limit_pct / 100:
        return RiskDecision(
            allowed=False,
            size_quote=0.0,
            reasons=[f"Günlük/oturum zarar limiti (%{rules.daily_loss_limit_pct}) aşıldı, yeni işlem açılmıyor."],
        )

    open_symbols = {p.symbol for p in open_positions}
    is_new_symbol = proposed_symbol not in open_symbols
    if is_new_symbol and len(open_symbols) >= rules.max_concurrent_positions:
        return RiskDecision(
            allowed=False,
            size_quote=0.0,
            reasons=[f"Maksimum eşzamanlı pozisyon sayısına (%{rules.max_concurrent_positions}) ulaşıldı."],
        )

    adjusted = proposed_size_quote

    existing_symbol_exposure = sum(p.size_quote for p in open_positions if p.symbol == proposed_symbol)
    max_symbol_quote = equity * rules.max_symbol_exposure_pct / 100
    symbol_headroom = max(max_symbol_quote - existing_symbol_exposure, 0.0)
    if adjusted > symbol_headroom:
        adjusted = symbol_headroom
        reasons.append(f"{proposed_symbol} için sembol bazlı maruziyet limiti (%{rules.max_symbol_exposure_pct}) uygulandı.")

    total_exposure = sum(p.size_quote for p in open_positions)
    max_total_quote = equity * rules.max_total_exposure_pct / 100
    total_headroom = max(max_total_quote - total_exposure, 0.0)
    if adjusted > total_headroom:
        adjusted = total_headroom
        reasons.append(f"Toplam portföy maruziyet limiti (%{rules.max_total_exposure_pct}) uygulandı.")

    if adjusted <= 0:
        return RiskDecision(allowed=False, size_quote=0.0, reasons=reasons or ["Kullanılabilir risk bütçesi kalmadı."])

    return RiskDecision(allowed=True, size_quote=round(adjusted, 6), reasons=reasons)
