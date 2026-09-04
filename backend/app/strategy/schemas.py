from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Kullanılabilir göstergeler: compute_indicators() çıktısındaki kolonlar
# (close, open, high, low, volume, ema_fast, ema_slow, rsi, macd,
# macd_signal, macd_hist, volume_sma, momentum)


class Operand(BaseModel):
    indicator: str | None = None
    value: float | None = None


class ConditionNode(BaseModel):
    """TradingView Pine Script yazmadan strateji tanımlamak için JSON kural ağacı.

    - `compare`: left OP right   (örn. rsi < 30)
    - `cross`: left, right'ı `direction` yönünde keser (örn. ema_fast, ema_slow'u yukarı keser)
    - `and` / `or`: `conditions` listesindeki alt kuralları birleştirir
    """

    type: Literal["compare", "cross", "and", "or"]
    left: Operand | None = None
    right: Operand | None = None
    op: Literal["gt", "lt", "gte", "lte", "eq"] | None = None
    direction: Literal["above", "below"] | None = None
    conditions: list["ConditionNode"] | None = None


ConditionNode.model_rebuild()


class StrategyDefinition(BaseModel):
    name: str
    direction: Literal["long", "short"]
    entry: ConditionNode
    exit: ConditionNode | None = None
    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None
    # İşlem maliyetleri (komisyon + kayma) — önceden hesaba katılmıyordu,
    # bkz. app.portfolio.schemas.RiskRules'daki AYNI alanlar/gerekçe.
    # Varsayılanlar Binance Futures taker ücretine (~%0.04) ve mütevazı
    # bir kayma tahminine (~%0.02) dayanır.
    commission_pct: float = 0.04
    slippage_pct: float = 0.02


class StrategyBacktestRequest(BaseModel):
    symbol: str
    strategy: StrategyDefinition
    timeframe: str | None = None
    lookback: int | None = None


class TradeRecord(BaseModel):
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str  # "rule" | "take_profit" | "stop_loss"


class StrategyBacktestResult(BaseModel):
    symbol: str
    strategy_name: str
    trades: list[TradeRecord]
    trades_closed: int
    win_rate_pct: float
    total_profit_pct: float
    max_drawdown_pct: float
