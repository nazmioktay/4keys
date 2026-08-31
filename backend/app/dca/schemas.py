from typing import Literal

from pydantic import BaseModel, Field


class DCAParams(BaseModel):
    base_order_size: float
    deviation_pct: float = Field(..., gt=0, description="İlk averaging order için sapma yüzdesi")
    deviation_multiplier: float = Field(..., ge=1.0, description="Sonraki averaging order sapma çarpanı")
    order_size_multiplier: float = Field(..., ge=1.0, description="Sonraki averaging order boyut çarpanı")
    max_safety_orders: int = Field(..., ge=0, le=20)
    take_profit_pct: float = Field(..., gt=0)
    stop_loss_pct: float | None = Field(default=None, gt=0)
    direction: Literal["long", "short"] = "long"


class DCABacktestResult(BaseModel):
    params: DCAParams
    trades_closed: int
    trades_open_at_end: int
    win_rate_pct: float
    total_profit_pct: float
    max_drawdown_pct: float
    max_capital_used: float
    avg_trade_duration_candles: float
    trade_pnls_pct: list[float] = []


class DCAOptimizeRequest(BaseModel):
    symbol: str
    balance: float = Field(..., gt=0, description="Bot için ayrılan maksimum sermaye (quote para birimi)")
    direction: Literal["long", "short"] = "long"
    timeframe: str | None = None
    lookback: int | None = None
    objective: Literal["profit", "profit_over_drawdown", "win_rate"] = "profit_over_drawdown"
    allow_stop_loss: bool = False
    top_n: int = Field(default=5, ge=1, le=20)


class DCAOptimizeResponse(BaseModel):
    symbol: str
    candidates: list[DCABacktestResult]
