from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.dca.schemas import DCAParams
from app.strategy.schemas import StrategyDefinition


class PerformanceMetrics(BaseModel):
    num_trades: int
    win_rate_pct: float
    total_return_pct: float
    cagr_pct: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_drawdown_pct: float
    calmar_ratio: float | None
    profit_factor: float | None
    avg_win_pct: float
    avg_loss_pct: float
    expectancy_pct: float


class DataSufficiency(BaseModel):
    candles_used: int
    candles_requested_final: int
    trades_found: int
    min_trades_target: int
    sufficient: bool
    reason: str


class BacktestReport(BaseModel):
    symbol: str
    timeframe: str
    data_sufficiency: DataSufficiency
    train_metrics: PerformanceMetrics | None
    test_metrics: PerformanceMetrics | None
    full_period_metrics: PerformanceMetrics | None
    warnings: list[str] = Field(default_factory=list)


class BacktestRequest(BaseModel):
    symbol: str
    timeframe: str | None = None
    dca_params: DCAParams | None = None
    strategy: StrategyDefinition | None = None
    min_trades: int = Field(30, ge=5, le=500, description="İstatistiksel olarak yeterli sayılacak minimum kapanan işlem sayısı")
    max_candles: int = Field(5000, ge=100, le=20000, description="Geriye doğru çekilecek maksimum mum sayısı")
    initial_candles: int = Field(500, ge=50, description="Veri yeterliliği taramasının başlangıç adımı")
    train_ratio: float = Field(0.7, gt=0.3, lt=0.95, description="Kronolojik eğitim (in-sample) bölümünün oranı")

    @model_validator(mode="after")
    def _exactly_one_target(self) -> "BacktestRequest":
        if (self.dca_params is None) == (self.strategy is None):
            raise ValueError("Tam olarak biri verilmeli: dca_params VEYA strategy.")
        return self

    @property
    def kind(self) -> Literal["dca", "strategy"]:
        return "dca" if self.dca_params is not None else "strategy"
