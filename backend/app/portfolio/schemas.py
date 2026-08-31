from typing import Literal

from pydantic import BaseModel, Field


class RiskRules(BaseModel):
    """Ana para yönetimi kuralları — tüm botlar/stratejiler bu kurallar
    üzerinden pozisyon açar; kurallar aşılıyorsa işlem küçültülür veya reddedilir.
    """

    max_risk_per_trade_pct: float = Field(1.0, gt=0, description="Bir işlemde riske edilecek sermaye yüzdesi (SL mesafesine göre boyutlandırma için)")
    max_total_exposure_pct: float = Field(50.0, gt=0, description="Tüm açık pozisyonların toplamının sermayeye oranı üst sınırı")
    max_symbol_exposure_pct: float = Field(15.0, gt=0, description="Tek bir sembole ayrılabilecek maksimum sermaye yüzdesi")
    max_concurrent_positions: int = Field(5, ge=1, description="Aynı anda açık olabilecek maksimum farklı sembol sayısı")
    daily_loss_limit_pct: float = Field(5.0, gt=0, description="Bu yüzdeye ulaşan günlük/oturum zararında yeni işlem açılmaz")


class PositionSizeRequest(BaseModel):
    equity: float = Field(..., gt=0)
    entry_price: float = Field(..., gt=0)
    stop_loss_price: float = Field(..., gt=0)
    direction: Literal["long", "short"] = "long"
    risk_per_trade_pct: float = Field(1.0, gt=0)


class PositionSizeResponse(BaseModel):
    size_quote: float
    risk_amount_quote: float
    stop_distance_pct: float


class PositionExposure(BaseModel):
    symbol: str
    size_quote: float


class RiskCheckRequest(BaseModel):
    equity: float = Field(..., gt=0)
    open_positions: list[PositionExposure] = Field(default_factory=list)
    realized_pnl_session: float = 0.0
    proposed_symbol: str
    proposed_size_quote: float = Field(..., gt=0)
    rules: RiskRules = Field(default_factory=RiskRules)


class RiskDecision(BaseModel):
    allowed: bool
    size_quote: float
    reasons: list[str] = Field(default_factory=list)


class PortfolioStatus(BaseModel):
    equity: float
    starting_equity: float
    realized_pnl_session: float
    open_positions: list[dict]
    closed_history: list[dict]
    rules: RiskRules
