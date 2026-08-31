from typing import Literal

from pydantic import BaseModel, Field


class OrderRequest(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit"] = "market"
    amount: float = Field(..., gt=0, description="Sembol miktarı (quote değil, base varlık miktarı)")
    price: float | None = Field(default=None, description="Limit emirler için zorunlu")
    market_type: Literal["spot", "future"] = "future"
    confirm: bool = Field(
        default=False,
        description="Gerçek emir göndermek için açıkça true olmalı. false ise istek reddedilir.",
    )


class OrderResult(BaseModel):
    raw: dict


class LeverageRequest(BaseModel):
    symbol: str
    leverage: int = Field(..., ge=1, description="İstenen kaldıraç; kod içi sabit tavanı (bkz. security.MAX_LEVERAGE) aşamaz")
    confirm: bool = Field(default=False, description="Gerçek kaldıraç değişikliği için açıkça true olmalı.")


class LeverageResult(BaseModel):
    raw: dict
