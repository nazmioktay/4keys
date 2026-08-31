from typing import Literal

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str | None = Field(default=None, description="Boş bırakılırsa .env'deki FOURKEYS_ALGOLAB_USERNAME kullanılır")
    password: str | None = Field(default=None, description="Boş bırakılırsa .env'deki FOURKEYS_ALGOLAB_PASSWORD kullanılır")


class LoginResponse(BaseModel):
    token: str


class LoginVerifyRequest(BaseModel):
    token: str
    sms_code: str


class LoginVerifyResponse(BaseModel):
    authenticated: bool


class BistOrderRequest(BaseModel):
    symbol: str
    direction: Literal["buy", "sell"]
    order_type: Literal["market", "limit"] = "market"
    quantity: float = Field(..., gt=0)
    price: float | None = Field(default=None, description="Limit emirler için zorunlu")
    market_type: Literal["equity", "viop"] = "equity"
    confirm: bool = Field(default=False, description="Gerçek emir göndermek için açıkça true olmalı.")


class BistOrderResult(BaseModel):
    raw: dict
