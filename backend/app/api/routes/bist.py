from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.bist.schemas import (
    BistOrderRequest,
    BistOrderResult,
    LoginRequest,
    LoginResponse,
    LoginVerifyRequest,
    LoginVerifyResponse,
)
from app.bist.service import BistTradingDisabled, get_exchange, login, login_verify, place_bist_order

router = APIRouter(prefix="/bist", tags=["bist"])


@router.post("/login", response_model=LoginResponse)
def start_login(payload: LoginRequest) -> LoginResponse:
    """1. adım: AlgoLab'a kullanıcı adı/şifre ile giriş yapar, SMS/e-posta
    doğrulama kodu gönderilmesini tetikler. Dönen `token`'ı /bist/login/verify'e verin."""
    try:
        token = login(payload.username, payload.password)
    except BistTradingDisabled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return LoginResponse(token=token)


@router.post("/login/verify", response_model=LoginVerifyResponse)
def verify_login(payload: LoginVerifyRequest) -> LoginVerifyResponse:
    """2. adım: telefonunuza/e-postanıza gelen doğrulama kodunu girip oturumu tamamlar."""
    try:
        authenticated = login_verify(payload.token, payload.sms_code)
    except BistTradingDisabled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return LoginVerifyResponse(authenticated=authenticated)


@router.get("/symbols")
def symbols(market_type: Literal["equity", "viop"] = "equity") -> list[str]:
    try:
        exchange = get_exchange()
        return exchange.list_symbols(quote_currency="TRY", market_type=market_type)
    except BistTradingDisabled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.get("/ohlcv")
def ohlcv(symbol: str, timeframe: str = "1d", limit: int = Query(200, ge=1, le=2000)) -> list[dict]:
    try:
        exchange = get_exchange()
        df = exchange.fetch_ohlcv(symbol, timeframe, limit)
    except BistTradingDisabled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    df = df.copy()
    df["timestamp"] = df["timestamp"].astype(str)
    return df.to_dict(orient="records")


@router.get("/positions")
def positions() -> list[dict]:
    try:
        exchange = get_exchange()
        return exchange.fetch_positions()
    except BistTradingDisabled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.post("/order", response_model=BistOrderResult)
def order(payload: BistOrderRequest) -> BistOrderResult:
    """BIST/VIOP'a gerçek emir gönderir. Bkz. `app/bist/service.py` — kill
    switch, `FOURKEYS_ENABLE_BIST_TRADING=true` ve `confirm: true` üçü
    birden sağlanmadan hiçbir emir gönderilmez."""
    try:
        raw = place_bist_order(payload)
    except BistTradingDisabled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BistOrderResult(raw=raw)
