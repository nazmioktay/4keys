import secrets

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.auth.tokens import create_token
from app.core.config import settings

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str
    expires_in_hours: int


def _auth_configured() -> bool:
    return bool(settings.auth_username) and bool(settings.auth_password.get_secret_value()) and bool(
        settings.auth_secret_key.get_secret_value()
    )


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    """Bkz. `app.auth.middleware.AuthMiddleware` — bu, token GEREKTİRMEYEN
    tek uç noktadır. Sabit-zamanlı karşılaştırma (`secrets.compare_digest`)
    kullanıcı adı/şifre için zamanlama saldırılarını (timing attack) önler."""
    if not _auth_configured():
        raise HTTPException(
            status_code=500,
            detail="Kimlik doğrulama yapılandırılmamış — .env'de FOURKEYS_AUTH_USERNAME, "
            "FOURKEYS_AUTH_PASSWORD ve FOURKEYS_AUTH_SECRET_KEY eksik.",
        )
    valid_user = secrets.compare_digest(payload.username, settings.auth_username)
    valid_pass = secrets.compare_digest(payload.password, settings.auth_password.get_secret_value())
    if not (valid_user and valid_pass):
        raise HTTPException(status_code=401, detail="Kullanıcı adı veya şifre hatalı.")
    return LoginResponse(
        token=create_token(payload.username),
        username=payload.username,
        expires_in_hours=settings.auth_token_ttl_hours,
    )


@router.get("/me")
def me(request: Request) -> dict:
    """Sayfa yenilendiğinde/uygulama ilk açıldığında saklanan token'ın hâlâ
    geçerli olup olmadığını doğrulamak için — bu uç noktaya ulaşabildiyse
    (middleware zaten `request.state.username`'i doldurmuştur) token
    geçerlidir."""
    return {"username": request.state.username}
