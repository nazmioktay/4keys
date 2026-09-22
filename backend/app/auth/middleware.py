from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .tokens import verify_token

# Kimlik doğrulama gerektirmeyen tek yollar: giriş uç noktası ile izleme/
# sağlık kontrolleri (Prometheus/uptime araçları token taşımaz).
PUBLIC_PATHS = {"/health", "/metrics", "/auth/login"}


class AuthMiddleware(BaseHTTPMiddleware):
    """Uygulamanın TÜMÜNÜ (bu üçü hariç) `Authorization: Bearer <token>`
    arkasına kilitler — VPS'te herkese açık çalışan bu backend'de daha
    önce HİÇBİR erişim kontrolü yoktu (bkz. canlı Binance işlem ekranı
    eklendikten sonra fark edilen güvenlik açığı)."""

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        token = auth_header[7:] if auth_header.lower().startswith("bearer ") else None
        username = verify_token(token) if token else None
        if not username:
            return JSONResponse(status_code=401, content={"detail": "Giriş yapmanız gerekiyor."})

        request.state.username = username
        return await call_next(request)
