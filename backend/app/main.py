import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import backtest, bank, bist, dca, engine, macro, ml, orderbook, portfolio, rl, scheduler, screener, security, strategy, trading
from app.api.routes import db as db_routes
from app.core.config import settings
from app.db.session import init_db
from app.scheduler.scheduler import start_scheduler, stop_scheduler

logger = logging.getLogger(__name__)


class UnhandledExceptionMiddleware(BaseHTTPMiddleware):
    """Beklenmeyen (yakalanmamış) hataları düzgün bir JSON yanıtına çevirir.

    Bunun bir `@app.exception_handler(Exception)` yerine gerçek bir
    middleware olmasının nedeni: Starlette, bare `Exception` için kayıtlı
    bir exception handler'ı `ServerErrorMiddleware`'e ekler — bu da
    `CORSMiddleware`'in DIŞINDA (üstünde) çalışır. Sonuç: hata yanıtı CORS
    başlıklarını hiç almaz ve tarayıcıda gerçek hata mesajı yerine anlamsız
    bir "Failed to fetch" görünür. Middleware olarak (ve CORSMiddleware'den
    SONRA eklenerek, böylece ondan İÇERİDE çalışacak şekilde) yazmak,
    ürettiği yanıtın normal akışla CORSMiddleware'den geçmesini sağlar.
    """

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:  # noqa: BLE001 - kullanıcıya anlamlı bir hata dönmek için kasıtlı geniş yakalama
            logger.exception("unhandled exception on %s %s", request.method, request.url.path)
            return JSONResponse(status_code=502, content={"detail": f"Beklenmeyen bir hata oluştu: {exc}"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title="4keys", description="Algoritmik kripto trading platformu", lifespan=lifespan)

# Sıra önemli: CORSMiddleware önce eklenir (dıştaki katman), hata yakalama
# middleware'i sonra eklenir (içteki katman) — böylece hata yanıtları da
# CORS işleminden geçer. Starlette `add_middleware` her çağrıda listenin
# başına ekler; bu yüzden SONRA eklenen İÇTE çalışır.
app.add_middleware(UnhandledExceptionMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(screener.router)
app.include_router(ml.router)
app.include_router(engine.router)
app.include_router(dca.router)
app.include_router(strategy.router)
app.include_router(portfolio.router)
app.include_router(trading.router)
app.include_router(bank.router)
app.include_router(backtest.router)
app.include_router(scheduler.router)
app.include_router(db_routes.router)
app.include_router(security.router)
app.include_router(bist.router)
app.include_router(macro.router)
app.include_router(orderbook.router)
app.include_router(rl.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
