from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import backtest, bank, dca, engine, ml, portfolio, scheduler, screener, strategy, trading
from app.scheduler.scheduler import start_scheduler, stop_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title="4keys", description="Algoritmik kripto trading platformu", lifespan=lifespan)

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
