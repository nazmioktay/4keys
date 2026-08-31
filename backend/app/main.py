from fastapi import FastAPI

from app.api.routes import backtest, bank, dca, engine, ml, portfolio, screener, strategy, trading

app = FastAPI(title="4keys", description="Algoritmik kripto trading platformu")

app.include_router(screener.router)
app.include_router(ml.router)
app.include_router(engine.router)
app.include_router(dca.router)
app.include_router(strategy.router)
app.include_router(portfolio.router)
app.include_router(trading.router)
app.include_router(bank.router)
app.include_router(backtest.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
