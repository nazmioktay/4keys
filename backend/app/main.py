from fastapi import FastAPI

from app.api.routes import dca, engine, ml, portfolio, screener, strategy

app = FastAPI(title="4keys", description="Algoritmik kripto trading platformu")

app.include_router(screener.router)
app.include_router(ml.router)
app.include_router(engine.router)
app.include_router(dca.router)
app.include_router(strategy.router)
app.include_router(portfolio.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
