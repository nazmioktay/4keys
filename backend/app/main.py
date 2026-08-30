from fastapi import FastAPI

from app.api.routes import dca, engine, ml, screener

app = FastAPI(title="4keys", description="Algoritmik kripto trading platformu")

app.include_router(screener.router)
app.include_router(ml.router)
app.include_router(engine.router)
app.include_router(dca.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
