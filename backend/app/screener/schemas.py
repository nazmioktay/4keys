from pydantic import BaseModel


class ScreenerResult(BaseModel):
    symbol: str
    score: float
    close: float
    rsi: float
    trend: str  # "up" | "down"
