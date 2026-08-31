from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    exchange_id: str = "binance"
    quote_currency: str = "USDT"
    market_type: str = "future"  # "future" (USDT-M perpetuals) or "spot"
    candle_timeframe: str = "4h"
    candle_lookback: int = 200
    screener_top_n: int = 10
    default_starting_equity: float = 1000.0

    class Config:
        env_prefix = "FOURKEYS_"


settings = Settings()
