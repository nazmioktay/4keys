from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    exchange_id: str = "binance"
    quote_currency: str = "USDT"
    market_type: str = "future"  # "future" (USDT-M perpetuals) or "spot"
    candle_timeframe: str = "4h"
    candle_lookback: int = 200
    screener_top_n: int = 10
    default_starting_equity: float = 1000.0

    # --- Binance canlı işlem (Modül 6 — API hazırlığı) ---
    # ASLA koda veya git'e yazmayın; yalnızca ortam değişkeni / .env dosyasından okunur.
    binance_api_key: SecretStr = SecretStr("")
    binance_api_secret: SecretStr = SecretStr("")
    binance_testnet: bool = True  # varsayılan güvenli: gerçek para riske girmez
    enable_live_trading: bool = False  # ikinci güvenlik kapısı — açıkça true yapılmadıkça emir gönderilmez

    # --- Denizbank Açık Bankacılık (bakiye görüntüleme) ---
    # Denizbank'ın TPP/fintech geliştirici portalından alınan gerçek değerlerle doldurulmalı.
    denizbank_base_url: str = ""
    denizbank_client_id: str = ""
    denizbank_client_secret: SecretStr = SecretStr("")
    denizbank_redirect_uri: str = "http://localhost:8000/bank/denizbank/callback"

    model_config = SettingsConfigDict(env_prefix="FOURKEYS_", env_file=".env")


settings = Settings()
