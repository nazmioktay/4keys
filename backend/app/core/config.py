from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    exchange_id: str = "binance"
    quote_currency: str = "USDT"
    market_type: str = "future"  # "future" (USDT-M perpetuals) or "spot"
    candle_timeframe: str = "4h"
    candle_lookback: int = 200
    screener_top_n: int = 10

    # --- Feature snapshot biriktirme (LSTM/RL için ileride kullanılacak
    # zaman serisi veri seti) --- Virgülle ayrılmış sembol listesi; her
    # tarama döngüsünde bu sembollerin ML özellik vektörü (bkz.
    # app.ml.features.FEATURE_COLUMNS) feature_snapshots tablosuna kaydedilir.
    feature_snapshot_symbols: str = "BTC/USDT:USDT"

    @property
    def feature_snapshot_symbols_list(self) -> list[str]:
        return [s.strip() for s in self.feature_snapshot_symbols.split(",") if s.strip()]
    default_starting_equity: float = 1000.0

    # --- CORS ---
    # Virgülle ayrılmış izinli origin listesi. Varsayılan yerel geliştirme
    # (Vite dev server) içindir; production'da .env üzerinden gerçek
    # frontend domain'i (ör. https://app.4kyonetim.com.tr) ile değiştirilmelidir.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

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

    # --- Periyodik zamanlayıcı (Modül: Screener + motorların otomatik döngüsü) ---
    scheduler_enabled: bool = True
    screener_refresh_seconds: int = 60
    engine_cycle_seconds: int = 300

    # --- Kalıcı veritabanı (Modül: TimescaleDB/PostgreSQL) ---
    # Boş bırakılırsa tamamen devre dışıdır; sistem bellek içi durumla
    # (mevcut davranış) çalışmaya devam eder. Örnek:
    # postgresql+psycopg2://user:pass@localhost:5432/fourkeys
    database_url: str = ""

    # --- Güvenlik protokolü (Modül: Kripto Bot Rehberi Bölüm 9) ---
    # Günlük/oturum drawdown bu yüzdeyi aşarsa kill switch OTOMATİK devreye girer
    # (bkz. app.security.kill_switch) — tüm yeni pozisyon açma girişimleri durur.
    kill_switch_daily_drawdown_pct: float = 15.0
    # Canlı emirden önce Binance'ten API anahtarının çekim izninin kapalı
    # olduğu doğrulanır; doğrulama başarısız olursa (ör. borsa erişilemedi)
    # varsayılan olarak temkinli davranılıp emir engellenir.
    require_api_key_permission_check: bool = True

    # --- BIST/VIOP (Denizbank AlgoLab) ---
    # AlgoLab API başvurunuzdan aldığınız değerlerle doldurun. Kimlik
    # doğrulama iki adımlıdır (kullanıcı adı/şifre -> SMS/e-posta kodu),
    # bkz. /bist/login ve /bist/login/verify.
    algolab_base_url: str = "https://www.algolab.com.tr/api"
    algolab_api_key: SecretStr = SecretStr("")
    algolab_username: str = ""
    algolab_password: SecretStr = SecretStr("")
    enable_bist_trading: bool = False  # Binance'teki enable_live_trading ile aynı mantık — ikinci güvenlik kapısı

    model_config = SettingsConfigDict(env_prefix="FOURKEYS_", env_file=".env")


settings = Settings()
