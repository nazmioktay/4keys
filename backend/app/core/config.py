from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    exchange_id: str = "binance"
    quote_currency: str = "USDT"
    market_type: str = "future"  # "future" (USDT-M perpetuals) or "spot"
    candle_timeframe: str = "4h"
    candle_lookback: int = 200
    screener_top_n: int = 10

    # --- ML eğitimi için ayrı zaman dilimi/geçmiş derinliği ---
    # Screener'ın canlı görüntülediği candle_timeframe/candle_lookback'ten
    # BİLİNÇLİ OLARAK ayrı tutulur: screener 4h'de kalmaya devam ederken,
    # eğitim veri seti daha ince taneli (1h) ve çok daha derin bir geçmişle
    # (pagination ile, bkz. app/exchanges/binance.py) kurulur — az veriyle
    # yaşanan overfitting sorununu (bkz. README "Faz B — LSTM" notu) daha
    # fazla satırla azaltmak için.
    ml_train_timeframe: str = "1h"
    ml_train_lookback: int = 10000

    # --- Eğitim sembol seçimi: BTC-öncelikli + uyumluluk filtresi ---
    # Önceden eğitim evreni doğrudan screener'ın Top-N Long + Top-N Short
    # çıktısıydı — bu liste yalnızca kısa vadeli teknik skora göre seçiliyor,
    # likidite/hacim eşiği yok. Kullanıcı bunun "zayıf seçilmiş bir grup"
    # olduğunu ve öncelikle BTC/USDT üzerinden eğitim yapılıp diğer
    # sembollerin yalnızca bu eğitime UYUMLU olmaları hâlinde katılmaları
    # gerektiğini belirtti (bkz. sohbet). Uyumluluk = BTC ile getiri
    # korelasyonu (aynı rejimde hareket ediyor mu) + minimum likidite.
    ml_primary_symbol: str = "BTC/USDT:USDT"
    ml_train_max_symbols: int = 5
    ml_min_correlation_with_primary: float = 0.4
    ml_min_quote_volume_24h: float = 5_000_000.0

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
    macro_refresh_seconds: int = 21600  # 6 saat — makro veriler (VIX, altın, faiz oranları vb.) günde birkaç kez yeterli
    orderbook_refresh_seconds: int = 1800  # 30 dakika — emir defterinin ANLIK görüntüsü, geçmişi yoktur (bkz. app.orderbook)

    # --- Otomatik yeniden eğitim (bkz. app.scheduler.jobs.job_auto_retrain) ---
    # Neden 24 saat: ml_train_lookback (10.000 saatlik mum, ~1.14 yıl) ile
    # kıyaslandığında bir günde biriken ~24 yeni bar, toplam veri setinin
    # ~%0.24'ü — bundan daha sık yeniden eğitmek (ör. saatlik) hesaplama
    # maliyetini artırır ama modelin öğrendiği dağılımı neredeyse hiç
    # değiştirmez. Bu, gerçek bir "backtest ile bulunmuş" optimum DEĞİL —
    # canlı piyasa verisine bu ortamdan erişilemediği için ampirik olarak
    # doğrulanamadı; rejim değişikliklerini makul bir gecikmeyle yakalayan,
    # sektörde yaygın bir varsayılan kabul edilmelidir. `enabled=False`
    # yapılıp elle (`/ml/train`) tetiklenmeye devam edilebilir.
    ml_auto_retrain_enabled: bool = False
    ml_auto_retrain_seconds: int = 86400  # 24 saat

    # --- Ücretsiz makro veri kaynakları (bkz. app.macro.data) ---
    # FRED (ABD Merkez Bankası) API anahtarı — ücretsiz, anında alınır:
    # https://fred.stlouisfed.org/docs/api/api_key.html . Boş bırakılırsa
    # Fed faiz oranı verisi atlanır (diğer kaynaklar etkilenmez).
    fred_api_key: str = ""

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
