from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    exchange_id: str = "binance"
    quote_currency: str = "USDT"
    market_type: str = "future"  # "future" (USDT-M perpetuals) or "spot"
    candle_timeframe: str = "4h"
    candle_lookback: int = 200
    screener_top_n: int = 5

    # --- Screener ön-filtresi (hacim + fiyat tabanı) ---
    # Önceden `scan_market` PİYASADAKİ TÜM sembolleri (yüzlerce) tek tek
    # `fetch_ohlcv` ile taramaya çalışıyordu — bu, taramanın kendi periyodundan
    # (screener_refresh_seconds) çok daha uzun sürmesine, hatta hiç
    # bitmemesine yol açıyordu (bkz. APScheduler "maximum number of running
    # instances reached" logları). Şimdi önce TEK bir toplu istekle
    # (`Exchange.fetch_tickers`) tüm sembollerin hacmi/fiyatı alınıyor;
    # pahalı gösterge hesaplaması yalnızca bu ön-filtreyi geçen KÜÇÜK bir
    # alt kümede çalışıyor.
    screener_min_price: float = 0.1  # bu fiyatın altındaki semboller elenir
    screener_volume_top_pct: float = 20.0  # 24s işlem hacmine göre en yüksek %X (fiyat tabanını geçenler arasında)

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

    # --- XGBoost + LSTM ensemble (opsiyonel, opt-in) ---
    # LSTM'in BTC-only sınamalarda (bkz. README "Faz B" notu) rastgele
    # seviyenin belirgin üzerine çıktığı doğrulandıktan sonra eklendi;
    # yine de LSTM kalitesi henüz her sembol/zaman diliminde ayrı ayrı
    # doğrulanmadığından varsayılan KAPALI — kullanıcı açıkça açmalı.
    ensemble_lstm_enabled: bool = False

    # `river` ARFClassifier tabanlı online model (bkz. app.ml.online_model)
    # — prequential değerlendirmede BTC-only veride overall_balanced_accuracy
    # ~%49.7 gösterdi (XGBoost/LSTM'den daha iyi), yine de sembol/zaman
    # dilimine göre ayrı ayrı doğrulanmadığından varsayılan KAPALI.
    ensemble_online_enabled: bool = False

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
    # 60sn'den 300sn'e çıkarıldı: hacim ön-filtresiyle (bkz. yukarıdaki
    # screener_min_price/screener_volume_top_pct) tarama artık TÜM piyasa
    # yerine yalnızca en yüksek hacimli ~%20'yi işliyor, ama yine de her
    # sembol için bir ağ isteği (fetch_ohlcv) gerektirdiğinden 60sn hâlâ
    # dar olabilir — bkz. `job_refresh_screener`'ın önceki periyottan uzun
    # sürüp "maximum number of running instances reached" ile kilitlenmesi.
    screener_refresh_seconds: int = 300
    engine_cycle_seconds: int = 300
    macro_refresh_seconds: int = 21600  # 6 saat — makro veriler (VIX, altın, faiz oranları vb.) günde birkaç kez yeterli
    orderbook_refresh_seconds: int = 1800  # 30 dakika — emir defterinin ANLIK görüntüsü, geçmişi yoktur (bkz. app.orderbook)

    # --- Otomatik yeniden eğitim (bkz. app.scheduler.jobs) ---
    # Aralık, sabit bir takvim süresi yerine HESAPLANIR (bkz.
    # `app.scheduler.jobs.compute_auto_retrain_interval_seconds`): eğitim
    # penceresinin (`ml_train_lookback` bar, `ml_train_timeframe`) ne kadarı
    # YENİ veriyle değişmiş olmalı ki yeniden eğitmeye değsin. Varsayılan
    # `ml_auto_retrain_refresh_fraction=0.05` (%5) ile, varsayılan
    # ml_train_lookback=10000 / ml_train_timeframe=1h için ham hesap
    # 10000 saat * 3600sn * 0.05 = 1.800.000sn (~20.8 gün) verir — ama bu
    # kullanıcı için çok uzun bulunduğundan `ml_auto_retrain_max_seconds`
    # (varsayılan 7 gün) ile YUKARI SINIRLANIR: ham hesap tavanı aşarsa
    # tavan kullanılır, en geç haftada bir yeniden eğitim garanti edilir.
    # `%5` oranının kendisi bir "backtest ile bulunmuş" optimum DEĞİL —
    # sektörde yaygın "pencerenin ~%5'i tazelenince yeniden eğit" pratiğine
    # dayanan, veri hacmine göre GEREKÇELENDİRİLMİŞ bir varsayılan (önceki
    # sabit 24 saatten farkı: `ml_train_lookback`/`ml_train_timeframe`
    # değişirse aralık da otomatik ölçeklenir, tavana çarpmadığı sürece).
    # `ml_auto_retrain_seconds` açıkça verilirse (None değilse) bu
    # hesaplamanın (taban VE tavan dahil) YERİNE geçer.
    #
    # Hangi modeller: XGBoost + (varsa) meta-label HER ZAMAN bu job'a dahildir
    # (canlı karar motorunun birincil modeli). LSTM/PatchTST/online/regime
    # modelleri yalnızca ZATEN EN AZ BİR KEZ elle eğitilmişse (disk'te dosyası
    # varsa) veya ilgili ensemble bayrağı açıksa otomatik yenilenir — hiç
    # kullanılmayan bir modeli sıfırdan otomatik eğitmeye başlamaz.
    ml_auto_retrain_enabled: bool = True
    ml_auto_retrain_seconds: int | None = None  # None = compute_auto_retrain_interval_seconds() kullanılır
    ml_auto_retrain_refresh_fraction: float = 0.05
    ml_auto_retrain_max_seconds: int = 604800  # 7 gün — "en geç haftada bir" üst sınırı

    # --- Eğitim kalite kapısı ---
    # Bir modelin out-of-sample (veya online modelde prequential) dengeli
    # doğruluğu (balanced_accuracy) bu eşiğin ALTINDAYSA, model diske
    # KAYDEDİLMEZ (önceden eğitilmiş — varsa — model dosyası KORUNUR, canlı
    # karar motoru eski/iyi modeli kullanmaya devam eder) ve sonuç
    # (`TrainingResult.accepted`/`rejection_reason` vb.) açıkça "reddedildi"
    # olarak işaretlenir. 0.37, 3 sınıflı (long/short/nötr) bir problemde
    # rastgele seviyenin (1/3 ≈ 0.333) hemen üzerinde, "en azından rastgele
    # tahminden biraz daha iyi" için makul bir taban — ampirik olarak
    # optimize edilmiş bir değer DEĞİL.
    ml_min_balanced_accuracy: float = 0.37

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
