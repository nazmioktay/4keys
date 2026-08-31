# 4keys

Kendi algoritmik kripto trading sistemimiz. 3Commas benzeri botlardaki eksiklikleri
gidermek için tasarlanan, aşağıdaki modüllerden oluşan bir platform:

1. **Tarama motoru (Screener)** — Borsadaki tüm paritelerde teknik analiz çalıştırıp
   Long ve Short için en güçlü ilk 10 sinyali üretir. *(İlk geliştirilen modül)*
2. **ML/Sinyal modülü** — Tarama sonuçları üzerinde makine öğrenmesi ile yön tahmini
   yapıp işlemleri otomatik açıp kapatır.
3. **DCA optimizasyon motoru** — Manuel DCA botu parametrelerini (base order,
   deviation, take profit vb.) geçmiş veriye göre optimize eder.
4. **Hızlı strateji motoru** — TradingView'a ihtiyaç duymadan JSON/DSL tabanlı
   strateji tanımlayıp canlıya alma.
5. **Çoklu borsa desteği** — Binance ile başlayıp BIST, VIOP ve diğer dünya
   borsalarına genişleyecek soyutlama katmanı.
6. **Portföy / risk yönetimi** — Ana para yönetimi kurallarını belirleyen katman.

## Mimari

- **Backend:** Python 3.11+, FastAPI
- **Borsa erişimi:** `ccxt` üzerinden soyutlanmış `Exchange` arayüzü — yeni bir
  borsa eklemek `exchanges/` altında yeni bir adapter yazmak demektir.
- **Veri/indikatörler:** `pandas` tabanlı teknik gösterge hesaplama (EMA, RSI, MACD — harici bağımlılık yok).
- **ML:** `scikit-learn` `MLPClassifier` (çok katmanlı yapay sinir ağı) ile long/short/neutral yön sınıflandırması.

## Durum

### Modül 1 — Screener ✅
Binance USDT-M vadeli paritelerinde RSI, EMA trend, MACD ve hacim momentumunu
birleştiren bir skor ile Long/Short Top 10 listesi üretiyor.

### Modül 2 — ML Sinyal + Otomatik Karar Motoru ✅ (paper-trading)
- `app/ml/features.py` — göstergelerden normalize edilmiş özellik vektörü
- `app/ml/labeling.py` — N mum sonraki getiriye göre long/short/neutral etiketleme
- `app/ml/model.py` — StandardScaler + MLPClassifier pipeline, eğit/kaydet/yükle
- `app/ml/train.py`, `app/ml/dataset.py` — çoklu sembolden eğitim seti kurup modeli eğitme
- `app/engine/decision.py` — model tahminini mevcut pozisyonla birleştirip
  aksiyon üretir (open_long / open_short / close / hold)
- `app/engine/positions.py` — **paper-trading (simülasyon) pozisyon defteri**

**Önemli güvenlik notu:** Karar motoru şu an yalnızca simülasyon modunda çalışır,
gerçek borsaya emir göndermez. Gerçek parayla otomatik işlem açma/kapama, API
anahtarı yönetimi ve kullanıcının açık onayını gerektiren ayrı, bilinçli olarak
eklenmemiş bir katmandır — bu bilerek bir sonraki adıma bırakılmıştır.

**ML metodolojisi yükseltmesi** ("Kripto Bot Tam Rehber" Bölüm 2.4-2.5'e göre):
- `app/ml/labeling.py::triple_barrier_labels` — sabit eşikli "N mum sonra ne
  oldu?" etiketlemesine ek olarak, kâr hedefi/stop-loss/zaman aşımı
  bariyerlerinden hangisi ÖNCE tetiklenirse ona göre etiketleyen, gerçek
  işlem mantığını daha doğru yansıtan bir yöntem (mum içi high/low kullanır).
  `POST /ml/train`'de `labeling_method: "triple_barrier"` ile seçilir.
- `app/ml/model.py` — model artık `CalibratedClassifierCV` ile (Platt
  scaling / isotonic regression) **kalibre edilmiş** olasılık üretir;
  kalibre edilmemiş bir "%60 güven" gerçek bir olasılık değildir ve Kelly
  kriterine (Modül: Portföy) doğrudan verilirse pozisyon boyutları
  sistematik olarak hatalı çıkar. Eğitim seti çok küçük/dengesizse (bir
  sınıfta yetersiz örnek) kalibrasyon otomatik ve güvenli şekilde atlanır.
- `app/ml/meta_label.py` — **meta-labeling**: sabit ağırlıklı bir ensemble
  yerine, ikinci bir modelin "birincil modelin sinyaline gir/girme" kararı
  verdiği yaklaşım. `POST /ml/train-meta` ile eğitilir; eğitilmişse
  `DecisionEngine` her açılış sinyalini önce meta modele danışır — meta
  model "güvenme" derse pozisyon açılmaz, `hold`a düşülür. Bu tamamen
  opsiyoneldir; meta model eğitilmemişse sistem eskisi gibi çalışır.

### Çalıştırma

**Veritabansız (varsayılan, bellek içi):**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Kalıcı veritabanıyla (Docker Compose — TimescaleDB + backend):**
```bash
cp backend/.env.example backend/.env   # gerekli anahtarları/ayarları doldurun
docker compose up
```

| Endpoint | Açıklama |
|---|---|
| `GET /screener/top?direction=long\|short&limit=10` | Long/Short Top N tarama sonucu |
| `POST /ml/train` | Birincil modeli eğitir (`labeling_method`: `"threshold"`\|`"triple_barrier"`, `calibrate`: bool, `calibration_method`: `"sigmoid"`\|`"isotonic"`) |
| `POST /ml/train-meta` | Meta-label modelini eğitir (önce `/ml/train` çağrılmış olmalı) |
| `GET /ml/predict?symbol=BTC/USDT:USDT` | Yön + kalibre güven tahmini (meta model varsa `meta_act`/`meta_confidence` de döner) |
| `POST /engine/run-cycle` | Screener top listesi üzerinde bir karar döngüsü çalıştırır (paper-trading) |
| `GET /engine/status` | Açık paper pozisyonlar ve kapanan işlem geçmişi |
| `POST /dca/optimize` | Verilen sembol/sermaye için en iyi DCA parametre kombinasyonlarını bulur |

### Modül 3 — DCA Optimizasyon Hesaplayıcısı ✅
- `app/dca/simulator.py` — bir DCA botunun (base order + averaging orders +
  take profit + opsiyonel stop loss) geçmiş fiyat serisi üzerindeki
  davranışını mum mum simüle eder; kapanan işlem sayısı, kazanma oranı,
  toplam getiri %, maksimum drawdown % ve kullanılan maksimum sermayeyi hesaplar.
- `app/dca/optimizer.py` — deviation, deviation/order-size çarpanları, safety
  order sayısı ve take profit için bir parametre ızgarasını (grid search)
  tarar; her kombinasyon için base order büyüklüğünü, tüm averaging order'lar
  teorik olarak dolsa dahi verilen sermayeyi aşmayacak şekilde otomatik
  hesaplar, ardından sonuçları seçilen hedefe (`profit`,
  `profit_over_drawdown`, `win_rate`) göre sıralar.

`POST /dca/optimize` örnek gövde:
```json
{
  "symbol": "BTC/USDT:USDT",
  "balance": 500,
  "direction": "long",
  "objective": "profit_over_drawdown",
  "top_n": 5
}
```

### Modül 4 — TradingView'sız Strateji Motoru ✅
- `app/strategy/schemas.py` — JSON kural ağacı: `compare` (örn. `rsi < 30`),
  `cross` (örn. `ema_fast`, `ema_slow`'u yukarı keser), `and`/`or` ile
  birleştirme. Pine Script yazmaya gerek yok.
- `app/strategy/evaluator.py` — kural ağacını tek bir mum için değerlendirir
- `app/strategy/engine.py` — entry/exit kuralları + opsiyonel take-profit/stop-loss
  ile geçmiş veri üzerinde tam backtest çalıştırır
- `app/strategy/examples.py` — hazır örnekler (RSI dip alımı, EMA altın
  kesişim, MACD momentum short) — `GET /strategy/examples` ile alınıp
  doğrudan değiştirilip denenebilir

`POST /strategy/backtest` örnek gövde:
```json
{
  "symbol": "BTC/USDT:USDT",
  "strategy": {
    "name": "RSI Aşırı Satım Sıçraması",
    "direction": "long",
    "entry": {"type": "compare", "left": {"indicator": "rsi"}, "op": "lt", "right": {"value": 30}},
    "exit": {"type": "compare", "left": {"indicator": "rsi"}, "op": "gt", "right": {"value": 55}},
    "take_profit_pct": 4.0,
    "stop_loss_pct": 3.0
  }
}
```

| Endpoint | Açıklama |
|---|---|
| `GET /strategy/examples` | Hazır strateji örnekleri (kopyalayıp değiştirilebilir) |
| `POST /strategy/backtest` | Verilen JSON stratejiyi geçmiş veri üzerinde test eder |

### Modül 5 — Portföy / Risk Yönetimi ✅
- `app/portfolio/risk_manager.py` — saf fonksiyonlar:
  - `calculate_position_size` — equity, giriş fiyatı, stop-loss fiyatı ve
    işlem başına risk yüzdesinden pozisyon boyutunu geriye hesaplar (SL'e
    çarpılırsa kaybedilecek tutar tam olarak istenen risk kadar olur)
  - `evaluate_risk` — önerilen bir pozisyonu şu kurallara karşı denetler:
    işlem başına risk, toplam portföy maruziyeti, sembol bazlı maruziyet,
    maksimum eşzamanlı pozisyon sayısı, günlük/oturum zarar limiti (circuit
    breaker) — mümkün olduğunda reddetmek yerine boyutu güvenli sınıra küçültür
- `app/portfolio/manager.py` — `PortfolioManager`: equity, açık pozisyonlar
  ve gerçekleşen kâr/zararı tutan, yukarıdaki kuralları uygulayan durum
  yöneticisi
- **Entegrasyon:** `app/engine/decision.py`'deki ML karar motoru artık
  opsiyonel bir `PortfolioManager` alıyor; verildiğinde her açılış sinyali
  ham haliyle uygulanmıyor, risk kurallarından geçip boyutlandırılıyor veya
  reddediliyor (`"blocked"` aksiyonu + sebep). `/engine/run-cycle` bu
  entegrasyonu varsayılan olarak kullanıyor — yani ana para yönetimi artık
  ML motorunun bir parçası, ayrı bir hesap makinesi değil.

**Kelly kriteri (çeyrek/yarım/tam) pozisyon boyutlandırma:**
- `RiskRules.position_sizing_method`: `"fixed_risk"` (klasik, varsayılan) veya
  `"kelly"`.
- `app/portfolio/risk_manager.py::kelly_fraction` — full Kelly formülü
  (`f* = p - q/b`); beklenen değeri negatif çıkan "kenarlar" için asla
  negatif pozisyon önermez, 0 döner.
- `kelly_multiplier`: çeyrek Kelly=`0.25`, yarım Kelly=`0.5` (varsayılan,
  önerilen — full Kelly pratikte çok volatildir), tam Kelly=`1.0`.
- `max_kelly_fraction_pct`: formül ne derse desin bir işleme ayrılacak
  sermayenin üst güvenlik sınırı (istatistikler az örneklemli/yanlış
  olabileceği için).
- **Otomatik/canlı entegrasyon:** Kelly istatistikleri (kazanma oranı,
  ortalama kazanç/kayıp) varsayılan olarak `PortfolioManager`'ın **kendi
  kapanmış işlem geçmişinden** otomatik hesaplanır — yani otomatik alım
  satım motoru (ML karar motoru / DCA / stratejiler, hepsi aynı
  `PortfolioManager`'dan geçiyor) canlı performansına göre kendi kendini
  ayarlar. `kelly_min_trades` (varsayılan 20) kadar kapanmış işlem
  birikene kadar güvenli tarafta kalınıp otomatik olarak `fixed_risk`'e
  düşülür. `/backtest/run` raporundan gelen istatistikleri "önsel" olarak
  denemek isterseniz `/portfolio/kelly-size` ile bağımsız hesaplayabilir,
  veya `propose_open(..., kelly_stats_override=...)` ile programatik
  olarak geçebilirsiniz.

| Endpoint | Açıklama |
|---|---|
| `GET /portfolio/status` | Equity, açık pozisyonlar, kapanan işlemler, aktif kurallar, işlem istatistikleri |
| `PUT /portfolio/rules` | Risk kurallarını (fixed_risk veya Kelly) günceller |
| `POST /portfolio/reset` | Portföyü verilen sermaye/kurallarla sıfırdan başlatır |
| `POST /portfolio/position-size` | Risk yüzdesi + SL mesafesinden pozisyon boyutu hesaplar (fixed_risk) |
| `GET /portfolio/trade-stats` | Portföyün kendi geçmişinden hesaplanan kazanma oranı / ort. kazanç-kayıp |
| `POST /portfolio/kelly-size` | Çeyrek/yarım/tam Kelly'ye göre bağımsız pozisyon boyutu hesaplar |
| `POST /portfolio/risk-check` | Durumsuz "ne olurdu" risk kontrolü (paylaşılan portföyü etkilemez) |

### Modül 6 — Binance & Denizbank API Hazırlığı ✅
**Binance canlı işlem** (`app/exchanges/binance.py`, `app/trading/`):
- `BinanceExchange` artık opsiyonel `api_key`/`api_secret` ile kimlik
  doğrulamalı çalışabiliyor: bakiye, pozisyon, açık emirler, emir gönderme/iptal.
- Piyasa verisi (screener/ML/DCA/strateji) için kullanılan `get_exchange()`
  hâlâ tamamen kimlik doğrulamasız ve gerçek veriye bakıyor — anahtarlarınız
  bu modüllere hiç dokunmuyor.
- Gerçek emir göndermek **üç ayrı güvenlik kapısından** geçmek zorunda
  (`app/trading/executor.py::place_live_order`):
  1. Ortam değişkeninde `FOURKEYS_ENABLE_LIVE_TRADING=true`
  2. İstek gövdesinde `confirm: true` (her çağrıda ayrı ayrı)
  3. `.env`'de tanımlı Binance API anahtarları
  Üçünden biri eksikse istek 409 ile reddedilir. `FOURKEYS_BINANCE_TESTNET`
  varsayılan `true` — gerçek hesaba geçmeden önce testnet'te deneyin.
- API anahtarları **kesinlikle** koda/git'e yazılmaz; yalnızca `.env`
  dosyasından (`.gitignore`'da) okunur — bkz. `backend/.env.example`.

**Denizbank Açık Bankacılık** (`app/bank/`) — bakiye/hesap görüntüleme:
- Denizbank'ın tek ve belgelenmiş bir "trading API"si yok; hesap bilgisine
  erişim Türkiye'nin BDDK düzenlemesindeki Açık Bankacılık çerçevesi
  üzerinden, bir TPP/fintech olarak kayıt olup OAuth2 onay akışıyla yapılır.
- `DenizbankOpenBankingClient`, bu standart akışın (yetkilendirme URL'i ->
  kod değişimi -> access token -> hesap/bakiye sorgusu) **genel kalıbını**
  uyguluyor. Uç nokta yolları (`/oauth2/authorize`, `/oauth2/token`,
  `/accounts`, ...) Türkiye Açık Bankacılık ekosisteminde yaygındır ama
  **Denizbank'a özgü kesin değerler değildir** — TPP başvurunuz onaylanıp
  geliştirici portalından gerçek `base_url`/uç noktaları aldığınızda
  `.env` ve gerekirse `endpoint_overrides` ile güncelleyin.
- Token'lar şu an bellek içi (süreç yeniden başlarsa kaybolur, kullanıcı
  onay akışını tekrarlar) — üretimde şifrelenmiş bir secrets store'a taşınmalı.

| Endpoint | Açıklama |
|---|---|
| `GET /trading/balance` | Gerçek Binance bakiyesi (kimlik bilgisi gerekir) |
| `GET /trading/positions` | Gerçek açık pozisyonlar |
| `POST /trading/order` | Gerçek emir gönderir — 3 güvenlik kapısı aktif |
| `GET /bank/denizbank/authorize` | Onay için ziyaret edilecek URL'i döner |
| `GET /bank/denizbank/callback` | Yetkilendirme kodunu token'a çevirir |
| `GET /bank/denizbank/accounts` | Hesap listesi |
| `GET /bank/denizbank/balances/{account_id}` | Hesap bakiyesi |

### Modül 7 — Güçlü Backtest Motoru (Otomatik Veri Yeterliliği + Train/Test) ✅
DCA ve JSON-strateji motorlarını ortak bir çatı altında birleştiren, tek
uçlu, "gerçekten güvenilir mi?" sorusuna cevap veren bir backtest sistemi.

- `app/backtest/data.py` — `fetch_full_history`: Binance gibi borsaların
  tek istekte verdiği sınırlı mum sayısını (`since` parametresini ilerleterek)
  sayfalayıp istenen kadar (veya borsada mevcut olan kadar) geçmişi birleştirir.
- `app/backtest/runner.py::_discover_sufficient_history` — **"geçmiş veri
  miktarını öğrenerek oluştur"**: sabit bir mum sayısı varsaymak yerine, az
  veriyle (`initial_candles`) başlayıp verilen strateji/DCA parametreleriyle
  kaç kapanan işlem ürettiğine bakar; hedef işlem sayısına (`min_trades`,
  istatistiksel anlamlılık için varsayılan 30) ulaşılana, borsanın geçmişi
  tükenene ya da `max_candles` sınırına varılana kadar veriyi ikişer katına
  çıkararak genişletir. Sonuçta kaç mumun gerçekten yeterli olduğunu ve
  yeterli olup olmadığını raporlar.
- `app/backtest/metrics.py` — zengin performans metrikleri: toplam getiri,
  CAGR, **Sharpe**, **Sortino**, **Calmar**, **profit factor**, kazanma
  oranı, ortalama kazanç/kayıp, expectancy, maksimum drawdown — işlem
  sıklığından yıllıklaştırma otomatik hesaplanır.
- `app/backtest/runner.py::run_backtest_report` — keşfedilen veriyi
  kronolojik olarak **eğitim (in-sample) / test (out-of-sample)** olarak
  ikiye böler, her ikisi ve tüm veri için ayrı metrik hesaplar; eğitimde
  kârlı ama testte zararlıysa veya test performansı eğitimin çok altında
  kalıyorsa **aşırı uyum (overfitting) uyarısı** üretir.
- DCA ve strateji motorları tek bir arayüzden (`_simulate`) çağrıldığı için
  aynı backtest altyapısı ikisinde de kullanılıyor — kod tekrarı yok.

`POST /backtest/run` örnek gövde (DCA veya strateji, tam olarak biri):
```json
{
  "symbol": "BTC/USDT:USDT",
  "strategy": {
    "name": "RSI Aşırı Satım Sıçraması",
    "direction": "long",
    "entry": {"type": "compare", "left": {"indicator": "rsi"}, "op": "lt", "right": {"value": 30}},
    "take_profit_pct": 2.0
  },
  "min_trades": 30,
  "max_candles": 5000,
  "train_ratio": 0.7
}
```
Yanıt: `data_sufficiency` (kaç mum kullanıldı, yeterli miydi, neden),
`train_metrics`, `test_metrics`, `full_period_metrics`, `warnings`.

| Endpoint | Açıklama |
|---|---|
| `POST /backtest/run` | DCA veya strateji için otomatik veri keşifli, train/test ayrımlı tam backtest raporu |

### Modül 8 — Screener ve Motorların Periyodik Zamanlayıcıya Bağlanması ✅
Artık kullanıcı her seferinde `/screener/top` veya `/engine/run-cycle`'ı elle
çağırmak zorunda değil — uygulama ayağa kalktığı anda arka planda otomatik
çalışan bir zamanlayıcı (APScheduler) devreye giriyor:

- `app/screener/service.py` — screener önbelleğini (`refresh()`) hem API hem
  zamanlayıcı için paylaşılan tek kaynak haline getirdi.
- `app/engine/service.py` — `/engine/run-cycle`'daki mantığı `run_cycle_once()`
  olarak dışarı çıkardı; **screener önbelleğini tekrar taramadan** yeniden
  kullanıyor (iki job aynı veriyi paylaşıyor, gereksiz borsa çağrısı yok).
- `app/scheduler/` — `job_refresh_screener` (varsayılan her 60 saniyede bir)
  ve `job_run_engine_cycle` (varsayılan her 300 saniyede bir) işlerini
  `BackgroundScheduler` ile FastAPI'nin `lifespan`'ına bağladı: uygulama
  başlarken otomatik başlıyor, kapanırken düzgün kapanıyor.
- **Dayanıklılık:** Bir job'daki hata (borsa erişilemedi, model henüz
  eğitilmedi vb.) zamanlayıcı thread'ini asla çökertmez — yakalanıp
  `/scheduler/status` üzerinden görülebilir şekilde kaydedilir. Model henüz
  eğitilmemişse bu bir hata değil, "atlandı" olarak işaretlenir.
- `FOURKEYS_SCHEDULER_ENABLED=false` ile tamamen kapatılabilir;
  `FOURKEYS_SCREENER_REFRESH_SECONDS` / `FOURKEYS_ENGINE_CYCLE_SECONDS` ile
  aralıklar ayarlanabilir.

| Endpoint | Açıklama |
|---|---|
| `GET /scheduler/status` | Her job için sonraki/son çalışma zamanı, son sonuç, çalışma/hata sayısı |
| `POST /scheduler/trigger/{job_id}` | Bir job'ı (`screener_refresh` veya `engine_cycle`) beklemeden hemen çalıştırır |

### Modül 9 — Kalıcı Veritabanı Katmanı (TimescaleDB/PostgreSQL) ✅
"Kripto Bot Tam Rehber" Bölüm 3'teki mimariyi izler: `app/db/` altında
SQLAlchemy tabanlı, **tamamen opsiyonel** bir kalıcılık katmanı.

- `app/db/models.py` — rehberle aynı isimlendirme: `ohlcv_raw` (ham mum
  verisi, TimescaleDB varsa hypertable'a çevrilir), `signals` (screener/ML/
  meta'nın ürettiği her tahmin), `trades` (kapanan her işlem — Sharpe/win
  rate/drawdown gibi tüm performans ölçümünün temeli).
- `app/db/session.py` — `FOURKEYS_DATABASE_URL` boşsa katman tamamen
  devre dışıdır, sistem eskisi gibi bellek içi çalışır (geriye dönük
  uyumlu, DB kurulum zorunluluğu yok). Doluysa `init_db()` uygulama
  başlarken tabloları oluşturur ve mümkünse TimescaleDB hypertable'ını kurar
  (düz PostgreSQL/SQLite'ta bu adım sessizce atlanır).
- `app/db/repository.py` — **kalıcılık asla ana işlem akışını bozmaz**:
  yazma fonksiyonları veritabanı kapalı/erişilemez/hatalı olsa bile
  exception fırlatmaz, sadece loglar. Screener taraması, ML tahminleri ve
  kapanan işlemler DB açıkken otomatik olarak buraya yazılır
  (`scanner.py`, `engine/decision.py`, `portfolio/manager.py` içine
  kancalanmıştır).
- `docker-compose.yml` + `backend/Dockerfile` — rehberin Bölüm 5'indeki
  servis haritasının (şu an var olan kısmı): TimescaleDB + backend, tek
  komutla (`docker compose up`) ayağa kalkar.

| Endpoint | Açıklama |
|---|---|
| `GET /db/status` | Veritabanı etkin mi ve bağlantı kuruluyor mu |
| `GET /db/trades?limit=50` | Kalıcı işlem geçmişi (süreç yeniden başlasa da kaybolmaz) |
| `GET /db/signals?limit=50&symbol=&source=` | Kalıcı sinyal geçmişi (`source`: screener\|ml\|meta) |

**Not:** Rehberin Redis "canlı cache" katmanı (son 500 mum, aktif sinyal)
şimdilik eklenmedi — mevcut tek-process mimaride bellek içi önbellekler
(`app/screener/service.py`, `PortfolioManager`) aynı işlevi görüyor; Redis,
sistem çoklu-process/çoklu-sunucuya ölçeklenmeye başladığında gerçek değer
katacak, o aşamaya bırakıldı.

### Modül 10 — Güvenlik Protokolü Sertleştirme (Bölüm 9) ✅
"Kripto Bot Tam Rehber" Bölüm 9'daki kontrol listesinin **kod içinde
gerçekten uygulanabilir** maddeleri:

- `app/security/kill_switch.py` — **kill switch**: manuel (`POST
  /security/kill-switch/activate`) veya **otomatik** (oturum drawdown'u
  `FOURKEYS_KILL_SWITCH_DAILY_DRAWDOWN_PCT`'i, varsayılan %15, aştığında —
  bkz. `PortfolioManager._maybe_trip_kill_switch`) devreye girer. Aktifken:
  `DecisionEngine` yeni pozisyon açmaz (`"blocked"` aksiyonu), zamanlanmış
  `engine_cycle` işi çalışmaz (piyasa verisi bile çekmez), ve
  `place_live_order`/`set_live_leverage` gerçek borsaya hiçbir şey göndermez.
  Açık pozisyonlar otomatik kapatılmaz — kapatma kararı bilinçli olarak
  kullanıcıya bırakılmıştır (bkz. Bölüm 0.1 roller).
- `app/security/safety.py::MAX_LEVERAGE = 3` — Bölüm 9.3'teki "kod içi sabit
  limit" birebir: `.env`/ortam değişkeniyle **değiştirilemez**, sadece kodu
  düzenleyip yeniden deploy ederek değiştirilebilir. `POST /trading/leverage`
  bu tavanı aşan hiçbir isteği kabul etmez.
- `app/security/safety.py::check_withdrawals_disabled` — canlı emirden/kaldıraç
  değişikliğinden önce Binance'e API anahtarının **çekim izninin kapalı**
  olduğu sorulur (Bölüm 9.1, birinci madde); izin açıksa VEYA doğrulama
  başarısız olursa (ör. borsa erişilemedi) varsayılan olarak **temkinli
  davranılıp emir engellenir** ("fail closed").
- `scripts/check_secrets.py` — Bölüm 9.5'teki "commit'te .env/anahtar
  sızıntısı tespiti" için çalıştırılabilir bir tarayıcı; git tarafından
  takip edilen tüm dosyaları API anahtarı deseni, `.env` dosyası ve özel
  anahtar bloğu için tarar. Bu repoya karşı çalıştırıldı, temiz çıktı verdi.

| Endpoint | Açıklama |
|---|---|
| `GET /security/status` | Kill switch durumu, live-trading bayrağı, max kaldıraç, drawdown eşiği |
| `POST /security/kill-switch/activate` | Kill switch'i manuel devreye alır |
| `POST /security/kill-switch/deactivate` | Kill switch'i kapatır |
| `POST /trading/leverage` | Gerçek kaldıracı değiştirir — `MAX_LEVERAGE` tavanına ve tüm canlı-işlem kapılarına tabidir |

**Operasyonel güvenlik (Bölüm 9.4) — bunlar kodla değil, sizin altyapı/hesap
ayarlarınızla sağlanır, bu proje kapsamının dışındadır:** Binance hesabında
2FA, VPS'e SSH-key ile giriş, fail2ban, düzenli veritabanı yedeği.

## Yol haritası

- [x] Screener (Binance, teknik skor)
- [x] ML sinyal modülü (MLP tabanlı yön tahmini)
- [x] Otomatik açma/kapama karar motoru (paper-trading)
- [x] DCA optimizasyon hesaplayıcısı
- [x] JSON tabanlı strateji tanımlama motoru (TradingView'sız)
- [x] Portföy / risk yönetimi kuralları (pozisyon boyutlandırma, maruziyet limitleri, günlük zarar devre kesici) + karar motoruna entegrasyon
- [x] Binance canlı işlem hazırlığı (güvenlik kapılı) + Denizbank Açık Bankacılık şablonu
- [x] Güçlü backtest motoru (otomatik veri yeterliliği keşfi + train/test + Sharpe/Sortino/Calmar)
- [x] Kelly kriteri (çeyrek/yarım/tam) pozisyon boyutlandırma + canlı işlem geçmişinden otomatik entegrasyon
- [x] Screener + motorları periyodik/zamanlanmış bir job'a bağlama (APScheduler, FastAPI lifespan)
- [x] ML metodolojisi yükseltmesi: triple-barrier etiketleme + olasılık kalibrasyonu + meta-labeling ("Kripto Bot Tam Rehber" entegrasyonu)
- [x] Kalıcı veritabanı katmanı (TimescaleDB/PostgreSQL, opsiyonel) + Docker Compose
- [x] Güvenlik protokolü sertleştirme: kill switch (manuel+otomatik), sabit kaldıraç tavanı, API anahtarı çekim izni kontrolü, sır tarama betiği
- [ ] Redis canlı cache katmanı (çoklu-process ölçeklenme gerektiğinde)
- [ ] BIST/VIOP adapter'ları
