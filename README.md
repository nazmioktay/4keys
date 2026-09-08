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

**Faz A — XGBoost (rehberin önerdiği ilk model)** ✅
`app/ml/model.py::SignalModel`'in varsayılan algoritması artık **XGBoost**
(gradient boosted karar ağaçları) — eski MLP sinir ağı `algorithm="mlp"`
ile karşılaştırma amaçlı hâlâ seçilebilir.

**Faz B — LSTM** ⚠️ Altyapı kuruldu, ama **canlı sonuçlar overfit** —
kullanıma alınmadı (rehberin "Faz A stabilleşmeden geçilmez" kuralı
kullanıcı kararıyla bilinçli olarak atlanarak, kod/altyapı hazırlığı
için erken kuruldu)

Production'da (`app.acromer.com`, 2026-09-03) BTC/USDT.P + top screener
sembolleriyle (20 sembol, 1160 pencere) yapılan ilk canlı eğitim:
`final_train_accuracy=%76.2` ama `out_of_sample_accuracy=%28.9`
(3 sınıflı rastgele tahminden -%33- bile kötü) — klasik overfitting.
Karar: **LSTM şimdilik rafta**, `feature_snapshots` tablosunda yeterli
geçmiş birikene kadar ve/veya daha fazla sembol+geçmiş ile tekrar
denenene kadar `/ml/predict-lstm` sonuçlarına güvenilmemeli. Odak
tekrar Faz A (XGBoost)'a döndü.
`app/ml/lstm_model.py::LSTMSignalModel` — çok katmanlı, dropout'lu bir
LSTM (Long Short-Term Memory) sinir ağı. XGBoost her barı BAĞIMSIZ bir
satır olarak görürken, LSTM son `seq_len` barın (bu bölüm yazıldığında
24 olan, şimdi 39'a çıkan) özellikli vektörünü
SIRAYLA okuyup önceki adımlardan öğrendiğini bir gizli duruma taşır —
rehberin "Güçlü olduğu alan: Sekans ve zaman örüntüleri" satırının
karşılığı.
- `app/ml/sequence_dataset.py::build_sequence_dataset` — kayan pencereli
  (sliding window) veri seti kurar; her pencere yalnızca KENDİ sembolünün
  kesintisiz kronolojik serisinden gelir, semboller arası sızıntı olmaz.
- Overfitting koruması: LSTM katmanları arası **dropout** + Adam'ın
  **L2 regularizasyonu** (`weight_decay`) + kronolojik son %20'lik
  **out-of-sample holdout** (fit() sırasında modele hiç gösterilmez) —
  rehber "2.4 Overfitting"in LSTM'e özgü önerileriyle uyumlu. Walk-forward
  CV, her fold için sıfırdan sinir ağı eğitmenin maliyeti nedeniyle burada
  uygulanmadı (XGBoost'tan farklı olarak).
- `POST /ml/train-lstm` — `seq_len`, `epochs` gibi parametrelerle eğitir,
  train loss/accuracy ve out-of-sample doğruluğunu döner.
- `GET /ml/predict-lstm?symbol=...` — en son `seq_len` bardan tahmin üretir.

**Lookback/otomatik eğitim optimizasyonu (2026-09):**
- `POST /ml/sweep-lookback` (`app.ml.train.sweep_lookback_values`) —
  belirtilen `lookback` değerlerinin HER biriyle sıfırdan eğitim yapıp
  walk-forward + out-of-sample metriklerini karşılaştırmalı döner.
  **Bilinçli tasarım kararı**: bu endpoint "en iyi" lookback'i OTOMATİK
  seçmez — hangi noktadan sonra ek geçmişin doğruluğu anlamlı şekilde
  artırmadığını (platoya ulaştığını) gözlemleyip karar vermek operatöre
  bırakılır, çünkü bu hem doğruluk hem hesaplama maliyeti arasında bir
  değer yargısıdır. `train_signal_model_validated(..., persist=False)`
  ile production modeli sweep sırasında ASLA değiştirilmez.
- `Settings.ml_auto_retrain_enabled` (varsayılan **False**) +
  `ml_auto_retrain_seconds` (varsayılan 24 saat) — `app.scheduler.jobs.job_auto_retrain`,
  açıksa screener'ın top long/short listesiyle XGBoost'u (ve daha önce
  eğitilmişse meta-label modelini) periyodik olarak otomatik yeniler.
  **Önemli dürüstlük notu**: 24 saatlik varsayılan, "10.000 mumluk
  (~1.14 yıl) veri setine göre bir günde biriken ~24 yeni barın toplamın
  ~%0.24'ü olduğu, bu yüzden daha sık yeniden eğitmenin maliyeti
  artırıp faydayı neredeyse hiç artırmayacağı" akıl yürütmesine
  dayanır — ama bu ortamdan (sandbox) canlı piyasa verisine erişilemediği
  için GERÇEK bir backtest ile ampirik olarak doğrulanmadı. Üretim
  sunucusunda `/ml/sweep-lookback` ve zaman içinde birikecek gerçek
  performans verisiyle bu değer daha isabetli kalibre edilebilir.
  Varsayılan kapalı: otomatik olarak production modelinin üzerine
  yazılması, kullanıcının bilinçli bir tercihi olmalı.

**Veri altyapısı genişletmesi (2026-09, LSTM'in overfit sonucuna tepki
olarak — "önce veri, sonra daha fazla özellik" sırası):**
- `app/exchanges/binance.py::BinanceExchange.fetch_ohlcv` artık **sayfalama
  (pagination)** yapabiliyor: `limit`, Binance'in tek istekteki üst sınırını
  (1000 mum) aşarsa otomatik olarak birden fazla istekle birleştiriyor —
  ücretsiz olarak aylar/yıllar süren geçmiş veri çekilebiliyor.
- `Settings.ml_train_timeframe` (varsayılan `1h`) ve `ml_train_lookback`
  (varsayılan `10000`) — screener'ın canlı görüntülediği `candle_timeframe`
  (`4h`) / `candle_lookback`'ten **bilinçli olarak ayrı** tutulur: screener
  4h'de kalırken, ML eğitimi artık daha ince taneli ve çok daha derin
  (~10.000 saat ≈ 416 gün / ~1.14 yıl, sembol başına ~10.000 mum) bir
  geçmişle çalışır — pagination (`BinanceExchange.fetch_ohlcv`) sayesinde
  tek istek sınırı (1000) aşılarak ücretsiz şekilde çekilir. **Karar
  motoru** (`app/engine/service.py::run_cycle_once`, otomatik açma/kapama
  döngüsü) de bu ayarları kullanır — modelin eğitildiği dağılımla (1h)
  AYNI zaman diliminden tahmin üretir; yalnızca screener'ın kendi teknik
  skor gösterimi 4h'de kalmaya devam eder. `DecisionEngine._predict` de
  artık `/ml/predict` ile aynı şekilde canlı makro/order-book özelliklerini
  (bkz. aşağı) tahmine ekliyor — önceden bunları hiç görmüyordu (model
  eğitimde gördüğü 13 kolonu canlıda sessizce 0.0/nötr varsayıyordu).
- `FEATURE_COLUMNS` 24'ten **39 teknik özelliğe** çıkarıldı: ham OHLC mum
  yapısı (`candle_body_pct`, `candle_upper_wick_pct`, `candle_lower_wick_pct`,
  `true_range_pct` — ham fiyat değil, ölçeklenmiş oranlar) ve kullanıcının
  istediği ek TradingView göstergeleri (`app/ml/advanced_indicators.py`):
  **Bollinger Bands** (`bb_percent_b`, `bb_bandwidth_norm`), **ATR**
  (`atr_pct`), **ADX** (`adx_norm`, `di_diff_norm`), **VWAP**
  (`vwap_gap_pct`), **OBV** (`obv_slope_norm`), **SuperTrend**
  (`supertrend_trend`, `supertrend_dist_pct`), **Ichimoku Bulutu**
  (`ichimoku_cloud_position`, `ichimoku_tk_cross`), **Fibonacci geri
  çekilme** (`fib_retracement_position`).
- `MACRO_FEATURE_COLUMNS` (11 ayrı kolon) — `app/ml/macro_features.py`,
  `macro_snapshots` tablosundaki geçmişi zaman bazlı **en-yakın-geçmiş
  eşleştirmeyle** (`pd.merge_asof`, `direction="backward"` — geleceğe
  bakma YOK) OHLCV barlarına ekler; her makro kolon kendi geçmişinin
  ortalama/std'siyle normalize edilir. Makro geçmişi kısa olduğu sürece
  (toplama yakın zamanda başladı) bu kolonlar çoğu eski bar için NaN
  kalır — bu satırlar EĞİTİMDEN ATILMAZ (XGBoost NaN'ı doğal olarak ele
  alabiliyor), yalnızca `/ml/predict` gibi canlı/tekil tahmin yollarında
  eksik makro değerler 0.0 (nötr) ile doldurulur
  (`app.ml.model._select_features`). Makro geçmişi biriktikçe bu
  özelliklerin gerçek ayırt ediciliği otomatik olarak artacak.
- `ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + MACRO_FEATURE_COLUMNS` (39+11=50)
  — XGBoost'un artık gerçekte gördüğü tam girdi seti budur.
- **`feature_snapshots` backfill** — ML eğitimi (`/ml/train`, `/ml/train-lstm`)
  zaten geniş bir geçmiş (`ml_train_lookback`) çektiği için, artık bu
  geçmişi eğittiği HER sembol için `feature_snapshots`'a da yazar
  (`app.db.repository.record_feature_snapshots_bulk`, `app.ml.dataset._persist_feature_snapshots`)
  — `feature_snapshot_symbols` ayarındaki kısıtlamadan bağımsızdır. Bu,
  LSTM/RL için gereken uzun/kesintisiz zaman serisinin aylarca sürecek
  periyodik birikim yerine **tek seferde** oluşmasını sağlar. Zaten
  kayıtlı barlar `ON CONFLICT DO NOTHING` ile atlanır — aynı geçmişi
  tekrar tekrar eğitmek güvenlidir.
- **Ham hacim büyüklüğü** — `volume_zscore` (`FEATURE_COLUMNS`, teknik):
  hacmin log-dönüşümlü, kendi rolling ortalama/std'sine göre z-skoru.
  `volume_ratio` (kısa vadeli MA'ya oran) farklı olarak, hacmin MUTLAK
  büyüklüğündeki anormallikleri (ani hacim patlaması) daha geniş bir
  pencerede yakalar.
- **Emir defteri (order book) derinliği** — `app/orderbook/` (yeni paket,
  `app.macro` ile aynı desen): `BinanceExchange.fetch_order_book_metrics`
  (bid/ask hacmi, imbalance, spread %) periyodik olarak (varsayılan 30
  dakikada bir, `feature_snapshot_symbols` sembolleri için)
  `orderbook_snapshots` tablosuna kaydedilir. **Önemli kısıt**: borsalar
  geçmişe dönük emir defteri saklamaz/satmaz — bu veri yalnızca
  toplamaya BAŞLADIĞIMIZ andan itibaren birikir, geçmiş 10.000 muma
  geriye dönük eklenemez. `ORDERBOOK_FEATURE_COLUMNS` (3 kolon:
  `orderbook_imbalance`, `orderbook_spread_norm`, `orderbook_depth_norm`),
  `app.ml.orderbook_features` ile (makro gibi) as-of merge edilir —
  sembol bazında. `ALL_FEATURE_COLUMNS` artık 39 teknik + 11 makro + 3
  order book = **53** kolon. Endpoint'ler: `GET/POST /orderbook/latest`,
  `/orderbook/history`, `/orderbook/refresh`.

**Faz C — Reinforcement Learning (opsiyonel)** ⚠️ Yalnızca HAZIRLIK
yapıldı (ortam + veri pipeline'ı), gerçek bir ajan (PPO/DQN) HENÜZ
EĞİTİLMEDİ — rehberin kendisi de RL'i zorunlu tutmuyor.
- `app/rl/environment.py::TradingEnv` — bağımlılıksız (gymnasium
  GEREKTİRMEYEN, ama aynı `reset()`/`step()` sözleşmesini izleyen) basit
  tek-pozisyonlu (flat/long/short) alım-satım ortamı. Gerçek bir ajan
  eğitilmek istendiğinde `stable-baselines3` + `gymnasium` eklenip bu
  sınıf ince bir sarmalayıcıya dönüştürülebilir.
- **Gözlem**: o barın 53 özelliği (`ALL_FEATURE_COLUMNS`) + [güncel
  pozisyon yönü, pozisyonun açık olduğu bar sayısı (normalize)].
- **Ödül**: 1-bar yürütme gecikmeli mark-to-market getiri EKSİ pozisyon
  değiştirildiğinde işlem maliyeti (`transaction_cost_pct`) — maliyetsiz
  bir ortamda ajanın gürültüyü "kâr" sanıp anlamsızca sık işlem yapmasını
  (RL'e özgü bir overfitting biçimi) engellemek için.
- **Ezberlemeyi önleme**: her eğitim epizodu, verinin KENDİSİ değil,
  İÇİNDEKİ RASTGELE bir pencerede başlar (bootstrap benzeri — ajan tek
  bir sabit sırayı ezberleyemez); kronolojik son `holdout_frac` (%20)
  eğitim epizotlarına HİÇBİR ZAMAN dahil edilmez (`reset(use_holdout=True)`
  yalnızca değerlendirme için, XGBoost/LSTM'deki aynı out-of-sample
  disiplini).
- `app/rl/dataset.py::build_episode_data` — bir sembol için kronolojik
  özellik matrisi + kapanış fiyatı dizisini hazırlar (aynı 53 özellik,
  makro/order-book eksikse 0.0/nötr ile doldurulur — RL ağı XGBoost'un
  aksine NaN kabul etmez).
- `app/rl/train.py::evaluate_random_policy` — henüz gerçek bir ajan
  olmadığı için, TAMAMEN RASTGELE bir politikanın "şans seviyesi"
  referansını ölçer (XGBoost'taki "%33 rastgele tahmin" referansının
  RL karşılığı) — hem ortamın doğru çalıştığını doğrular hem de gelecekte
  eğitilecek gerçek ajanın aşması gereken tabanı verir.
- `GET /rl/random-baseline?symbol=...&episodes=20` — yukarıdakini canlı
  veriyle çalıştırır.
- **Sırada**: `stable-baselines3` bağımlılığının eklenmesi (kullanıcı
  onayı gerekir — yeni, nispeten büyük bir bağımlılık), gerçek bir PPO
  ajanının eğitimi, ve ajanın XGBoost/rastgele referanslarla
  karşılaştırılması.

**Overfitting koruması** (rehber "2.4 Overfitting"):
- `app/ml/validation.py::walk_forward_splits` — **Walk-Forward Validation**:
  model bir pencerede eğitilir, hemen sonraki (görülmemiş) pencerede test
  edilir; pencere ileri kaydırılır.
- Aynı fonksiyonda **embargo/purge** — eğitim ile test penceresi arasına
  `embargo_frac` genişliğinde bir zaman boşluğu konur; etiketleme ufku
  (horizon) yüzünden test'e sızabilecek bilgi bu boşlukta atılır.
- `app/ml/validation.py::split_out_of_sample` — **Out-of-Sample Test**:
  kronolojik olarak en yeni dilim (varsayılan son %20) hem walk-forward
  CV'de hem de nihai `fit()`'te ASLA kullanılmaz, yalnızca son doğrulama
  metriği için ayrılır.
- `SignalModel.shap_values()` — **SHAP değerleri**: her özelliğin
  tahmine ortalama mutlak katkısını döner (`GET /ml/explain`); anlamsız
  özellikler bu sırlamadan görülüp elenebilir. Yalnızca XGBoost için
  çalışır (MLP bir kara kutudur).
- **Regularization**: XGBoost L1/L2 (`reg_alpha`/`reg_lambda`) ve
  subsample/colsample_bytree; MLP tarafında sklearn'ün L2 (alpha) +
  early-stopping.
- `POST /ml/train` yanıtı artık `walk_forward_mean_accuracy`,
  `overfit_gap` (eğitim doğruluğu - ortalama test doğruluğu; büyükse
  ezberleme işareti) ve `out_of_sample_accuracy`'yi de döner — bir
  modelin gerçekten mi öğrendiğini yoksa geçmişi mi ezberlediğini
  görünür kılar.

**Paper trading / gerçek para notu:** Bu doğrulama katmanı da dahil
sistemin tamamı hâlâ yalnızca paper-trading'dir; `enable_live_trading`
ve `enable_bist_trading` varsayılan olarak `false`'tur ve hiçbir ortamda
açılmamıştır (bkz. Modül: Güvenlik protokolü).

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
| `POST /ml/train` | Birincil modeli eğitir (`algorithm`: `"xgboost"`\|`"mlp"`, `labeling_method`: `"threshold"`\|`"triple_barrier"`, `calibrate`: bool, `calibration_method`: `"sigmoid"`\|`"isotonic"`, `holdout_frac`, `walk_forward_splits`) — yanıt walk-forward + out-of-sample metriklerini de içerir |
| `GET /ml/explain?symbol=...` | Eğitilmiş XGBoost modelinin SHAP özellik önemlerini döner |
| `POST /ml/train-meta` | Meta-label modelini eğitir (önce `/ml/train` çağrılmış olmalı) |
| `POST /ml/sweep-lookback` | Farklı `lookback` değerleriyle art arda eğitip walk-forward/out-of-sample metriklerini karşılaştırır (production modelini DEĞİŞTİRMEZ) — "en küçük yeterli lookback" kararı için veri sağlar |
| `GET /ml/predict?symbol=BTC/USDT:USDT` | Yön + kalibre güven tahmini (meta model varsa `meta_act`/`meta_confidence` de döner) |
| `POST /ml/train-lstm` | LSTM (Faz B) modelini sekans veri setiyle eğitir (`seq_len`, `epochs` vb.) |
| `GET /ml/predict-lstm?symbol=BTC/USDT:USDT` | LSTM ile yön + güven tahmini |
| `POST /ml/train-patchtst` | PatchTST'ten esinlenilmiş patch-tabanlı Transformer'ı sekans veri setiyle eğitir (LSTM'e alternatif) |
| `GET /ml/predict-patchtst?symbol=BTC/USDT:USDT` | PatchTST ile yön + güven tahmini |
| `POST /ml/train-regime` | Hibrit rejim+ML: GMM ile piyasayı rejimlere ayırıp her rejim için ayrı bir XGBoost modeli eğitir, karşılaştırma verisi döner |
| `POST /ml/train-online` | Çevrimiçi (online) öğrenme: `river` ARFClassifier'ı bar-bar (prequential) eğitip/değerlendirir |
| `GET /rl/hurst-execution-timing?symbol=BTC/USDT:USDT` | Hurst-tabanlı işlem zamanlaması hipotez testi (canlı strateji değil, tarihsel gözlem) |
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
| `GET /portfolio/status` | Equity, açık pozisyonlar (kademeli dilim durumu dahil), kapanan işlemler, aktif kurallar, işlem istatistikleri |
| `GET /portfolio/pnl` | Kayan pencereli (son 24s/7g/30g) + toplam PNL özeti |
| `PUT /portfolio/rules` | Risk kurallarını (fixed_risk/Kelly, kademeli alım-satım dilimleri dahil) günceller |
| `POST /portfolio/reset` | Portföyü verilen sermaye/kurallarla sıfırdan başlatır |
| `POST /portfolio/position-size` | Risk yüzdesi + SL mesafesinden pozisyon boyutu hesaplar (fixed_risk) |
| `GET /portfolio/trade-stats` | Portföyün kendi geçmişinden hesaplanan kazanma oranı / ort. kazanç-kayıp |
| `POST /portfolio/kelly-size` | Çeyrek/yarım/tam Kelly'ye göre bağımsız pozisyon boyutu hesaplar |
| `POST /portfolio/risk-check` | Durumsuz "ne olurdu" risk kontrolü (paylaşılan portföyü etkilemez) |

**Confidence-weighted boyutlandırma + VIX rejim filtresi (2026-09):**
- `RiskRules.confidence_scaling_enabled` (varsayılan **True**) — Kelly/fixed_risk'in
  önerdiği boyut, artık yalnızca geçmiş kazanma oranına değil, O ANKİ
  tahminin `confidence`'ına da duyarlı: `confidence_scaling_min_confidence`
  (varsayılan 0.6, genelde `open_confidence` eşiğiyle aynı tutulmalı) ve
  altında ölçek `confidence_scaling_min_scale`'e (varsayılan 0.5) sabitlenir;
  `1.0` confidence'ta ölçek her zaman `1.0`'dır, arası doğrusal enterpole edilir.
- `RiskRules.vix_regime_filter_enabled` (varsayılan **False**, opt-in) —
  açılırsa, VIX'in kendi geçmişine göre z-skoru (`macro_vix_norm`,
  `app.ml.macro_features`) `vix_zscore_reduce_threshold`'u (varsayılan 1.5)
  aşarsa boyut yarıya iner, `vix_zscore_block_threshold`'u (varsayılan 2.5)
  aşarsa yeni pozisyon TAMAMEN engellenir — "modelin normal piyasa
  koşullarında öğrendiği örüntüler kriz anlarında güvenilmez olabilir"
  varsayımıyla. `Action.reason`'da (`blocked` durumunda) neden açıkça
  belirtilir.
- İkisi de `DecisionEngine._open` → `PortfolioManager.propose_open`
  üzerinden otomatik uygulanır; `PUT /portfolio/rules` ile parametrik.

**Kademeli (aşamalı) alım/satım (2026-09):**
- `RiskRules.entry_tranche_weights` (varsayılan `[0.5, 0.5]`) / `exit_tranche_weights`
  (varsayılan `[0.5, 0.5]`) — parametrik, `PUT /portfolio/rules` ile
  değiştirilebilir (frontend: **Paper Trading** ekranı).
- **Kademeli alım**: `PortfolioManager.open()` hesaplanan TAM (Kelly/fixed_risk)
  boyutun yalnızca ilk dilimini (`entry_tranche_weights[0]`) hemen açar.
  `DecisionEngine`, sinyal AYNI yönde ve yeterince güvenli kalmaya devam
  ederse (yani bir sonraki karar döngüsünde de teyit edilirse) sonraki
  dilim(ler)i `add_entry_tranche` ile ekler — ortalama giriş fiyatı
  ağırlıklı olarak yeniden hesaplanır. Bu, piyasayı tek büyük emirle
  hareket ettirmemek VE sinyalin geçici bir gürültü olmadığını teyit
  etmek içindir.
- **Kademeli satış**: kapanış sinyali geldiğinde `close_tranche` yalnızca
  ilk dilimi satar (Action tipi `close_partial`); sinyal sonraki
  döngü(ler)de de sürerse kalan dilim(ler) kapanır (Action tipi `close`).
  Son dilim, yuvarlama artığı kalmaması için pozisyonun TAMAMINI kapatır.
  `PortfolioManager.close()` (tek seferde tam kapatma) geriye dönük
  uyumluluk ve acil durumlar için hâlâ mevcuttur.
- `GET /portfolio/pnl`: toplam + son 24 saat/7 gün/30 gün (takvim
  sınırı değil, kayan pencere) PNL ve işlem sayısı/kazanma oranı.
- Frontend: `frontend/src/pages/PaperTrading.jsx` — PNL kartları, açık
  pozisyonların dilim durumu, kademeli dilimler dahil kapanan işlem
  geçmişi, Kelly çeşidi + dilim ağırlıklarını düzenleyen parametrik form.

### Modül — Prometheus + Grafana İzlenebilirliği ✅ (2026-09)
- `starlette-exporter` (`app/main.py`) — HTTP istek sayısı/gecikmesi/hata
  oranı otomatik toplanır, `GET /metrics`'te Prometheus formatında sunulur.
  `group_paths=True`: `/ml/predict?symbol=X` gibi query param'lı istekler
  tek path etiketi altında toplanır (kardinalite patlamasını önler).
- `app/monitoring/metrics.py` — iş mantığına özgü metrikler: portföy
  equity/açık pozisyon/oturum PNL'i (Gauge), kapanan işlemler (Counter,
  kazanç/kayıp/nötr etiketli), ML tahmin güveni (sembol+yön etiketli),
  zamanlayıcı işlerinin son çalışma durumu — hepsi aynı registry'de,
  ekstra kablolama gerekmez.
- **Docker Compose** (`docker-compose.yml`, yerel geliştirme): Prometheus
  (`:9090`) + Grafana (`:3001`, anonim Viewer erişimi açık — yerel geliştirme
  kolaylığı için, production'da KAPALI, bkz. `deploy/add-monitoring.sh`).
- **Production** (`deploy/add-monitoring.sh`): `4keys-net` ağına
  `4keys-prometheus` + `4keys-grafana` container'larını ekler; Grafana
  admin şifresi varsayılan `admin` — **ilk girişte değiştirilmeli**.
  Prometheus yalnızca `127.0.0.1`'e bağlanır (dışa kapalı); Grafana `:3001`
  dışa açık — HTTPS/Nginx reverse-proxy henüz eklenmedi (ileride yapılacak).
- **Hazır dashboard**: `monitoring/grafana/dashboards/4keys-overview.json`
  — istek oranı, hata oranı, p95 gecikme, işlenmekte olan istek sayısı
  (topluluğun `starlette-exporter` için bilinen "FastAPI Observability"
  panosundaki standart panellerin karşılığı) + 4keys'e özgü paneller
  (portföy equity, açık pozisyon, kapanan işlemler, ML güven skoru,
  zamanlayıcı iş durumu). Grafana açılışta otomatik provizyon edilir
  (`monitoring/grafana/provisioning*/`), elle dashboard import gerekmez.

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
- **Monte Carlo bootstrap** (`app/backtest/metrics.py::monte_carlo_bootstrap`,
  2026-09) — `vectorbt` gibi ağır bağımlılıklar (numba vb.) eklemeden,
  mevcut motorun ÜSTÜNE eklenen hafif bir katman: test (out-of-sample)
  dönemindeki GERÇEKLEŞEN işlem getirileri YERİNE KOYARAK (with
  replacement) `monte_carlo_simulations` (varsayılan 1000) kez yeniden
  örneklenir; her simülasyon için toplam getiri ve max drawdown hesaplanıp
  p5/p50/p95 yüzdelikleri + kayıpla sonuçlanma olasılığı raporlanır. Bu,
  işlem SIRASININ/ÖRNEKLEMESİNİN "şans" payını ölçer — az sayıda işlemle
  (<10) güvenilmez olacağından o durumda `null` döner. `monte_carlo_simulations: 0`
  ile tamamen atlanabilir.

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
  rate/drawdown gibi tüm performans ölçümünün temeli), `feature_snapshots`
  (aşağıda ayrıca açıklanıyor).
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
| `GET /db/features?symbol=BTC/USDT:USDT&limit=5000` | Biriken ML özellik vektörleri (aşağıya bkz.) |

**Feature snapshot biriktirme (LSTM/RL için zaman serisi veri seti hazırlığı):**
`app/db/models.py::FeatureSnapshot` — her tarama döngüsünde, `FOURKEYS_FEATURE_SNAPSHOT_SYMBOLS`
ile belirlenen sembollerin (varsayılan: `BTC/USDT:USDT`) 39 teknik ML
özelliği (bkz. `app.ml.features.FEATURE_COLUMNS`) zaman damgasıyla kaydedilir
(`app/screener/scanner.py` içine kancalanmıştır). XGBoost şu an hâlâ her
eğitimde Binance'ten anlık ham veri çekiyor — bu tablo onu DEĞİŞTİRMİYOR,
ayrı ve bağımsız bir birikim. Amaç: zamanla burada gerçek, kesintisiz bir
piyasa zaman serisi oluşsun; LSTM (sekans modeli) ve Reinforcement Learning
ajanı ileride Binance'ten sınırlı bir geçmişle değil, burada biriken uzun
gerçek veriyle eğitilebilsin. `app/db/repository.py::get_feature_snapshots`
bu veriyi kronolojik DataFrame olarak okur — hem bu ileriye dönük kullanım
hem de istenirse XGBoost eğitimini de canlı Binance çekişinden bu tabloya
geçirmek için hazır.

**Kullanıcının manuel TradingView göstergeleri (`app/ml/advanced_indicators.py`):**
Orijinal 9 özelliğin üzerine, kullanıcının kendi manuel işlemde kullandığı
göstergelerin Python karşılıkları eklendi — Heikin Ashi, Stochastic RSI
(log-getiri üzerinden), MavilimW, PMax, Doğrusal Regresyon Kanalı,
WaveTrend (LazyBear), Nadaraya-Watson Envelope ve LonesomeTheBlue'nun
pivot-kümeleme tabanlı Dinamik Destek/Direnç göstergesi — toplam 24
özellik. Hepsi **causal** (yalnızca geçmiş veriye bakar, "repaint" etmez);
Nadaraya-Watson ve Dynamic S/R için bu özellikle test edilmiştir
(`tests/test_advanced_indicators.py::test_*_is_causal`) çünkü LuxAlgo'nun
varsayılan Nadaraya-Watson scripti gibi popüler versiyonlar geleceğe
bakarak repaint eder — canlı işlemde güvenilmez sonuç verir, burada
kullanılmadı. `feature_snapshots` tablosuna yeni kolonlar eklendiğinde
(`app/db/session.py::_add_missing_columns`) var olan tablo/satırlar
bozulmadan otomatik tamamlanır — şema göçü gerekmez.

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

### Modül 11 — BIST/VIOP Entegrasyonu (Denizbank AlgoLab) ✅
BIST/VIOP'un Binance gibi tek, halka açık bir retail API'si yok; Türkiye'de
algoritmik erişim bir aracı kurum API'si üzerinden olur. Zaten Denizbank
bağlantımız olduğu için **Denizbank AlgoLab** (retail algoritmik trading
API'si, BIST hisse + VIOP vadeli) hedeflendi.

**Dürüstlük notu (Denizbank Açık Bankacılık entegrasyonuyla aynı prensip):**
AlgoLab'ın sabit, halka açık bir OpenAPI şeması yok. `app/exchanges/algolab.py`,
AlgoLab'ın yaygın bilinen genel kimlik doğrulama akışını (API key + kullanıcı
adı/şifre → SMS/e-posta doğrulama kodu → oturum hash'i) ve tipik uç nokta
kalıbını doğru mimariyle uyguluyor; kesin uç nokta yolları/yanıt alan adları
API key başvurunuz sonrası erişeceğiniz güncel dokümantasyonla teyit
edilmeli (`_endpoints` sözlüğü tek noktadan güncellenecek şekilde tasarlandı).

- `AlgoLabExchange`, screener/ML/backtest modüllerinin kullandığı aynı
  `Exchange` arayüzünü uyguluyor — mimari olarak Binance ile birebir aynı
  soyutlamayı paylaşıyor.
- **İki adımlı oturum**: `POST /bist/login` (kullanıcı adı/şifre → SMS/e-posta
  kodu tetiklenir) → `POST /bist/login/verify` (kodu doğrulayıp oturum
  hash'ini alır). Binance'ten farklı olarak AlgoLab'da **piyasa verisi bile
  oturum gerektirir**.
- **Aynı güvenlik prensipleri**: gerçek emir göndermek kill switch'in kapalı
  olmasını, `FOURKEYS_ENABLE_BIST_TRADING=true`'yu VE `confirm: true`'yu
  gerektirir — Binance'teki üç kapılı sistemin birebir aynısı.

| Endpoint | Açıklama |
|---|---|
| `POST /bist/login` | 1. adım: kullanıcı adı/şifre, SMS/e-posta kodu tetikler |
| `POST /bist/login/verify` | 2. adım: doğrulama kodunu girip oturumu tamamlar |
| `GET /bist/symbols?market_type=equity\|viop` | Sembol listesi |
| `GET /bist/ohlcv?symbol=&timeframe=&limit=` | Geçmiş mum verisi |
| `GET /bist/positions` | Açık pozisyonlar |
| `POST /bist/order` | Gerçek emir gönderir — 3 güvenlik kapısı aktif |

### Modül 12 — Frontend (React) ✅
`frontend/` altında, backend'e gerçekten bağlanan basit bir React paneli.
Görsel olarak ilk paylaşılan 3Commas ekran görüntülerindeki koyu tema ve alt
sekme yapısını (Portföy / Al-Sat / AI Asistan / Araştırıcı / Ayarlar) izler,
ama her sekme sahiden var olan bir backend uç noktasına bağlıdır — hiçbir
sahte/işlevsiz buton yok. **AI Asistan** sekmesi bilinçli olarak "henüz
eklenmedi" diyor, çünkü arkasında gerçek bir özellik yok.

| Sekme | Bağlı olduğu modül |
|---|---|
| Portföy | `/portfolio/status`, `/security/status` — equity, açık pozisyonlar, kill switch durumu |
| Al-Sat | `/dca/optimize`, `/strategy/examples`+`/strategy/backtest`, `/engine/run-cycle` |
| Araştırıcı | `/screener/top` — Top 10 Long/Short |
| Ayarlar | `/portfolio/rules` (düzenlenebilir), `/security/*` (kill switch aç/kapa), `/scheduler/status`, `/db/status` |

**Çalıştırma:**
```bash
cd frontend
npm install
npm run dev
```
Backend'in `localhost:8000`'de çalıştığını varsayar; `docker compose up`
ile de (backend + frontend + TimescaleDB) birlikte ayağa kalkar.

**Bu modülü kurarken gerçek bir backend hatası bulundu ve düzeltildi:**
FastAPI/Starlette'te bare `Exception` için kayıtlı bir `exception_handler`,
`CORSMiddleware`'in DIŞINDA çalışan `ServerErrorMiddleware`'e ekleniyor —
bu yüzden yakalanmamış hatalar CORS başlıklarını hiç almıyor ve tarayıcıda
gerçek hata mesajı yerine anlamsız bir "Failed to fetch" görünüyordu. Çözüm:
`app/main.py::UnhandledExceptionMiddleware` — bir exception handler değil,
gerçek bir middleware, CORSMiddleware'den SONRA eklenerek onun İÇİNDE
çalışacak şekilde. Bu, hem gerçek sunucuya karşı curl ile hem de
`tests/test_error_handling.py` ile doğrulandı.

### Modül 13 — Demo/sentetik veri modu ✅
Gerçek borsaya ağ erişimi olmayan ortamlarda (ör. bu geliştirme
sandbox'ı, kısıtlı ağ politikası olan CI, offline geliştirme) tüm
boru hattını (screener → ML eğitimi → karar motoru) uçtan uca canlı
göstermek için `app/exchanges/demo.py::DemoExchange` eklendi. Sembol
başına sabit bir seed ile deterministik, GBM benzeri sentetik OHLCV
üretir; `Exchange` arayüzünü implemente eder, bu yüzden screener/ML/
strateji modülleri onu Binance'ten ayırt etmeden kullanır.

**Nasıl açılır:**
```bash
FOURKEYS_EXCHANGE_ID=demo uvicorn app.main:app --reload
```

**Güvenlik notu:** Demo modu yalnızca salt-okunur piyasa verisi
arayüzüne (`app.exchanges.get_exchange`) bağlıdır. Gerçek emir verme
yolu (`app.trading.executor.get_trading_exchange`) `exchange_id`
ayarını hiç okumaz ve her zaman gerçek `BinanceExchange`'e sabitlenmiştir
— yani demo modu asla gerçek bir emrin gönderilmesine yol açamaz. Bu,
`tests/test_demo_exchange.py::test_trading_executor_never_uses_demo_exchange`
ile doğrulanır.

### Modül 14 — Ücretsiz Makro/Piyasa Bağlamı Verileri ✅
`app/macro/` — kripto fiyatı yalnızca kendi grafiğinde hareket etmiyor;
daha geniş piyasa bağlamını (TOTAL, BTC dominansı, funding rate, VIX,
altın, dünya borsa endeksleri, Fed/ECB faiz oranları) periyodik olarak
toplayıp `macro_snapshots` tablosuna kaydeder — LSTM/RL eğitiminde OHLCV
tabanlı özelliklerin yanına ek bağlam olarak kullanılabilir.

- `app/macro/data.py` — her kaynak izole bir fonksiyon, **asla exception
  fırlatmaz** (bir kaynak geçici erişilemez olursa yalnızca `None` döner,
  diğerlerini etkilemez):
  - **CoinGecko** (`/global`, key gerekmez): TOTAL piyasa değeri, BTC dominansı
  - **Binance** (kimlik doğrulamasız, `BinanceExchange.fetch_funding_rate`): BTC perpetual funding rate
  - **Yahoo Finance** (`yfinance`, key gerekmez): VIX, altın (`GC=F`), S&P 500, Nasdaq, Nikkei, DAX
  - **ECB İstatistik Veri Ambarı (SDW)** (key gerekmez): mevduat faizi
  - **FRED** (ABD Merkez Bankası, ücretsiz ama key gerekir — `FOURKEYS_FRED_API_KEY`): efektif federal fon oranı; key boşsa yalnızca bu kaynak atlanır
- `app/scheduler/jobs.py::job_refresh_macro` — `FOURKEYS_MACRO_REFRESH_SECONDS`
  (varsayılan 6 saat) periyoduyla otomatik çalışır; makro veriler günlük/
  saatlik değiştiği için screener/motor kadar sık yenilenmesine gerek yok.
- `app/db/models.py::MacroSnapshot` — TimescaleDB/PostgreSQL açıkken kalıcı.

| Endpoint | Açıklama |
|---|---|
| `GET /macro/latest` | En son kaydedilen makro anlık görüntü |
| `GET /macro/history?limit=500` | Zaman içindeki makro birikimi |
| `POST /macro/refresh` | Tüm kaynakları şimdi çeker ve kaydeder (zamanlayıcıyı beklemeden) |

**Faz 2'ye ertelenenler (ücretsiz/güvenilir bir kaynağı olmadığı için):**
BTC likidasyon heatmap'i (Coinglass gibi kaynaklar çoğunlukla ücretli API
veya kırılgan scraping gerektiriyor) ve BTC ETF akışları (Farside/SoSoValue'nun
resmi ücretsiz API'si yok) — bunlar için ücretli bir API'ye abone olmak ya
da scraping riskini kabul etmek gerekecek; kullanıcıyla ayrıca karar verilecek.

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
- [x] XGBoost (Faz A) — birincil model + walk-forward/purged CV + out-of-sample holdout + SHAP açıklanabilirlik
- [~] LSTM (Faz B) — altyapı kuruldu (dropout + L2 + erken durdurma + gradyan kırpma + sınıf ağırlıklandırma ile). Sınıf ağırlıklandırma sonrası BTC-only sınamada balanced_accuracy rastgele seviyeden (%33) %38.7'ye çıktı ve ezberleme (train/out-of-sample farkı) pratik olarak ortadan kalktı — ama ne lookback artırma (10K→20K) ne de model kapasitesini küçültme (hidden_size=32,num_layers=1) bu ~%38-39 tavanını aşabildi; sınırlayıcı faktörün veri miktarı/model boyutu değil, mimari/özellik seti olabileceğine işaret ediyor. Kullanıma alınmadı, rafta
- [~] PatchTST (`app/ml/patchtst_model.py`) — LSTM'in taktığı bu tavanı aşıp aşamayacağını test etmek için eklenen, patch-tabanlı Transformer sınıflandırıcı (PatchTST'ten ESİNLENİLMİŞ, BASİTLEŞTİRİLMİŞ bir uygulama — kanal-bağımsız değil, kanal-karışık). LSTM ile AYNI eğitim disiplinini (holdout + erken durdurma + gradyan kırpma + sınıf ağırlıklandırma, `app.ml.train._train_sequence_model` üzerinden ortaklaştırıldı) paylaşır, adil karşılaştırma için. Henüz sunucuda test edilmedi (`deploy/train-patchtst-btc.sh`)
- [x] LSTM/PatchTST'in makro/order-book özelliklerini (13 kolon) hiç görmediği tutarsızlık düzeltildi — önceden `build_sequence_dataset` yalnızca 39 teknik özelliği kullanıyordu; artık XGBoost eğitim yolu (`app.ml.dataset`) ve canlı karar motoruyla (`app.engine.decision`) AYNI `ALL_FEATURE_COLUMNS` (53) kullanılıyor, macro/order-book geçmişi olmayan barlarda `fillna(0.0)` ile nötrleniyor (satır elenmiyor)
- [x] `/ml/train-lstm` ve `/ml/train-patchtst` artık isteğe bağlı bir `feature_columns` alt kümesi kabul ediyor (varsayılan `ALL_FEATURE_COLUMNS`) — `/ml/explain`'in SHAP önem sıralamasından seçilen en değerli N özellikle küçük veri setlerinde boyut/örnek oranını iyileştirme denemeleri için
- [ ] Reinforcement Learning (Faz C, opsiyonel)
- [x] Kalıcı veritabanı katmanı (TimescaleDB/PostgreSQL, opsiyonel) + Docker Compose
- [x] Güvenlik protokolü sertleştirme: kill switch (manuel+otomatik), sabit kaldıraç tavanı, API anahtarı çekim izni kontrolü, sır tarama betiği
- [ ] Redis canlı cache katmanı (çoklu-process ölçeklenme gerektiğinde)
- [~] Ağır eğitim işlerinin (LSTM/XGBoost) ayrı bir process'te (subprocess/worker) çalıştırılması — art arda birkaç ağır eğitim isteği, Python/PyTorch bellek ayırıcısının belleği işletim sistemine tam geri vermemesi nedeniyle kümülatif bir bellek artışına ve OOM'a yol açabiliyor (bkz. BTC-only lookback sweep testinde 10K/20K başarılı, 30K'de tekrar OOM; ve aşağıdaki "OOM üretim olayı" maddesi — bu risk gerçekten materyalize oldu). **`train-all.sh` için ÇÖZÜLDÜ VE SUNUCUDA DOĞRULANDI** (bkz. aşağıdaki madde: `app.cli` + `docker run --rm` + `--oom-score-adj` ile canlı API process'inden izole edildi, gerçek bir OOM'da canlı API dokunulmamış kaldı); diğer tekil eğitim script'leri (`train-lstm-btc.sh` vb.) ve doğrudan `/ml/train-all` HTTP çağrıları HÂLÂ canlı process içinde çalışıyor — kanıtlanmış risk (train-all'ın beş adımı zincirlemesi) öncelikliydi, kapsam bilerek dar tutuldu
- [x] BIST/VIOP adapter'ı (Denizbank AlgoLab — oturum tabanlı, aynı Exchange arayüzü, aynı güvenlik kapıları)
- [x] Frontend (React) — Portföy/Al-Sat/Araştırıcı/Ayarlar, gerçek backend'e bağlı
- [x] Demo/sentetik veri modu (`FOURKEYS_EXCHANGE_ID=demo`) — ağ erişimi olmadan uçtan uca canlı gösterim
- [x] Kullanıcının manuel işlemde kullandığı 8 gösterge ML özelliklerine eklendi (Heikin Ashi, Stoch RSI log, MavilimW, PMax, Regresyon Kanalı, WaveTrend, Nadaraya-Watson, Dynamic S/R) — 9 özellik → 24
- [x] Ücretsiz makro/piyasa bağlamı verileri (TOTAL, BTC dominansı, funding rate, VIX, altın, dünya endeksleri, Fed/ECB faiz oranları)
- [ ] BTC likidasyon heatmap'i + BTC ETF akışları — ücretsiz/güvenilir kaynak yok, Faz 2'ye ertelendi
- [ ] Çoklu zaman dilimi mimarisi (4h karar / 1D yön / 1h destek)
- [~] RL hazırlığı (ortam + veri pipeline'ı + rastgele referans) — gerçek ajan (PPO/DQN) henüz eğitilmedi
- [x] RL yönü netleştirildi: doğrudan alım-satım sinyali üretmek yerine (ödül tasarımının zorluğu nedeniyle riskli) daha odaklı kullanımlar değerlendirildi (optimal execution, model-tabanlı RL, hiyerarşik RL). Optimal execution (emri parçalara bölerek piyasa etkisini azaltma) seçildi ama BİLİNÇLİ OLARAK klasik haliyle uygulanmadı: order-book derinliği verimiz yok ve işlem boyutlarımız (çeyrek Kelly) BTC/USDT likiditesine göre ihmal edilebilir — piyasa etkisi modeli olmadan "bölmek daha iyi" sonucu yapay/yanıltıcı olurdu
- [x] Bunun yerine `app/rl/execution_timing.py`: Hurst üsteline dayalı, ölçülebilir bir hipotez test ediliyor — sinyal anında H<0.5 (ortalamaya-dönüş) iken sabit bir gecikmeyle (look-ahead yanlılığından kaçınmak için veriye göre optimize edilmez) yürütmek, H>0.5 (trend-devamlılığı) durumuna göre ortalama olarak daha ucuz mu? `GET /rl/hurst-execution-timing` — canlı bir ajan/strateji DEĞİL, tarihsel bir gözlem/hipotez testi
- [ ] Model-tabanlı RL (piyasa simülatörü öğrenip onda eğitme) ve hiyerarşik RL (portföy dağılımı + varlık-bazlı zamanlama) — kullanıcı önerdi, kapsamları (ayrı bir simülatör bileşeni; çoklu-varlık portföy optimizasyonu) şu anki tek-sembol (BTC-only) odağın ötesinde, backlog'da
- [x] Çevrimiçi (online) öğrenme / kavram kayması (concept drift) yönetimi (kullanıcı önerisi): `app/ml/online_model.py` — `river` kütüphanesi (yeni, hafif bağımlılık, kullanıcı onayıyla eklendi) ile `ARFClassifier` (Adaptive Random Forest: Hoeffding ağaçlarından oluşan, HER AĞACIN kendi ADWIN kavram kayması tespitine sahip olduğu bir topluluk) — kullanıcının hem "Hoeffding Ağaçları" hem "Online Random Forest" önerilerini birden karşılıyor. XGBoost'un toptan `fit(X,y)`'inin AKSİNE `learn_one`/`predict_one` ile bar-bar öğrenir; değerlendirme "test-then-train" (prequential) protokolüyle yapılır (`run_prequential_evaluation`) — online öğrenmede standart, look-ahead'siz bir yöntem, XGBoost'un statik holdout'undan FARKLI. `POST /ml/train-online` — sonuçlar pencere pencere (varsayılan 500 bar) raporlanır, modelin zaman içinde adapte olup olmadığını görmek için.
- [x] **Online model sunucuda doğrulandı — sonuç iyi**: BTC-öncelikli veride (19.686 satır) `overall_balanced_accuracy=%49.7` — XGBoost'un (varsayılan etiketlemeyle) takıldığı %33'ten ve LSTM'in ulaştığı %38-44'ten daha iyi. Pencere pencere bakıldığında net bir adaptasyon deseni var: ilk ~15 pencere (soğuk başlangıç, az veri) balanced_accuracy ~%33'te sabit, ~19. pencereden itibaren %38-55 aralığına sıçrayıp istikrarlı kalıyor (son pencerelerin ortalaması ~%43-45) — ADWIN kavram kayması tespitinin işe yaradığının somut kanıtı. `ensemble_online_enabled` (varsayılan `False`, opt-in) ile `DecisionEngine`'e üçüncü bir "oy" olarak eklendi — `_combine_predictions` AYNI kuralla (önce XGBoost+LSTM birleştirilir, sonra online modelle) uygulanır
- [x] Hibrit rejim+ML (kullanıcı önerisi: Markov Regime-Switching): `app/ml/regime.py` — tam bir Markov-Switching modeli (`statsmodels`) YENİ bir bağımlılık gerektirdiğinden, kullanıcının onayıyla bunun yerine scikit-learn'ün zaten kurulu `GaussianMixture`'ı kullanılıyor. Piyasa volatilite+trend uzayında `n_regimes` kümeye ayrılır (0=en düşük volatilite, yorumlanabilirlik için sıralı), rejimler sembole değil piyasaya özgü olduğundan semboller arası havuzlanmış (pooled) tek bir paylaşılan model eğitilir (kayan pencereli özellikler semboller BİRLEŞTİRİLMEDEN önce hesaplanır, sızıntı olmaz). `POST /ml/train-regime`: her rejim için AYRI bir XGBoost modeli eğitip out-of-sample sonuçlarını karşılaştırmayı sağlar — canlı karar motoruna HENÜZ bağlanmadı, önce "rejime ayırmak tek global modelden daha mı iyi?" sorusuna offline veri sağlamak amaçlanıyor
- [x] **İlk rejim denemesi (BTC-only) başarısız oldu, kök nedeni bulunup düzeltildi**: her üç rejimde de out_of_sample_balanced_accuracy tam olarak %33.3'e (rastgele seviye) oturdu — LSTM'de daha önce çözülen AYNI çoğunluk-sınıf çöküşü, burada rejimlere bölmenin zaten sınırlı tek-sembol veri setini küçük alt kümelere (1175-5023 satır) ayırmasıyla tetiklendi. İki düzeltme yapıldı: (1) `SignalModel`in XGBoost katmanına (`_XGBClassifierWrapper.fit`) LSTM'dekiyle AYNI ters-frekans sınıf ağırlıklandırması eklendi (artık `/ml/train`/`/ml/train-regime`/tüm XGBoost eğitim yolları için varsayılan), (2) rejim eğitimi artık BTC-only yerine çoklu-sembol (BTC-öncelikli + uyumlu semboller, `deploy/train-regime-multi.sh`) kullanıyor — henüz sunucuda yeniden doğrulanmadı
- [x] Prometheus + Grafana izlenebilirlik (hazır dashboard, otomatik provizyon)
- [x] Backtest Monte Carlo bootstrap (işlem sırası/örneklemesinin "şans" payını ölçer)
- [x] Confidence-weighted pozisyon boyutlandırma (tahminin güvenine göre ek ölçekleme)
- [x] VIX rejim filtresi (opsiyonel — aşırı piyasa stresinde boyut küçültme/engelleme)
- [ ] Sembol-çapraz doğrulama (bir grup sembolde eğitip hiç görmediği sembollerde test etme) — henüz yapılmadı
- [x] BTC-öncelikli eğitim sembol seçimi (`app/ml/symbol_selection.py`) — eskiden eğitim evreni doğrudan screener'ın Top-N Long+Short çıktısıydı (yalnızca kısa vadeli teknik skora göre, likidite/uyumluluk kontrolü yoktu — "zayıf seçilmiş bir grup"). Artık BTC/USDT (`ml_primary_symbol`) her zaman ilk sırada eğitiliyor; diğer semboller yalnızca (1) minimum likidite (`ml_min_quote_volume_24h`) ve (2) BTC ile getiri korelasyonu (`ml_min_correlation_with_primary`, varsayılan 0.4) eşiğini geçerlerse ek olarak katılıyor. Toplam sembol sayısı `ml_train_max_symbols` (varsayılan 5) ile sınırlandırılıp eğitim süresi kısaltıldı (önceden 20 sembole kadar çıkabiliyordu)
- [ ] Meta-labeling/karar eşiklerinin (`open_confidence`/`close_confidence`) backtestle optimizasyonu — ML karar motoruna özel bir backtest harness'i gerektiriyor, henüz yok (`/backtest` modülü şu an yalnızca DCA/JSON-strateji motorlarını destekliyor)
- [ ] `/ml/sweep-lookback` sonuçlarının periyodik/otomatik izlenmesi — henüz yok, elle çalıştırılıyor; Grafana panel/alert olarak da karşılık bulabilir (bkz. aşağıdaki Grafana maddeleri)
- [~] Ensemble (XGBoost + LSTM) — LSTM'in BTC-only sınamalarda (doğru etiketleme ile, bkz. yukarıdaki not) rastgele seviyenin belirgin üzerine çıktığı doğrulandıktan sonra eklendi. `DecisionEngine._combine_predictions`: basit, kural tabanlı bir birleştirme — iki model AYNI yönde mutabıksa güven artırılır, biri nötr diğeri yönlüyse indirimli güvenle kullanılır, ZIT yönlerdeyse (biri long biri short) belirsizlik nedeniyle nötre düşülür. Ağırlıklı oy birliği veya RL tabanlı bir meta-ensemble DEĞİL — RL henüz eğitilmediği için o kısım hâlâ eksik. Varsayılan KAPALI (`ensemble_lstm_enabled=False`, opt-in) — LSTM kalitesi her sembol/zaman diliminde ayrı ayrı doğrulanmadı
- [x] Frontend'den Grafana'ya "Monitoring" linki (yeni sekmede açılır) — Grafana bilinçli olarak ayrı bir izleme aracı olarak bırakıldı, uygulama içine gömülmedi
- [ ] Grafana alerting — eşik tabanlı uyarı kuralları (equity düşüşü, kill switch tetiklenmesi, art arda başarısız zamanlayıcı işi) + Slack/Telegram/e-posta bildirimi
- [ ] RL/LSTM'e özel Grafana panelleri — her retrain'de out-of-sample accuracy/overfit_gap'in kalıcı bir metrik olarak kaydedilip zaman içindeki değişiminin grafiğe dökülmesi (şu an yalnızca o anki API yanıtında görünüyor)
- [x] Mikro yapı: mum-bazlı agresif alım/satım akışı (`taker_buy_ratio_norm`, bkz. `app.ml.orderflow_features`) — Binance'in kline uç noktasındaki "taker buy base asset volume" alanından türetilir. Emir defteri anlık görüntüsünden (`orderbook_imbalance` vb., geçmişi yok) FARKLI ve TAMAMLAYICI: mum bazlı olduğu için TAM GEÇMİŞE sahip, backfill gerektirmez — ama her istekte ekstra bir ağ çağrısı gerektirdiğinden opsiyonel/NaN-toleranslı tutuldu (borsa desteklemiyorsa veya hata olursa nötr/0.0)
- [x] Fraktal analiz: kayan pencereli Hurst üsteli (`hurst_exponent`, bkz. `app.ml.advanced_indicators.rolling_hurst_exponent`) — trend-devamlılığı (H>0.5) ile ortalamaya-dönüşü (H<0.5) ayırt eder. Ayrı bir "fraktal boyut" özelliği BİLİNÇLİ OLARAK eklenmedi: D=2-H ilişkisiyle Hurst'ten birebir türetilebildiğinden modele yeni bilgi katmaz (gereksiz redundant kolon)
- [ ] Duygu analizi (haber/sosyal medya, FinBERT vb.) — kullanıcı önerdi, henüz eklenmedi: yeni, ağır bir bağımlılık (`transformers`, ~440MB+ model indirme) VE ücretsiz/güvenilir bir veri kaynağı (RSS + FinBERT mümkün ama backfill'i yok, yalnızca ileriye dönük toplanabilir — makro/order-book'la aynı sınırlama) gerektiriyor; kullanıcının açık onayını bekliyor (bkz. RL'nin `stable-baselines3` bağımlılığı için izlenen aynı desen)
- [x] Tüm modeller için otomatik yeniden eğitim + kalıcılık — önceden yalnızca XGBoost (+meta-label) için VARSAYILAN KAPALI, sabit 24 saatlik bir job vardı; LSTM/online/regime hiç otomatik yenilenmiyordu VE her redeploy (`recreate-backend.sh`) modelleri komple siliyordu (imaja `.gitignore`'lı `app/ml/artifacts/` gömülmediği için). Şimdi: (1) aralık ARTIK sabit değil, `app.scheduler.jobs.compute_auto_retrain_interval_seconds()` ile eğitim penceresinin (`ml_train_lookback`) ne kadarının (varsayılan %5) yeni veriyle değiştiğine göre HESAPLANIYOR (`ml_train_lookback`/`ml_train_timeframe` değişirse otomatik ölçeklenir) — ham hesap varsayılanlarla ~20.8 gün verir, ama `ml_auto_retrain_max_seconds` (varsayılan 7 gün) ile YUKARI SINIRLANIR, yani en geç haftada bir yeniden eğitim garanti edilir; (2) LSTM/online/regime için de ayrı job'lar eklendi (`job_auto_retrain_lstm/online/regime`) — YALNIZCA ilgili ensemble bayrağı açıksa veya model daha önce en az bir kez elle eğitilmişse çalışırlar, hiç kullanılmayan bir modeli sıfırdan başlatmazlar; (3) `docker run`/`docker-compose` artık `app/ml/artifacts`'i named volume'e (`fourkeys_ml_artifacts`) bağlıyor — redeploy'lar modelleri SİLMİYOR, yalnızca hesaplanan aralıkta gerektiğinde otomatik yenileniyor. Bilinen risk (henüz çözülmedi): ağır eğitimler hâlâ ayrı bir process'te değil, aynı uzun ömürlü uvicorn process'inde çalışıyor — PyTorch'un bellek ayırıcısı nedeniyle tekrarlanan LSTM eğitimleri kümülatif bellek artışına yol açabilir; job'un periyodu (en geç 7 günde bir) bunu pratikte hâlâ seyrek kılar
- [x] Sistem backtest'i (`app.backtest.system_runner`) YANLIŞ dönemi test ediyordu — `fetch_full_history`'i kullanıyordu, bu fonksiyon BİLEREK 2017'den İLERİYE doğru sayfalıyor (DCA/JSON-strateji backtest'i için doğru davranış, "piyasa döngülerinin en başından test et") — sistem backtest'i için YANLIŞ, çünkü model GÜNCEL veriyle eğitiliyor. Sonuç: 10.000 mumluk bir istek "2019-2020" gibi alakasız bir dönemi test ediyordu, "şimdi"ye yakın hiçbir şeyi değil. `fetch_ohlcv_cached` (eğitimle AYNI yol) kullanacak şekilde düzeltildi — artık en son mumlar test ediliyor, DB önbelleğinden de faydalanıyor
- [x] XGBoost'un tekrar tekrar %100 tek sınıfa (nötr) çökmesinin GERÇEK kök nedeni bulundu ve düzeltildi. Bisect için kullanıcı `/ml/train`'i HİÇBİR AYAR YAPMADAN (varsayılan `horizon=5`) çalıştırdı — sonuç, `horizon=3` ile AYNI şekilde tam çöküş (`out_of_sample_predicted_class_counts={"0.0": 1980}`, 1980/1980) verdi; bu, önceki oturumlarda "en iyi" bulunan etiketleme ayarının (`horizon=3`, aslında LSTM için taranmış, XGBoost'a hiç ayrı doğrulanmadan uygulanmış) suçlu OLMADIĞINI kanıtladı. Asıl neden: `SignalModel.predict()`/`predict_batch()` YÖN kararını (argmax) `CalibratedClassifierCV` ile KALİBRE EDİLMİŞ olasılıklardan alıyordu — ağır sınıf dengesizliğinde (bu veri setinde ~%82 nötr) kalibrasyon, olasılıkları gözlemlenen taban orana çekme doğası gereği, argmax'ı neredeyse HER ZAMAN çoğunluk sınıfına sabitliyor, `_XGBClassifierWrapper.fit()`'teki sınıf ağırlıklandırmasının ham modele kazandırdığı ayrımı MASKELİYORDU. Düzeltme: YÖN artık HAM (kalibrasyonsuz) modelden alınıyor; GÜVEN skoru (Kelly boyutlandırma gibi tüketiciler için) hâlâ kalibre edilmiş olasılıktan alınıyor — ikisi ayrıştırıldı. Gerçek regresyonu yakalayan bir birim testi eklendi (`test_calibration_direction.py` — bilinçli olarak ZOR/örtüşen bir senaryo kurulmalı, net ayrılmış senaryolar bug'ı hiç göstermiyordu)
- [x] LSTM/online model ensemble'ının canlıda aktif olması ARTIK statik bir ayar bayrağı (`FOURKEYS_ENSEMBLE_LSTM_ENABLED`/`FOURKEYS_ENSEMBLE_ONLINE_ENABLED`, kaldırıldı — artık okunmuyor) gerektirmiyor; her eğitimin SONUNDA otomatik belirleniyor (`app.ml.model_status`): bir model o eğitiminde `ml_min_balanced_accuracy` (0.37) eşiğini geçtiyse yanına bir `.status.json` yazılıp otomatik devreye giriyor, geçemediyse — DAHA ÖNCE KAYDEDİLMİŞ ESKİ DOSYASI DİSKTE OLSA BİLE — otomatik devre dışı kalıyor ("düşük puanlı modelin eski verisini kullanma" kuralı). `app.engine.service.run_cycle_once` artık `is_model_enabled()` ile bu durumu okuyor
- [x] XGBoost'un tekrar tekrar tam olarak rastgele seviyeye (balanced_accuracy=0.333) çökme sorununun teşhis araçları + olası bir kök neden düzeltmesi. `OutOfSampleReport` artık `true_class_counts`/`predicted_class_counts` taşıyor (API yanıtlarında ve `/ml/train-all` detaylarında görünür) — balanced_accuracy'nin TAM OLARAK 1/3 olması, modelin holdout'ta HER ZAMAN tek bir sınıfı tahmin ettiğinin matematiksel imzasıdır, bu alanlar bunu doğrudan gösterir. Ayrıca `SignalModel`'in kalibrasyon eşiği (`CalibratedClassifierCV`, cv=3) önceden yalnızca "sınıf başına en az 3 örnek" idi — bu, azınlık sınıfa cv katı başına ~1 örnek düşmesine izin veriyordu; bu kadar az örnekle fit edilen kalibrasyon eğrisi gürültüyü öğrenip azınlık sınıfın olasılığını sistematik olarak bastırabiliyor (bilinen bir sklearn `CalibratedClassifierCV` tuzağı, imbalanced küçük veride). Eşik "sınıf başına cv katı başına en az 10 örnek"e (`min_class_count >= cv * 10`) yükseltildi. Bu KESIN kanıtlanmış bir kök neden DEĞİL — bir sonraki `/ml/train-all` çalıştırmasında yeni class_counts alanlarıyla doğrulanacak
- [x] OHLCV DB önbelleği (`app.exchanges.cache.fetch_ohlcv_cached`) — önceden her eğitim/karar döngüsü `ml_train_lookback` (varsayılan 10.000) mumu HER SEFERİNDE baştan borsadan çekiyordu. Şimdi `ohlcv_raw` tablosunda önceden kaydedilmiş mumlar varsa borsaya HİÇ gidilmiyor; "bayat" (son 2 bardan eski) ise yalnızca EKSİK kuyruk borsadan çekilip (sıra ile, `since` ile tamamlayarak) DB'ye ekleniyor. XGBoost/meta/online/regime eğitim yolu (`app.ml.dataset`), LSTM/PatchTST (`app.ml.sequence_dataset`) ve canlı karar motorunun (`DecisionEngine`) her 5 dakikalık döngüsü hepsi bu önbelleği kullanıyor. DB kapalıysa şeffaf şekilde eski (doğrudan borsa) davranışa düşer
- [x] Eğitim veri kalitesi kontrolü + kalite kapısı — `app.ml.data_quality.warn_if_gaps` eğitim verisindeki (özellikle DB önbelleğinden okunan) zaman damgası boşluklarını (kayıp mum) tespit edip log'a uyarı yazar (satırların >%1'i kayıpsa). Ayrıca `Settings.ml_min_balanced_accuracy` (varsayılan 0.37 — 3 sınıflı problemde rastgele seviyenin ~1/3'ün hemen üzerinde) bir KALİTE KAPISI: XGBoost/LSTM/PatchTST/online/rejim modellerinden biri bu eşiğin ALTINDA bir out-of-sample (veya online'da prequential) dengeli doğruluk verirse model diske KAYDEDİLMEZ — önceden eğitilmiş (varsa) model dosyası KORUNUR, canlı karar motoru eski/iyi modeli kullanmaya devam eder. Reddedilme `TrainingResult.accepted`/`rejection_reason` (API yanıtlarında da görünür) ile açıkça raporlanır; meta-label, reddedilen (kaydedilmeyen) bir birincil model üzerinde eğitilmez (tutarsız olurdu), o adım da atlanır
- [x] Screener hacim/fiyat ön-filtresi — önceden `scan_market` piyasadaki TÜM sembolleri (yüzlerce) tek tek `fetch_ohlcv` ile tarıyordu; production'da log'lar bunun kendi periyodundan (`screener_refresh_seconds`, önceden 60sn) çok daha uzun sürdüğünü, hatta ASLA bitmediğini gösterdi (APScheduler sürekli "maximum number of running instances reached" ile atlıyordu) — bu yüzden eğitim evreni yalnızca 1 sembole düşüyor, sonuçlar (ör. XGBoost oos_balanced_acc) tekrarlanamaz/güvenilmez oluyordu. Şimdi: yeni `Exchange.fetch_tickers()` (TEK toplu istek, tüm sembollerin fiyat+24s hacmi) ile önce ucuz bir ön-filtre uygulanıyor — `screener_min_price` (varsayılan 0.1 USDT) altındaki semboller elenir, kalanların en yüksek `screener_volume_top_pct`'i (varsayılan %20) hacme göre tutulur — pahalı gösterge hesaplaması yalnızca bu küçük alt kümede çalışır. `screener_top_n` 10→5'e, `screener_refresh_seconds` 60→300'e çekildi; Binance ccxt client'ına da `enableRateLimit=True`+`timeout=15000` eklendi (önceden hiç ayarlanmamıştı)
- [x] `POST /ml/train-all` (`deploy/train-all.sh`) — tüm modelleri (XGBoost → meta-label → LSTM → online → regime) TEK çağrıda, deploy script'lerinde doğrulanmış AYNI parametrelerle sırayla eğitir. Kullanıcının 5 ayrı script'i elle çalıştırması yerine — özellikle ilk kurulumda (`fourkeys_ml_artifacts` volume'ü boşken). Her adım bağımsız try/except ile sarılı: biri başarısız olursa diğerleri yine de çalışır, her adımın sonucu (`ok`/`detail`) ayrı raporlanır
- [x] Sistem backtest'i (`POST /backtest/system/run`, `GET /backtest/system/latest`, frontend "Backtest" sayfası) — DCA/JSON-strateji motorlarından FARKLI: canlı karar motorunun kullandığı AYNI eğitilmiş model (+ varsa meta-label filtresi) BTCUSDT.P futures üzerinde varsayılan olarak 10.000 saatlik mumu (`ml_train_timeframe`/`ml_train_lookback` ile aynı) bar-bar tekrar oynatır. Talep üzerine (on-demand) çalışır, otomatik/periyodik değildir. Bilinen basitleştirmeler: makro/order-book/taker-flow özellikleri hesaplanmaz (yalnızca anlık değerleri var — eğitim setindeki eski barlarla AYNI şekilde nötr kabul edilir, yeni bir yanlılık değil); kademeli alım/Kelly/VIX filtresi yok (her sinyalde tüm equity ile tek giriş/çıkış) — hepsi yanıtın `warnings` alanında da raporlanır. Sonuç `backtest_runs`/`backtest_trades` tablolarına kaydedilir; kullanılan OHLCV geçmişi `ohlcv_raw`'a backfill edilir. Grafana'da yeni "4keys — Sistem Backtest" dashboard'u (Postgres veri kaynağı): BTCUSDT.P 1h candlestick paneli + al/sat noktaları (annotation) + kümülatif equity grafiği + son çalıştırma özeti
- [x] Sistem backtest'i: ATR tabanlı risk yönetimi + LSTM/online ensemble + karar şeffaflığı + pozisyon boyutu açıklaması — kullanıcı isteği üzerine sabit yüzdelik `stop_loss_pct` KALDIRILDI, yerine volatiliteye göre ölçeklenen ATR (Average True Range, `atr_period` varsayılan 14) tabanlı `atr_stop_loss_mult` (varsayılan 1.5×ATR) + opsiyonel `atr_take_profit_mult`/`atr_trailing_mult` eklendi (bkz. bir alt madde — ikisi de VARSAYILAN KAPALI). Ayrıca canlı karar motoruyla (`app.engine.service.run_cycle_once`) TUTARLI olacak şekilde `use_ensemble=true` (varsayılan) iken LSTM/online modeller (yalnızca `app.ml.model_status.is_model_enabled()` ile aktiflerse) `DecisionEngine._combine_predictions` AYNI kuralla (XGBoost+LSTM, sonra +online) birleştirilir — LSTM tahminleri `sliding_window_view` ile vektörize üretilir, online model her bar için ayrı `predict()` çağrılır. Her kapanan işlemde artık: `size_quote` + `size_explanation` (equity'nin tamamı kullanıldığını açıkça belirtir), her modelin ayrı yön/güveni (`xgboost_direction/confidence`, `lstm_direction/confidence`, `online_direction/confidence`) ve nihai kararın nasıl oluştuğunu özetleyen okunabilir bir `decision_reason` metni kaydedilir/raporlanır. `exit_reason` artık `"take_profit"`/`"trailing_stop"` değerlerini de alabiliyor. Var olan `backtest_trades` tablosuna eklenen yeni String kolonlar için `_add_missing_columns` (önceden yalnızca Float destekliyordu) String/Text tipini de otomatik göç edecek şekilde genişletildi
- [x] **Düzeltme — kâr-alma/trailing varsayılan olarak KAPALI'ya çevrildi**: ilk denemede (`atr_take_profit_mult=1.5`, `atr_trailing_mult=0.5` varsayılan AÇIK) production'da gerçek bir backtest, aynı dönem için toplam PnL'in %88'den %32'ye, kazanma oranının %90'dan %57'ye düştüğünü gösterdi. Kök neden: stop-loss yalnızca KAYBEDEN işlemleri sınırlar (kazananları hiç etkilemez), ama kâr-alma/trailing MEKANİK olarak kazanan bir işlemi SABİT bir ATR mesafesinde keser — modelin kendi (ensemble) sinyali hâlâ güçlüyken bile; eski (ATR öncesi, `stop_loss_pct=3.0`) "dinamik" yöntemde çıkışı her barda YENİDEN üretilen model sinyali (`close_confidence`) belirliyordu, sabit bir mesafe değil — bu da kazananların çok daha uzun "koşmasına" izin veriyordu. `atr_take_profit_mult`/`atr_trailing_mult` şimdi varsayılan `null` (kapalı) — çıkış yine birincil olarak dinamik sinyale dayanıyor, `atr_stop_loss_mult` (varsayılan 1.5×ATR, hâlâ açık) ise sadece volatiliteye duyarlı bir güvenlik ağı olarak kalıyor. Not: bu düzeltmeyle AYNI production koşusunda ensemble (LSTM+online) de İLK KEZ devredeydi (önceki %88'lik koşu yalnızca XGBoost kullanıyordu) — yani gözlenen düşüşün ne kadarının ATR çıkışlarından, ne kadarının ensemble birleştirme kuralından geldiği İZOLE EDİLMEDİ; bu iki değişken aynı anda değişti. `use_ensemble=true/false` ile aynı dönemi karşılaştırmak (bkz. roadmap) bunu ayrıştıracak
- [x] Stop-loss zorunlu uygulama (canlı karar motoru): `DecisionEngine._open()` içinde `stop_loss_price` önceden yalnızca Kelly boyutlandırma hesabında kullanılıp atılıyordu — pozisyon üzerinde HİÇBİR ZAMAN saklanmıyor/kontrol edilmiyordu, yani model sinyalini değiştirmediği sürece fiyat o seviyeyi geçse bile pozisyon asla kapanmıyordu. Artık `PortfolioPosition.stop_loss_price` alanına kaydediliyor ve her döngüde `stop_loss_breached()` ile kontrol edilip aşılırsa modelin güncel sinyalinden BAĞIMSIZ olarak zorla kapatılıyor (`RiskRules.stop_loss_enabled`, varsayılan açık)
- [x] İşlem maliyeti (komisyon + kayma) simülasyonu — önceden sistemde HİÇBİR YERDE modellenmiyordu (canlı `PortfolioManager`, JSON-strateji `/backtest` motoru, DCA simülatörü hepsi PnL'i saf fiyat farkından hesaplıyordu; DCA simülatörünün kendi docstring'i bunu açıkça bir basitleştirme olarak belirtiyordu). Artık üç motorda da `commission_pct`/`slippage_pct` (varsayılan Binance Futures taker ücreti ~%0.04 + mütevazı kayma tahmini ~%0.02) round-trip olarak gerçekleşen PnL'den düşülüyor: `PortfolioManager.close_tranche()` (giriş+çıkış = 2 bacak), `app.strategy.engine.run_backtest()` (giriş+çıkış = 2 bacak, bariyer kontrolleri BRÜT fiyatla yapılır — gerçek emir de bu seviyelere brüt hareketle ulaşır), `app.dca.simulator.simulate_dca()` (base order + her dolan safety order + 1 çıkış = değişken sayıda bacak, DCA'nın çok-bacaklı doğasına uygun)
- [x] **Karlılık odaklı 5 parçalı iyileştirme** (kullanıcı isteği: "günlük %1'e yakınsama" hedefiyle — bu hedefin kendisi gerçekçi değil, bileşik olarak yılda ~%3700 demek; buna göre "ayarlamak" aşırı uyuma (overfitting) yol açar, bunun yerine gerçek/tekrarlanabilir bir edge için mühendislik yapıldı):
  1. **Backtest/eğitim veri sızıntısı düzeltmesi** (muhtemelen en temel düzeltme): sistem backtest'i varsayılan olarak "en son N mum" istiyordu — bu, modelin KENDİ eğitim penceresiyle (`holdout_frac=0.2` varsayılanıyla) ~%80 çakışıyordu, yani şimdiye kadar görülen TÜM backtest sayılarının (88%, 32%, 14% PnL) güvenilirliği şüpheliydi (ezber/overfitting kısmen sorumlu olabilir). Artık her kabul edilen eğitim, holdout'un GERÇEK başlangıç zaman damgasını `app.ml.model_status`'a (`get_holdout_start_time`) kaydediyor; backtest varsayılan olarak (`restrict_to_holdout=true`) yalnızca bu tarihten SONRAKİ, modelin `fit()` sırasında HİÇ görmediği barları oynatıyor — ısınma (gösterge/ATR/LSTM `seq_len` penceresi) için holdout ÖNCESİ barlar hâlâ kullanılabiliyor (bu sızıntı SAYILMAZ, gelecek bilgisi değil). Eski/kayıtsız modeller için açık bir uyarıyla eski davranışa güvenli şekilde geri düşülür; `restrict_to_holdout=false` ile bilinçli "ezber dahil" bir önizleme de mümkün.
  2. **ATR-hizalı "triple barrier" etiketleme**: birincil XGBoost modeli artık varsayılan olarak (`labeling_method="atr_triple_barrier"`) "N mum sonra %X hareket etti mi" (gerçek işlemle ilgisiz, keyfi bir hedef) yerine "ATR-ölçekli bir kâr hedefine mi (varsayılan 1.5×ATR) yoksa stop'a mı (1.5×ATR) önce ulaştı" diye eğitiliyor — gerçek ATR tabanlı çıkış mantığıyla (backtest/canlı) AYNI ölçü, etiketleme ile gerçek kâr mekanizması arasındaki uyumsuzluğu (objective mismatch) giderir. `app.ml.labeling.triple_barrier_labels` artık satır-bazında (ATR gibi) değişen bariyer genişliği de kabul ediyor. LSTM/online/regime/meta-label BİLİNÇLİ OLARAK değiştirilmedi (etkiyi izole ölçmek için önce yalnızca birincil model) — bu, gelecekte genişletilebilecek bir roadmap maddesi.
  3. **Open Interest (açık pozisyon) veri kaynağı** — perpetual futures'a özgü, sistemde daha önce HİÇ olmayan bir sinyal: `oi_change_pct` (ardışık anlık görüntüler arası % değişim) + `oi_price_divergence` (fiyat yönü ile OI yönünün işaret uyumu — +1: fiyat+OI aynı yönde, "yeni pozisyonlarla" trend sürüyor; -1: zıt yönde, "pozisyon kapanışıyla" — klasik OI/fiyat 4 çeyrek analizinin basit sayısal karşılığı). `app.orderbook`/`app.macro` ile AYNI desen: borsalar geçmişe dönük OI saklamaz, bu yüzden `open_interest_snapshots` tablosu yalnızca toplamaya başladığımız andan itibaren (yeni periyodik job, varsayılan 30 dk) SEMBOL BAZINDA birikir — eski barlarda nötr (0.0) kabul edilir, yeni bir yanlılık eklenmez.
  4. **Çoklu zaman dilimi (4h/1d) trend bağlamı** — "üst zaman diliminin trendi yönünde işlem yap" prensibi: `htf_4h_ema_gap`/`htf_4h_rsi_norm`/`htf_1d_ema_gap`/`htf_1d_rsi_norm`. Macro/orderbook/OI'dan FARKLI olarak harici bir veri kaynağına bağlı DEĞİL — kaynak 1h OHLCV'den `pandas.resample` ile türetilir (`label="right", closed="left"`: her üst-TF bar, o barın TAMAMEN KAPANDIĞI zamanla etiketlenir), `merge_asof(direction="backward")` ile geriye dönük hizalanır — bir 1h bar YALNIZCA o ana kadar TAMAMEN KAPANMIŞ üst-TF barları görebilir, geleceğe bakma (look-ahead bias) YARATILMAZ (özel bir regresyon testiyle doğrulandı). Bu yüzden backtest'te de GERÇEK değerlerle hesaplanır (basitleştirme gerekmez) — ama yetersiz geçmişte (60 üst-TF barından az) TAMAMEN NaN kalabileceğinden (kısa bir backtest penceresinde TÜM veri setini boşaltabilirdi — gerçekte yaşanan bir regresyon, düzeltildi), yalnızca bu kolonlar zorunlu `dropna`'nın DIŞINDA tutulup eksikse nötr (0.0) ile doldurulur.
  5. **Ensemble birleştirmeyi beceri-ağırlıklı yap**: `DecisionEngine._combine_predictions`'daki sabit, elle yazılmış katsayılar (mutabakatta ortalama×1.1, tek yönlüde ×0.7) kaldırıldı — yerine her modelin KENDİ doğrulanmış `balanced_accuracy`'sinden (`app.ml.model_status.get_balanced_accuracy`) türetilen bir "beceri ağırlığı" (`_skill_weight`: rastgele seviyenin altı 0.0, mükemmel 1.0) geldi. Mutabakat durumunda güven artık beceriye göre AĞIRLIKLI ortalama (daha doğru model daha fazla ağırlık taşır); tek yönlü durumda indirim, o modelin KENDİ becerisine göre ölçeklenir (yüksek beceri → az indirim). Zıt yönlerde nötre düşme kuralı BİLEREK değiştirilmedi (çakışan sinyallerde temkinli kalmak kasıtlı). `app.backtest.system_runner` da AYNI ağırlıklandırmayı canlıyla tutarlı şekilde uygular.
  
- [x] **Üretimde bulunan 3 hata düzeltmesi** (ilk deploy sonrası backtest "0 işlem / %0 PnL" döndü — her biri ayrı regresyon testiyle kanıtlandı):
  1. **Holdout zaman damgası yanlış sembolden alınıyordu**: eğitim BTC + korelasyonlu ikinci bir sembolle yapıldığında (bkz. `select_training_symbols`), `holdout_start_time` TÜM sembollerin EN ERKEN tarihinden hesaplanıyordu. İkinci sembolün geçmişi farklı bir takvim aralığındaysa kaydedilen tarih çok erkene kayıyor, backtest'in holdout filtresi HİÇBİR barı dışlamıyordu (yani (1) numaralı sızıntı düzeltmesi pratikte hiç devreye girmiyordu). Artık yalnızca `settings.ml_primary_symbol`'ün (BTC) kendi holdout satırlarından hesaplanıyor.
  2. **Beceri ağırlıklandırması eski davranıştan daha SERT indirim yapıyordu**: tek-yönlü durumdaki indirim aralığı 0.5..1.0'dı; gerçek modellerin becerisi düşük olduğu için (balanced_accuracy ~0.40-0.46 → skill ~0.10-0.19) çarpan ~0.55'e düşüyor ve iki ardışık adımda (LSTM + online) güveni `open_confidence` eşiğinin altına çekiyordu. Aralık 0.7..1.0'a çekildi — eski sabit 0.7 artık TABAN, beceri ağırlıklandırması hiçbir zaman eski davranıştan kötü olamaz (regresyon testiyle sabitlendi).
  3. **ATR etiketlemesinde zaman bariyeri çok kısaydı**: `train_all_models` eski `horizon=3` değerini kullanıyordu (eski "N mum sonraki getiri" etiketlemesinden kalma). ATR bariyerleriyle 3 bar, 1.5×ATR'lik bir hareket için çok kısa — örneklerin ezici çoğunluğu zaman aşımıyla NÖTR etiketleniyordu (üretim holdout'unda %84,7 nötr), model yönlü sınıfları düşük güvenle tahmin ediyordu. `horizon=12` (1h'de yarım gün) yapıldı.
  
  4. **ASIL KÖK NEDEN — güven eşikleri ulaşılamazdı (hem backtest hem CANLI motor)**: yukarıdaki teşhis çıktısı sayesinde bulundu. `SignalModel`, kalibrasyon çökmesi düzeltmesinden beri yönü HAM modelden, GÜVENİ ise KALİBRE EDİLMİŞ olasılıktan okuyor. Kalibrasyon, olasılıkları taban orana doğru SIKIŞTIRIR: 3 sınıflı bir problemde rastgele seviye 0.333'tür ve gerçek ölçümde yönlü tahminlerin güveni p50=0.44, p90=0.50, p99=0.53, **maksimum 0.56** çıkıyor. Yani `open_confidence=0.6` varsayılanı (ham olasılık döneminden kalma) **matematiksel olarak ulaşılamazdı** — ölçümde barların **%0'ı** bu eşiği geçiyordu. Dahası şema `ge=0.5` alt sınırıyla doğru değerin API'den verilmesini bile engelliyordu. Eşikler ölçülen dağılıma göre yeniden belirlendi: `open_confidence=0.5` (yönlü barların en üst ~%12'si — hâlâ seçici ama ulaşılabilir), `close_confidence=0.45` (çıkmak girmekten kolay olmalı), alt sınır `ge=0.34`. **Bu, yalnızca backtest'i değil CANLI/paper motorunu da etkiliyordu** — `app.engine.service.run_cycle_once` bu eşikleri açıkça geçmediği için `DecisionEngine`'in aynı 0.6/0.55 varsayılanlarını kullanıyordu, yani canlı sistem de pratikte hiç pozisyon açamıyordu. İki testle korunuyor: gerçekçi gürültülü seri üzerinde uçtan uca duman testi + varsayılanların kalibre ölçekte kalmasını garanti eden deterministik kural testi.
  
  5. **Eğitim süresi regresyonu**: `compute_multi_timeframe_features`, yalnızca İKİ kolon (`ema_gap`, `rsi_norm`) için 42 göstergelik TÜM `build_features`'ı çağırıyordu — 10.000 barlık seride 2,35 s. Bu maliyet `_build_symbol_frames`/`build_sequence_dataset` üzerinden eğitim başına onlarca kez, ayrıca canlı karar döngüsünde sembol başına her turda ödeniyordu. Artık aynı EMA/RSI'yı `compute_indicators` ile üretiyor: **2,35 s → 0,059 s (40×)**, üretilen değerler `build_features` ile bire bir aynı (testle doğrulandı).
  6. **Ensemble üyeleri FARKLI hedefler öğreniyordu (yapısal hata)**: `train_all_models`, XGBoost'u `atr_triple_barrier`/`horizon=12` ile, LSTM'i `threshold`/`horizon=3` ile, online'ı `threshold`/`horizon=5` ile, rejim modellerini `threshold`/`horizon=3` ile eğitiyordu — yani üç ensemble üyesi ÜÇ FARKLI soruyu cevaplıyordu. Farklı soruların cevaplarını harmanlamak sağlam bir ensemble değildir: güvenleri aynı ölçekte olmaz ve `_skill_weight` FARKLI problemlerde ölçülmüş doğrulukları kıyaslar. Üretimde somut sonucu şuydu: XGBoost'un ZOR/dengeli problemdeki 0,375'i, online'ın KOLAY/nötr-ağırlıklı problemdeki 0,502'siyle kıyaslanıp haksız yere düşük ağırlık aldı (beceri 0,063'e düştü) ve üç model de aynı yönde hemfikirken bile birleşik güven 0,57'de kaldı. Artık hepsi tek bir `_ENSEMBLE_LABELING` sözlüğünden besleniyor; bir üyenin elle farklı parametre geçirmesi AST tabanlı bir testle engelleniyor.
  6b. **Beş üyeden biri (meta-label) hizalamada atlanmıştı**: yukarıdaki düzeltmeden SONRA bile backtest'te 768 açılış girişiminin 767'si meta-label filtresi tarafından veto ediliyordu (bkz. `app.backtest.system_runner`'ın yeni az-işlem teşhisi — bu sorunu tam olarak bunun için eklemiştik). Kök neden: `train_meta_label_model`, "birincil DOĞRU tahmin etti mi" sorusunu KENDİ eski varsayılanıyla (`threshold`/`horizon=5`) ölçüyordu — XGBoost'un GERÇEKTE öğrendiği (`atr_triple_barrier`/`horizon=12`) soru değil. Yani meta-label, primary kendi sorusunun cevabını doğru verse bile TAMAMEN FARKLI bir soruya göre "yanlış" damgası vuruyordu. Ölçüldü: AYNI etiketle "primary doğru" oranı %92,2, UYUMSUZ etiketle %52,7 — üretimde gözlenen neredeyse-tam-veto ile aynı yönde. `train_meta_label_model` çağrısı da artık `_ENSEMBLE_LABELING`'i kullanıyor; AST testi bu beşinci üyeyi de kapsayacak şekilde genişletildi.
  7. **Eski model / yeni özellik seti uyumsuzluğu anlaşılır hale getirildi**: `ALL_FEATURE_COLUMNS`'a yeni bir özellik eklendiğinde (bu turda 6 tane) diskteki eski model uyumsuz kalır ve XGBoost ham bir `feature_names mismatch` dökümü fırlatıyordu (kullanıcı için anlamsız). Artık `SignalModel`, eğitildiği özellik listesini model dosyasına da yazıyor; uyumsuzlukta `StaleModelFeaturesError` (bir `ValueError` — API katmanı bunu otomatik 422'ye çeviriyor) ile "kod güncellendikten sonra modeller yeniden eğitilmemiş, `bash deploy/train-all.sh` çalıştırın" diyor ve hangi özelliklerin eksik olduğunu listeliyor. Eski model dosyalarında bu alan olmadığı için kontrol atlanır (geriye dönük uyumlu).

  Ayrıca backtest artık "0 işlem" sonucunu SESSİZ bırakmıyor: sinyalin hiç yönlü çıkmadığını mı, güven eşiğinin mi aşılamadığını (ve görülen EN YÜKSEK yönlü güveni), meta-label'ın kaç sinyali veto ettiğini `warnings` alanında açıkça raporluyor — kök nedeni bulmayı sağlayan da tam olarak bu teşhis oldu.

  Not: (1) foundational olduğu için ÖNCE yapıldı; (2)-(5) birbirinden bağımsız, aynı commit'te ama ayrı ayrı test edildi. `restrict_to_holdout`/etiketleme/yeni özellikler nedeniyle TÜM modellerin (`bash deploy/train-all.sh`) yeniden eğitilmesi ZORUNLU — aksi halde eski modeller yeni özellik sayısıyla (`ALL_FEATURE_COLUMNS` uzunluğu değişti) uyumsuz kalır.
- [x] **Sistem backtest'i: gerçek Kelly/fixed-risk pozisyon boyutlandırma + güven eşiği taraması** (kullanıcı isteği: 169 işlem/%51,48 kazanma/%2,67 PnL/%11,22 max drawdown sonucu üzerine "her ikisi birden" — hem eşik sıkılaştırma hem Kelly boyutlandırma):
  1. **Pozisyon boyutlandırma artık backtest'e ÖZGÜ bir icat değil**: önceden her işlemde `size_quote = equity` (o anki TÜM sermaye) sabitti, `warnings` alanı bunu dürüstçe bir basitleştirme olarak belirtiyordu ama gerçek bir boyutlandırma HİÇ uygulanmıyordu. Artık `app.backtest.system_runner._compute_position_size`, canlı/paper motorunun (`app.portfolio.risk_manager.calculate_kelly_position_size`/`calculate_position_size`) KULLANDIĞI AYNI fonksiyonları çağırıyor — farklı bir formül icat edilmedi. Yeni `SystemBacktestRequest` alanları: `position_sizing_method` (`"kelly"` varsayılan veya `"fixed_risk"`), `risk_per_trade_pct`, `kelly_multiplier`, `kelly_min_trades`, `max_kelly_fraction_pct`, `max_position_exposure_pct`, `confidence_scaling_enabled`, `confidence_scaling_min_scale`. Kelly, yalnızca kapanmış işlem geçmişi `kelly_min_trades`'e ulaşınca VE en az bir kayıp işlem varsa devreye girer (aksi halde `b=kazanç/kayıp` sıfıra bölünür) — ikisi de sağlanmazsa stop mesafesine göre sabit risk yüzdesine (`fixed_risk`) güvenli şekilde düşülür. `closed_trade_pnls` listesi yalnızca o ana kadar GEÇMİŞTE kapanmış işlemlerden beslenir (sızıntı yok — bir işlemin boyutu kendi sonucunu asla göremez). Her işlemin `size_explanation` alanı artık hangi yöntemin, hangi sayılarla, ne sonuç verdiğini (ör. "Kelly (5 işlem geçmişi: kazanma=%80,0, ort.kazanç=%2,00, ort.kayıp=%-1,00 -> tam Kelly=%70,0, uygulanan 0.5x=%25,0); güven ölçeği x1.00 (confidence=0.90) -> 250,00 USDT.") okunabilir şekilde açıklıyor.
  2. **Güven eşiği taraması** (`POST /backtest/system/sweep-confidence`, `app.backtest.system_runner.sweep_confidence_thresholds`) — "eşiği sıkılaştırmak (daha az ama daha seçici işlem) kârlılığı artırır mı" sorusuna CEVAP değil, CEVABI BULMAK İÇİN VERİ sağlar (`app.ml.train.sweep_lookback_values` ile AYNI felsefe: otomatik "en iyi"yi seçmez — az işlemle görülen yüksek bir kazanma oranı, çok işlemle görülen daha düşükten DAHA GÜVENİLİR değildir, karar operatöre kalır). Verilen her `open_confidence` adayı için `close_confidence`'ı türetip (`open - close_confidence_gap`, varsayılan 0.05) tam bir backtest çalıştırır, işlem sayısı/kazanma oranı/PnL/max drawdown döner. `run_system_backtest`'e eklenen `persist=False` parametresiyle (`train_signal_model_validated(persist=...)` ile AYNI desen) ara denemeler `backtest_runs` tablosunu/Grafana panellerini KİRLETMEZ. Frontend'e eklenmedi — `sweep-lookback` ile AYNI konvansiyon: operatör aracı, `/docs`'tan (Swagger UI) veya `bash deploy/sweep-confidence.sh` ile çalıştırılır.
  3. Sonuçları test etmek üretim modelinin YENİDEN EĞİTİLMESİNİ gerektirmez — yalnızca backtest simülasyon mantığı değişti, hiçbir öğrenme/etiketleme/özellik değişmedi.
  4. **Sunucuda ölçüldü, varsayılan eşik yükseltildi**: `bash deploy/sweep-confidence.sh` gerçek (o anki en güncel eğitimin) holdout verisinde çalıştırıldı. Eski varsayılan `open_confidence=0.5`'te `total_pnl_pct` NEGATİFTİ (-%0,25). `0.55`'te 89 işlem, kazanma=%53,9, PnL=+%2,09, max_drawdown=%0,64 (taramanın EN DÜŞÜK drawdown'ı). `0.6`'da 48 işlem, PnL=+%2,70 (en yüksek nokta) ama drawdown biraz daha kötü. `0.65`/`0.7`'de örneklem (22/12 işlem) güvenilir yorum için çok küçüktü (0.7'de kazanma oranı %16,67'ye düşüyor — yüksek güvenin daha güvenilir olmadığının somut göstergesi, muhtemelen küçük örneklem gürültüsü). `0.55` seçildi: `0.6`'ya göre ~2× daha büyük örneklem VE en düşük drawdown; aradaki ham PnL farkı, örtüşen işlemler arasındaki örneklem gürültüsünün içinde kalıyor. `DecisionEngine.__init__` (CANLI motor) VE `SystemBacktestRequest`/frontend varsayılanları `open_confidence=0.55`/`close_confidence=0.5`'e senkron güncellendi — bu, kullanıcının önceki "eşik sıkılaştırma" isteğinin (bkz. yukarıdaki Kelly maddesi) TAMAMLANMASI. Not: TEK bir holdout penceresinde ölçüldü, kalıcı bir kanun değil — her yeniden eğitimden sonra `sweep-confidence.sh` ile yeniden ölçülüp gerekirse ayarlanmalı.
- [x] **OOM üretim olayı: ağır eğitim canlı API process'inden İZOLE edildi, sunucuda DOĞRULANDI** (`bash deploy/train-all.sh` sonrası kernel OOM-killer `uvicorn`'u ÜÇ KEZ öldürdü — üretim sunucusunda diagnostik: toplam RAM yalnızca 3.7GB, **swap YOK**, boşta bile konteyner %49 bellek kullanıyor, öldürülen her `uvicorn` süreci ~2.7-2.8GB anon-rss'e ulaşmıştı):
  1. **Kök neden, önceden roadmap'te işaretlenmişti ama düzeltilmemişti** (bkz. yukarıdaki "Ağır eğitim işlerinin ayrı bir process'te çalıştırılması" maddesi, `[ ]` — LSTM lookback sweep testinde daha önce gözlenmişti): `POST /ml/train-all`, canlı API'yi VE arka plan zamanlayıcısını (`app.scheduler`) barındıran AYNI uzun-ömürlü `uvicorn` process'i içinde çalışıyordu. Beş ağır adımı (XGBoost + meta-label + LSTM + online + regime×3) TEK istekte zincirlemek, bu deseni tetiklemenin en kötü senaryosuydu.
  2. **Acil önlem**: sunucuya 2GB swap dosyası eklendi (`fallocate`/`mkswap`/`swapon` + `/etc/fstab`'a kalıcı kayıt) — sert kill'i "yavaşlar ama hayatta kalır"a çevirir, ama kök nedeni ÇÖZMEZ.
  3. **Kalıcı çözüm — ayrı process izolasyonu**: yeni `backend/app/cli.py` (`python -m app.cli train-all`), `train_all_models`'ı FastAPI/`uvicorn`'dan TAMAMEN BAĞIMSIZ çağırır — `app.main`'i (ve onunla birlikte `start_scheduler()`'ı) KASITLI OLARAK import etmez, yalnızca `init_db()` (tamamen idempotent) çağırır. Bunu bir AST tabanlı test (`test_cli_never_imports_app_main_or_starts_scheduler`) zorunlu kılıyor — ileride biri yanlışlıkla `app.main` import ederse regresyon olarak yakalanır. `deploy/train-all.sh` artık canlı `4keys-backend` konteynerine `curl` ATMAK YERİNE `docker run --rm --memory=2g` ile AYNI imajı, AYRI bir konteynerde, farklı komutla (`python -m app.cli train-all`) çalıştırıyor — aynı ağ (`4keys-net`, DB erişimi için) ve aynı `fourkeys_ml_artifacts` volume'üne (eğitilen modeller kaybolmasın) bağlanıyor. Sonuç: eğitim ne kadar bellek yerse yesin, canlı API process'i ARTIK ETKİLENMEZ — en kötü ihtimalle bu eğitim konteyneri kendi `--memory` sınırında öldürülür (yeniden denenebilir, tek seferlik bir iş), canlı işlem motoru dokunulmamış kalır. `POST /ml/train-all` HTTP endpoint'i geriye dönük uyumluluk için (ör. `/docs`'tan manuel çağrı) hâlâ duruyor — ama ARTIK ÖNERİLEN/BELGELENEN yol değil, o yüzden bilerek AYNI izolasyonu almıyor.
  4. Diğer tekil eğitim script'leri (`train-lstm-btc.sh` vb.) bu turda İZOLE EDİLMEDİ — yalnızca `train-all.sh` (beş adımı zincirleyen, gerçekte OOM'a yol açtığı KANITLANMIŞ olan) düzeltildi; kapsam bilerek dar tutuldu, kanıtlanmamış bir riske erken optimizasyon yapılmadı.
  5. Yeni `deploy/diagnose-oom.sh` (RAM/swap, konteyner durumu/restart sayısı, anlık bellek, `dmesg` OOM geçmişi) ve `deploy/sweep-confidence.sh`'ın kompakt tablo + dosyaya kaydetme deseni — ikisi de Hetzner web konsolunun sınırlı kaydırma geçmişiyle karşılaşan gerçek bir kullanıcı sorununu çözmek için eklendi.
  6. **AYRI KONTEYNER TEK BAŞINA YETERSİZ ÇIKTI — 4. kez `uvicorn` öldürüldü**: `--memory` sınırının canlı API'yi KORUMADIĞI ortaya çıktı. Bu sınır yalnızca eğitim konteynerinin KENDİ üst sınırını belirliyor; canlı API'nin tabanı (~1.8GB) ile eğitim konteynerinin izin verilen üstü (ilk denemede 2GB) TOPLANDIĞINDA kutunun fiziksel RAM'ini (3.7GB) aşıyor, kernel'in GENEL (host-çapında, cgroup'tan BAĞIMSIZ) OOM-killer'ı devreye girip hangi konteyner olursa olsun seçebiliyor — bu sefer `uvicorn` yalnızca ~1.7GB'ta (öncekilerden DAHA DÜŞÜK bir seviyede) öldürüldü, çünkü baskının bir kısmı eş zamanlı çalışan eğitim konteynerinden geliyordu. İki ek önlem eklendi: (a) eğitim konteynerinin bellek sınırı daha muhafazakâr hale getirildi (2GB → 1.5GB) — canlı API'nin tabanıyla toplandığında fiziksel RAM'i AŞMAMASI, yani genel OOM-killer'a hiç ulaşılmaması hedeflenir; (b) her iki konteynere de `--oom-score-adj` verildi — canlı API `-500` ("EN SON öldür"), eğitim konteyneri `+500` ("ÖNCE bunu öldür") — böylece sınır yine de yetersiz kalırsa kernel'in tercihi açıkça canlı API LEHİNE yönlendirilir.
  7. **Sunucuda DOĞRULANDI**: bir sonraki `train-all.sh` çalıştırmasında OOM YİNE tetiklendi (kutu gerçekten 3.7GB'lık sınırında), ama bu sefer öldürülen süreç `python` (`oom_score_adj:500`, eğitim konteyneri) oldu — `uvicorn` DEĞİL. Canlı API doğrulandı: `GET /health` → `{"status":"ok"}`, `docker inspect ... RestartCount=0` (hiç yeniden başlamamış, yani hiç öldürülmemiş). `--oom-score-adj` tasarlandığı gibi çalıştı: kernel, sıkışınca artık HER ZAMAN canlı sistemi değil, yeniden çalıştırılabilir eğitim işini feda ediyor. Açık kalan soru: kutu gerçekten bu kadar küçük olduğu için `train-all` tek seferde nadiren tam bitebiliyor (bu koşuda LSTM adımından sonra kesildi) — birden fazla deneme gerekebilir; canlı API'nin boştaki ~1.8GB tabanının kendisi neden bu kadar yüksek olduğu (muhtemelen torch+xgboost+pandas+sklearn+river'ın hepsinin TEK process'e aynı anda yüklenmesinin doğal maliyeti) bu turda araştırılmadı.
  8. **Eksik parça bulundu: kutuda backend TEK BAŞINA değildi**. `docker stats --no-stream` tam resmi gösterdi: `4keys-backend` (1.42GB) + `4keys-grafana` (335MB) + `4keys-prometheus` (50MB) + `4keys-db`/TimescaleDB (348MB) — dördü BİRLİKTE ~2.15GB, 3.7GB'lık kutuda eğitime neredeyse hiç pay bırakmıyor (üçüncü OOM'da eğitim konteyneri daha ilk adım bitmeden, yalnızca ~1GB'ta öldürüldü — kendi 1.5GB sınırına bile ulaşamadan). Grafana/Prometheus canlı TRADING işlevine DAHİL DEĞİL, salt izleme — `deploy/train-all.sh` artık eğitim süresince bu ikisini GEÇİCİ olarak durdurup (`docker stop`), eğitim başarılı/başarısız/kesintiye uğrasa bile (`trap ... EXIT`) sonunda GERİ başlatıyor. TimescaleDB durdurulmadı (eğitimin kendisi DB önbelleğine ihtiyaç duyuyor). Script ayrıca artık eğitim konteynerinin çıkış kodunu yakalayıp (`set -e` KALDIRILDI — aksi halde OOM/hata durumunda `trap` çalışmadan script sonlanırdı) kendi çıkış kodu olarak kullanıyor, böylece başarısızlık sessizce yutulmuyor.
  9. **Asıl kök neden bulundu: `torch` HER ZAMAN yükleniyordu, kullanılmasa bile**. Ölçüldü: her bağımlılığın process RSS'ine tek başına eklediği miktar — pandas+numpy (58MB), sklearn (74MB), xgboost (25MB), **torch (~460MB, HEPSİNİN TOPLAMINDAN FAZLA)**, river (~0), ccxt (42MB). `LSTMSignalModel`/`PatchTSTSignalModel` (dolayısıyla `torch`) yedi farklı dosyada MODÜL SEVİYESİNDE import ediliyordu (`app.engine.decision`, `app.engine.service`, `app.backtest.system_runner`, `app.api.routes.backtest`, `app.api.routes.ml`, `app.scheduler.jobs`, `app.ml.train`) — LSTM o an DEVRE DIŞI olsa bile (ki üretimde sürekli kalite eşiğinin altında kalıp öyleydi), `app.main`'in kendisi (dolayısıyla HER istek/döngü) torch'u belleğe yüklemiş oluyordu. Kök neden: `DEFAULT_LSTM_MODEL_PATH`/`DEFAULT_PATCHTST_MODEL_PATH` (yalnızca birer dosya yolu `Path` sabiti) `import torch` satırının HEMEN ARDINDAN tanımlıydı — "sadece dosya var mı diye bakayım" gibi masum bir import bile torch'un TAMAMINI tetikliyordu. Düzeltme: (a) yeni `app.ml.model_paths` — bu iki sabiti torch'suz barındıran ayrı, hafif bir modül; (b) `LSTMSignalModel`/`PatchTSTSignalModel`'in GERÇEK sınıfları artık yalnızca GERÇEKTEN bir model yüklenecek/eğitilecek/tahmin edilecek fonksiyonun İÇİNDE (lazy) import ediliyor — tip belirteçleri (`lstm_model: LSTMSignalModel | None`) `from __future__ import annotations` + `TYPE_CHECKING` ile çalışma zamanında hiç çözülmüyor. Sonuç, sunucuda DOĞRULANMADI ama yerelde ÖLÇÜLDÜ: `app.main`'in TAMAMINI import etmek önceden muhtemelen ~700MB+ tutarken şimdi **267.7MB** (torch hiç `sys.modules`'a girmiyor). Yeni bir subprocess tabanlı regresyon testi (`test_importing_app_main_does_not_load_torch`) bunu kalıcı olarak koruyor — testin AYNI process'te değil AYRI bir subprocess'te çalışması ZORUNLU, çünkü LSTM/PatchTST testleri torch'u zaten import etmiş olurdu. **Bu değişiklik `docker build` GEREKTİRİR** (kod değişti) — bir sonraki `bash deploy/recreate-backend.sh` sonrası canlı API'nin boştaki bellek kullanımının gerçekten düştüğü `docker stats` ile doğrulanmalı.
  10. **Sunucuda DOĞRULANDI, dramatik**: `docker stats` canlı API tabanının **1.42GB → 589.5MB**'a (**%38 → %15.45**) düştüğünü gösterdi — tahmin edilenden bile fazla (~860MB). `--memory` sınırı buna göre 1.5GB → 2.2GB'a yükseltildi (bkz. 6-7. maddeler).
  11. **Hipotez ÇÜRÜTÜLDÜ: sorun çoklu-sembol veri hazırlama DEĞİLMİŞ**. Yeni headroom'la tam (çoklu-sembol) `train-all` yine öldü (~1.45GB'ta) — ama YALNIZCA BTC ile de (`deploy/train-all-btc-only.sh`) NEREDEYSE AYNI seviyede (~1.41GB) öldü, XGBoost+meta-label+LSTM'i BAŞARIYLA TAMAMLADIKTAN SONRA (ikisi kalite eşiğinin altında kalıp reddedildi ama ÇÖKMEDİ). Sembol sayısının belleğe katkısı önemsizmiş; asıl dinamik, README'de ZATEN işaretli olan "5 adımın TEK process'te zincirlenmesi, her adımın (özellikle LSTM'in — PyTorch) bıraktığı belleğin bir SONRAKİNE kümülatif taşınması" imiş.
  12. **Hedefli düzeltme: LSTM, `train_all_models`'a eklenen `skip_steps` parametresiyle ATLANABİLİR oldu**. LSTM bu oturumda HER seferinde kalite eşiğinin (0.37) altında kaldı (0.358, 0.362, 0.365) — yani şu an SIFIR pratik fayda sağlıyor, ama (torch nedeniyle) EN PAHALI tek adım. `python -m app.cli train-all --skip lstm` (ve `/ml/train-all`'ın `skip_steps` alanı) bu adımı HİÇ ÇALIŞTIRMAZ — sonuç "atlandı (skip_steps)" olarak raporlanır, `train_lstm_signal_model` çağrılmadığını doğrulayan bir test (`test_train_all_models_skip_steps_never_calls_the_skipped_function`) eklendi. `deploy/train-all.sh` artık VARSAYILAN OLARAK LSTM'i atlıyor (`INCLUDE_LSTM=1` ile eski davranışa dönülebilir). Henüz sunucuda "bu değişiklikle train-all TAMAMEN bitiyor mu" testi yapılmadı — bir sonraki adım bu.
- [x] **Temel sadeleşme (kullanıcı isteği)**: OOM araştırması sürerken kullanıcı, kurguyu köklü şekilde basitleştirmeye karar verdi — karmaşıklığı azaltıp (bellek baskısını da düşürerek) tek bir, iyi anlaşılan sinyale odaklanmak için:
  1. **Yalnızca BTC/USDT ile eğitim** (`ml_train_max_symbols`: 5 → **1**): `select_training_symbols` ve `_resolve_symbols` artık `limit<=1` olduğunda KISA-DEVRE yapıyor — diğer sembollerin OHLC'si için TEK BİR ağ çağrısı bile yapılmıyor (screener taraması dahil tamamen atlanıyor). Bunu doğrulayan iki test eklendi (`test_max_symbols_le_1_returns_primary_without_any_exchange_calls`, `test_resolve_symbols_skips_screener_scan_when_max_symbols_is_one`) — her ikisi de herhangi bir sembol için çağrılırsa PATLAYAN bir stub kullanıyor.
  2. **BTC mum sayısı 3× artırıldı** (`ml_train_lookback`: 10.000 → **30.000**): tek sembole inince kaybedilen satır hacmini (çoklu-sembolün eskiden kattığı ek satırlar) BTC'nin kendi geçmişini derinleştirerek dengelemek için.
  3. **Çok zamanlı dilim (4h/1d) özellikleri GEÇİCİ olarak durduruldu** (`ml_enable_multi_timeframe_features`: **False**): `compute_multi_timeframe_features` artık bu bayrak kapalıyken HİÇBİR resample/gösterge hesaplaması yapmıyor, tüm kolonları nötr (0.0) dolduruyor — makro/order-book'ta "veri yoksa nötr" için ZATEN kullanılan AYNI desen, kod SİLİNMEDİ (bayrak `True` yapılınca geri döner).
  4. **Rejim modelleri durduruldu**: `deploy/train-all.sh` artık VARSAYILAN OLARAK `regime`'i de atlıyor (LSTM'le AYNI `skip_steps` mekanizması, `INCLUDE_REGIME=1` ile geri açılabilir). Aktif kalan adımlar: XGBoost (birincil) + meta-label + online öğrenme.
  5. **Ortak hedef değişti — 12 bar/1.5×ATR → 3 bar/1.0×ATR**: "3 mum içinde 1×ATR kâr mı yoksa 1×ATR zarar mı önce gelir" — daha kısa vadeli, daha sık sinyal üreten bir soru. Değiştirmeden ÖNCE sentetik saf-gürültü kontrolüyle doğrulandı: bu kombinasyon %18.7 nötr / %40-41 yönlü veriyor (eski "%84.7 nötr" felaketine YAKLAŞMIYOR — o felaket de horizon=3'ün KENDİSİNDEN değil, o zamanki 1.5×ATR çarpanıyla BİRLİKTE davranışından kaynaklanıyordu). Bunu kalıcı olarak koruyan `test_all_ensemble_members_share_one_labeling_definition`'daki eski, yanıltıcı "horizon >= 6" sabit-eşik kontrolü, seçilen (horizon, çarpan) kombinasyonunun saf gürültüde AŞIRI tek-sınıfa çökmediğini doğrudan ölçen bir kontrolle DEĞİŞTİRİLDİ — eski kontrol yalnızca horizon'a bakıyordu, çarpanla BİRLİKTE davrandığını gözden kaçırıyordu.
  6. **Bilinen, kabul edilmiş yan etki — model kalitesi düşebilir**: BTC-only test çalıştırmalarında XGBoost'un `out_of_sample_balanced_accuracy`'si düştü (0.378 → 0.362, eşiği 0.37'yi ARTIK GEÇEMİYOR) — çoklu-sembolün kattığı ek satırların kaybı bunun nedeni. `ml_train_lookback`'in 30.000'e çıkarılması bunu KISMEN dengeler ama tam telafi ettiği DOĞRULANMADI; yeni etiketleme hedefinin (3 bar/1.0×ATR) etkisi de henüz gerçek veride ölçülmedi. Sunucuda yeniden eğitim sonrası kalite eşiğinin geçilip geçilmediği izlenmeli.
- [x] **Asıl OOM kaynağı bulundu: canlı API'nin belleği zamanla büyüyor**. Mum sayısı 30.000 → 15.000 → 10.000 → 5.000 (6× azaltma) ile tekrar tekrar denendi — ölüm noktası neredeyse HİÇ değişmedi (~1.2-1.35GB aralığında sabit kaldı), bu da sorunun eğitim verisinin boyutuyla ORANTILI olmadığını gösterdi. `bash deploy/diagnose-oom.sh` (yeni genişletilmiş hali — artık `docker system df` ve TÜM konteynerleri de gösteriyor) net cevabı verdi: canlı `4keys-backend`'in belleği taze bir restart'ta ~590MB iken, yalnızca **~2.7 saatlik NORMAL çalışma** (eğitim bile yokken — sadece zamanlayıcı + 5 dakikalık karar döngüsü) sonrasında **~1.74GB'a** (%46.7) çıkmıştı. Her eğitim denemesi, GİDEREK BÜYÜYEN bu tabana karşı yarışıyordu — bu yüzden mum sayısını küçültmek işe yaramıyordu, asıl "sabit" olan şey eğitim verisi değil, canlı API'nin kendi (uptime'a bağlı) ayak iziydi. **Kök nedeni (canlı API neden zamanla büyüyor) bu turda BULUNAMADI** — muhtemelen glibc'nin bellek ayırıcısının (zaten eğitim bağlamında bilinen) işletim sistemine tam geri vermeme davranışı, ama bu sefer periyodik zamanlayıcı işleri (screener taraması, karar döngüsü, makro/order-book/OI yenileme) üzerinden — dürüstçe, derinlemesine araştırılmadı. **Pratik önlem**: `deploy/train-all.sh` artık eğitimden HEMEN ÖNCE canlı API'yi otomatik olarak yeniden oluşturuyor (`deploy/recreate-backend.sh`) — her zaman EN DÜŞÜK bellek tabanından başlamasını garanti eder, birkaç saniyelik kısa bir kesinti pahasına (paper-trading, gerçek para riski yok; `SKIP_RECREATE=1` ile atlanabilir). Ayrıca `docker system df` disk tarafında da bir birikim gösterdi (imajların %70'i, 10.78GB, "dangling"/kullanılmayan) — RAM sorunuyla ilgisiz ama `docker image prune` ile temizlenmeye değer.
- [x] **OOM ölüm noktası KANITLANDI — eğitim verisi HACMİYLE İLGİSİ YOK**: yukarıdaki "canlı API'nin belleği zamanla büyüyor" bulgusundan sonra kontrollü bir karşılaştırma yapıldı — `LOOKBACK=5000` anon-rss=1399164kB'ta öldü, `LOOKBACK=1000` (5× daha az veri) anon-rss=1399148kB'ta öldü: **aradaki fark 16KB** (ölçüm hatası payı içinde). Bu, sabit ~1.2-1.4GB maliyetin mum sayısından TAMAMEN BAĞIMSIZ olduğunu kanıtlıyor — 1000 mumluk bir eğitim de 5000 mumluk kadar ölüyor, demek ki tüketilen bellek işlenen VERİYLE değil, BAŞKA bir şeyle orantılı.
  1. **Üç hipotez sırayla test edilip ÇÜRÜTÜLDÜ**: (a) XGBoost'un walk-forward CV + `CalibratedClassifierCV` katları boyunca tekrar tekrar `.fit()` çağırmasının bellek biriktirdiği — doğrudan bir sandbox testiyle çürütüldü: `XGBClassifier(n_estimators=300, max_depth=4)` ile 9 ardışık `.fit()` çağrısı (gerçek kat sayılarıyla eşleşen), 2. fit'ten sonra 184.4MB'ta DÜZ kaldı, büyümedi. (b) `load_macro_history()`/`load_orderbook_history()`/`load_open_interest_history()`'nin (`app/ml/macro_features.py`, `orderbook_features.py`, `openinterest_features.py`) `limit=200_000` ile (lookback'ten TAMAMEN BAĞIMSIZ) koşulsuz TÜM geçmişi çekmesi — gerçek üretim DB sorgusuyla çürütüldü: üçü de küçük (`macro_snapshots`=66 satır/72kB, `orderbook_snapshots`=164 satır/120kB, `open_interest_snapshots`=34 satır/80kB), 200.000 sınırından ÇOK uzak. Bu sorgu artık `deploy/diagnose-oom.sh`'ın kalıcı bir parçası (tablo boyutları raporu). (c) `feature_snapshots` (291.208 satır/147MB) ve `signals` (597.273 satır/90MB) tablolarının eğitim sırasında OKUNDUĞU — kod incelemesiyle çürütüldü: `app.ml.dataset._persist_feature_snapshots` yalnızca YAZAR (`record_feature_snapshots_bulk`, toplu INSERT), eğitim akışının hiçbir yerinde bu tablo geri OKUNMUYOR; onu okuyan TEK yer bir hata ayıklama API rotası (`GET /db/feature-snapshots`).
  2. **(c)'nin YAN ürünü olarak GERÇEK, ayrı bir bulgu — kullanıcı sordu, doğrulandı**: `_persist_feature_snapshots`, LSTM/RL ileride "periyodik birikim yerine tek seferde backfill bulsun" diye eklenmişti (bkz. kod içi docstring) — ama LSTM şu an (`temel sadeleşme` kararıyla) VARSAYILAN OLARAK ATLANIYOR ve onu okuyan hiçbir eğitim kodu yok; yani bu yazma HER `train-all` çağrısında amaçsızca `feature_snapshots`'ı şişiriyordu (291K satır/147MB üretimde ölçüldü). OOM'un NEDENİ olmadığı kanıtlandı (yazma miktarı lookback'le ORANTILI büyür, oysa ölüm noktası lookback'ten bağımsız) ama gereksiz oluşu bağımsız bir düzeltmeyi hak ediyordu: yeni `Settings.ml_persist_feature_snapshots` (varsayılan **False**) ile bu yazma tamamen kapatıldı; LSTM/RL çalışması yeniden başladığında elle `True` yapılabilir. Önceden birikmiş, artık amaçsız kalmış satırları temizlemek için yeni `deploy/truncate-feature-snapshots.sh` (yalnızca `TRUNCATE TABLE feature_snapshots` — canlı karar döngüsü/backtest/mevcut eğitim akışı bunu HİÇ okumadığından güvenli, trading davranışını etkilemez).
  3. **Asıl gizem hâlâ AÇIK — ama artık ölçülebilir**: veri hacmi, XGBoost'un kendi fit döngüsü ve üç "büyük tablo" hipotezinin hepsi çürütülünce geriye kalan en olası aday, `train-all`'ın KENDİ import/kurulum maliyeti (torch zaten lazy ama `app.cli` → `app.api.routes.ml` → `app.ml.train` zinciri hâlâ FastAPI + tüm ensemble modüllerini yüklüyor) ya da XGBoost/sklearn'in kendi C++ arka ucunun (OpenMP thread havuzları vb.) veri hacminden bağımsız, çekirdek sayısına göre ölçeklenen bir maliyeti. Bunu TAHMİN etmek yerine ÖLÇMEK için yeni `app/core/memory_probe.py` eklendi: `MEMORY_PROBE=1` ortam değişkeni ayarlıyken `app.cli` import edildiğinde, her `train_all_models` adımından ÖNCE/SONRA VE `_build_symbol_frames`'in her sembol-başı kontrol noktasında (fetch_ohlcv sonrası, build_features sonrası, tüm merge'ler sonrası) anlık RSS'i (`/proc/self/status`'tan VmRSS) ekrana yazdırır — varsayılan (kapalı) durumda sıfır ek maliyet. `deploy/train-all.sh` artık bunu geçiriyor: `MEMORY_PROBE=1 LOOKBACK=1000 bash deploy/train-all.sh` (en küçük/en hızlı deney) çalıştırılıp çıktı buraya yapıştırılırsa, büyümenin TAM OLARAK hangi kontrol noktasında (import mı, hangi adım mı, hangi sembol-alt-adım mı) olduğu görülebilir — bir sonraki adım bu.
  4. **Beklenmedik gelişme: VARSAYILAN (30.000 mum, TAM) `train-all` OOM'suz TAMAMLANDI**. Kullanıcı `LOOKBACK=1000`/`MEMORY_PROBE=1`'i AYRI komut satırlarında (`LOOKBACK=1000` [Enter] `MEMORY_PROBE=1` [Enter] `bash deploy/train-all.sh` [Enter]) çalıştırdı — bash'te bu şekilde `export` edilmeden atanan bir değişken alt process'e (script'in kendisi) GEÇMEZ, yalnızca `DEĞİŞKEN=değer komut` (tek satırda) veya `export DEĞİŞKEN=değer` bunu sağlar. Çıktıda ne "Mum sayısı ... ile eziliyor" ne de "MEMORY_PROBE=1: ..." satırı vardı — ikisi de UYGULANMADI, yani bu çalıştırma `settings.ml_train_lookback` varsayılanıyla (**30.000**, en kötü/en çok OOM'a yol açan senaryo) gerçekleşti ve **HİÇ OOM olmadan, tüm aktif adımlar (xgboost+meta+online) başarıyla tamamlandı**. Bonus: **kalite eşiği artık geçiliyor** — `xgboost: oos_balanced_acc=0.406` (eşik 0.37, REDDEDİLMEDİ), `online: overall_balanced_acc=0.421` (REDDEDİLMEDİ) — bkz. yukarıdaki "temel sadeleşme" maddesinin 6. alt maddesindeki açık soru, artık YANITLANDI: yeni (3 bar/1×ATR) hedef + 30K lookback kombinasyonu üretimde kaliteyi GEÇİYOR. Kesin kök neden hâlâ izole edilmedi ama en olası açıklama: bu turun tüm önlemlerinin (torch lazy import, Grafana/Prometheus geçici durdurma, `feature_snapshots` yazmasının kapatılması, eğitimden hemen önce canlı API'yi taze bir tabanla yeniden oluşturma, `--oom-score-adj` önceliklendirmesi) BİRİKİMLİ etkisi, sorunu pratikte çözecek kadar headroom açmış olması — güvenilirliği doğrulamak için birkaç kez daha (ideride `MEMORY_PROBE=1 LOOKBACK=1000 bash deploy/train-all.sh` TEK SATIRDA, doğru sözdizimiyle) tekrarlanmalı.
  5. **ASIL KÖK NEDEN BULUNDU VE DÜZELTİLDİ: online model (`river.forest.ARFClassifier`) sınırsız büyüyordu**. Kullanıcı Backtest sayfasında "Failed to fetch" hatası bildirdi; nginx hata logları kanıtı verdi: `POST /backtest/system/run` sırasında `"upstream prematurely closed connection"` — backend TAM O İSTEĞİ İŞLERKEN çöküyordu. `bash deploy/watch-backend-memory.sh` ile canlı izlerken bir OOM+otomatik-restart olayını GERÇEK ZAMANLI yakaladık (`dmesg`: `uvicorn` anon-rss 2.7GB'ta öldürüldü, `-500` "en son öldür" korumasına RAĞMEN — çünkü basit bir şekilde çok fazla mutlak bellek tüketiyordu) ve hemen ardından kullanıcı Backtest'e bastığında taze restart (668MB) ~50 saniyede 2.04GB'a fırladı, sonra 1.13GB'a indi — **yeni "taban" eskisinden tam ~460MB daha yüksekti**. İlk şüphe (torch/LSTM'in yanlışlıkla hâlâ aktif olması) `diagnose-oom.sh`'ın yeni model-durumu bölümüyle ÇÜRÜTÜLDÜ (LSTM `enabled: false` — doğru). Ama dosya listesi asıl suçluyu ele verdi: `online_model.joblib` **208MB** (normalde birkaç MB olması beklenen "hafif" bir online model için akıl almaz büyük). Kod incelemesi + river kütüphanesinin kaynağı okununca kök neden netleşti: `ARFClassifier`'ın bellek sınırlama mekanizması (`max_size`, ağaç başına MB) yalnızca `memory_estimate_period` kadar (varsayılan 1-2 MİLYON) örnek AĞIRLIĞI biriktikten SONRA bir kez kontrol ediyor — bizim eğitimlerimiz (`ml_train_lookback` onbinlerce satır, ARF'nin `lambda_value=6` ortalama ağırlığıyla bile en fazla birkaç yüz bin) bu eşiğe HİÇ ULAŞMIYOR, yani sınır fiilen HİÇ UYGULANMIYOR ve 10 Hoeffding ağacı sınırsız büyüyor. Doğrudan ölçüldü (aynı sentetik, 50 özellikli, ~30K satırlık veride, prequential doğrulukla birlikte): varsayılan ayarlarla 85.3MB/tahmin doğruluğu 0.754; `max_size=5.0` (ağaç başına MB) + `memory_estimate_period=200` ile **45.1MB (%47 azalma)**/doğruluk **0.753** (PRATİKTE DEĞİŞMEDİ). `OnlineSignalModel.__init__` artık bu iki parametreyi (gerçek veri hacmimize göre küçültülmüş varsayılanlarla) `ARFClassifier`'a geçiriyor — `test_online_signal_model_caps_tree_memory_by_default` bunu kalıcı olarak koruyor. **Etkisi**: her 5 dakikalık canlı karar döngüsü VE her backtest çağrısı bu modeli disk'ten TEKRAR TEKRAR yüklüyor (`OnlineSignalModel.load_from()`, önbelleklenmiyor) — 208MB yerine artık ~40-50MB yükleneceğinden hem yükleme başına bellek darbesi hem de büyük ihtimalle "canlı API'nin belleği zamanla büyüyor" bulgusunun asıl kaynağı önemli ölçüde küçülüyor. **Önemli**: bu düzeltme yalnızca YENİ eğitimlerde geçerli — sunucudaki mevcut 208MB'lık dosya, bir sonraki `train-all.sh` (online adımı ATLANMADIĞI sürece varsayılan olarak çalışır) çalıştırmasında otomatik olarak küçük haliyle DEĞİŞTİRİLECEK; o ana kadar risk sürer. `docker build` GEREKTİRİR (kod değişti).
- [x] **Karlılık: Backtest artık çalışır durumda, gerçek veriyle ilk iki iyileştirme yapıldı** (OOM saga'sı çözülüp `POST /backtest/system/run` sorunsuz döndükten sonra kullanıcı isteği: "karlılığı artırmamız gerekiyor"):
  1. **İlk ölçüm** (BTCUSDT.P, 1h, 10.000 mum, holdout-only): 149 işlem, kazanma=%63.09, toplam PnL=+%2.68, aylık ort.=+%0.32, **maks. düşüş yalnızca %0.76** — sistem çok temkinli duruyor, aldığı riskin çok altında bir getiri elde ediyor. Bu, en güvenli/en büyük kaldıracın pozisyon boyutu olduğunu gösterdi.
  2. **Gerçek işlem listesinden somut bulgu — Kelly'nin "soğuk başlangıç" sorunu**: 149 işlemin ~25-30'u ($0.00 boyutla) HİÇ SERMAYE KULLANMADAN açılıp kapanmış — Ocak-Şubat 2026 penceresinde (ilk ~40 işlemlik gürültülü örneklem) kazanma oranı geçici olarak %37-44'e düşünce Kelly formülü matematiksel olarak "tam Kelly=%0" hesaplıyor (kazanç/kayıp oranı + düşük kazanma oranıyla edge negatif çıkıyor) — model daha sonra %63'e çıkan gerçek performansına RAĞMEN o dönem sermaye tamamen atıl kalmış. Bunu ölçmek için `app.backtest.system_runner.sweep_position_sizing` + `POST /backtest/system/sweep-position-sizing` + `deploy/sweep-position-sizing.sh` eklendi (`sweep_confidence_thresholds` ile AYNI felsefe/desen: `kelly_min_trades` × `kelly_multiplier` ızgarasında veri üretir, otomatik "en iyi"yi SEÇMEZ) — henüz çalıştırılıp karar verilmedi, sıradaki adım bu.
  3. **Güven eşiği YENİDEN ölçüldü, ters yönde çıktı — 0.55'ten 0.5/0.45'e GERİ DÖNÜLDÜ**: kullanıcı `bash deploy/sweep-confidence.sh`'ı (model/veri "temel sadeleşme" ve online model düzeltmesinden sonra kökten değiştiği için) yeniden çalıştırdı. Eski ölçüm 0.55'i önermişti (89 işlem, PnL=+%2.09) — YENİ ölçümde AYNI eşikte (0.55) yalnızca 31 işlem, kazanma=%45.16, PnL **NEGATİF** (-%0.888)! `0.50`'de ise 149 işlem (5× daha büyük örneklem), kazanma=%63.09, PnL=+%2.68 — taramanın en iyi sonucu; 0.6+'ta örneklem 3/0/0'a çöküyor. `DecisionEngine.__init__` VE `SystemBacktestRequest` varsayılanları şemanın ORİJİNAL değerine (`open_confidence=0.5`/`close_confidence=0.45`) geri alındı. Bu tersine dönüş kendi başına önemli bir ders: sabit bir eşik YOK, model/veri her köklü değiştiğinde (bu durumda: BTC-only + yeni etiketleme + online model boyut düzeltmesi) sweep'in YENİDEN çalıştırılması ŞART — periyodik yeniden ölçüm bir "nice-to-have" değil, önceki turdaki tavsiye artık AKTİF ZARAR ETTİRİYORDU.
  4. **Pozisyon boyutu taraması çalıştırıldı, Kelly varsayılanları güncellendi — hem backtest HEM canlı paper-trading'de**: `bash deploy/sweep-position-sizing.sh` sonucu net bir örüntü verdi — `kelly_min_trades`/`kelly_multiplier`'ı artırmak işlem sayısını/kazanma oranını HİÇ değiştirmiyor (sinyal aynı kalıyor) ama toplam PnL'i MONOTON büyütüyor: eski varsayılan (20/0.5) +%2.68 iken, 40/0.75 +%5.22 (~2×), 60/1.0 (tam Kelly) +%7.03 (en yüksek). Düşüş payı aralık boyunca küçük kaldı (%0.76 → %1.14). Tam Kelly'ye (1.0x) GEÇİLMEDİ — küçük örneklemde (149 işlem) tahmin hatasına karşı bir güvenlik payı bırakmak için `kelly_min_trades=40`/`kelly_multiplier=0.75` seçildi (`SystemBacktestRequest`). Ayrıca ÖNEMLİ bir keşif: canlı/paper-trading motoru (`app.portfolio.schemas.RiskRules`) VARSAYILAN OLARAK `"fixed_risk"` kullanıyordu — Kelly yalnızca backtest'in varsayılanıydı, yani bu iyileşme gerçek paper-trading sermayesine HİÇ yansımıyordu. Kullanıcı açıkça "paper trading'e de uygula" dedi — `RiskRules` artık `SystemBacktestRequest` ile SENKRON (`kelly`/`0.75`/`40`).
  5. **Kullanıcının nihai hedefi netleşti — "tam otomatik, öğrenme algoritmalarıyla optimizasyon yapan, güvenli günlük %1+ getiri hedefleyen otonom bot"**: bu hedefin matematiksel olarak imkansıza yakın olduğu AÇIKÇA belirtildi (günlük %1 bileşik = yılda ~%3700; dünyanın en iyi kantitatif fonu bile bunun onlarca katı altında kalır) — kullanıcı buna katılıp "şimdi sürece odaklanalım" dedi. Bunun üzerine kullanıcının kendi önerdiği 4 seviyeli adaptasyon planından (walk-forward → çevrimiçi adaptasyon → RL → meta-öğrenme) **Seviye 1'in KÜÇÜLTÜLMÜŞ/GÜVENLİ hâli** kuruldu:
     - Yeni `app.backtest.system_runner.run_periodic_optimization`: `sweep_confidence_thresholds` + `sweep_position_sizing`'i ART ARDA çalıştırıp GÜVENİLİR (`MIN_RELIABLE_TRADES_FOR_OPTIMIZATION=30` işlem ve üzeri — az işlemli ama cazip görünen noktaları, ör. eski turdaki 3/0/0 işlemlik "yüksek PnL" tuzaklarını, ASLA seçmez) en iyi kombinasyonu bulup MEVCUT (o an canlıda kullanılan) değerlerle karşılaştırmalı döner.
     - Yeni haftalık zamanlayıcı işi `job_periodic_optimization` (`Settings.ml_periodic_optimization_enabled`, varsayılan AÇIK, `ml_periodic_optimization_seconds` varsayılan 7 gün) bu sonucu yeni `optimization_runs` tablosuna (`GET /backtest/system/optimization-history`) KAYDEDER.
     - İlk turda **BİLEREK CANLI AYARLARI OTOMATİK DEĞİŞTİRMİYORDU** — yalnızca ölçüp raporluyordu (`applied` her zaman `False`). Gerekçe: tek bir sweep'in küçük örneklemli önerisini (bkz. yukarıdaki 3. madde — `open_confidence=0.55` bir turda kârlı, bir sonraki turda ZARARDI) otomatik uygulamak tehlikelidir.
  6. **Kullanıcı "şimdi kur" dedi — Seviye 2 (otomatik uygulama) KADEMELİ olarak eklendi**: `Settings.ml_periodic_optimization_auto_apply_enabled` (varsayılan AÇIK) açıkken, öneri iki güvenlik kapısından geçerse canlı ayarlara uygulanır — (1) yeterli örneklem (zaten `run_periodic_optimization`'ın kendi güvenilirlik filtresiyle sağlanır — filtrelenmişse `recommended == current` döner, otomatik no-op), (2) mevcut PnL'den en az `ml_periodic_optimization_min_improvement_pct` (varsayılan %0.1) kadar İYİ olmalı (gürültü farkına karşı). Geçse BİLE TAM SIÇRAMA yapılmaz — yalnızca mevcutla önerilen arasındaki mesafenin `ml_periodic_optimization_max_step_fraction`'ı (varsayılan %50) kadarı uygulanır (`_dampened_step`) — kullanıcının kendi Level 2 önerisindeki "öğrenme hızı düşük tutulmalı" uyarısını birebir karşılar: tek bir gürültülü haftanın parametreleri uçtan uca sıçratmaması, birkaç hafta boyunca kademeli yakınsaması için.
     - **Teknik keşif, düzeltildi**: güven eşikleri (`open_confidence`/`close_confidence`) canlı karar döngüsünde (`app.engine.service.run_cycle_once`) ÖNCEDEN hiçbir yerden okunmuyordu — `DecisionEngine` her 5 dakikada bir sıfırdan, yalnızca kendi SABİT Python varsayılanlarıyla (0.5/0.45) kuruluyordu; bir arka plan işinin bunları çalışma zamanında değiştirmesi için hiçbir mekanizma YOKTU. Yeni `Settings.live_open_confidence`/`live_close_confidence` eklendi, `run_cycle_once` artık bunları HER döngüde okuyup `DecisionEngine`'e açıkça geçiyor — `job_periodic_optimization` bu iki alanı güncelleyince BİR SONRAKİ 5 dakikalık döngüde hemen etkili oluyor (`docker build`/redeploy GEREKMEDEN). Kelly parametreleri için ayrı bir ayara gerek yoktu — `app.portfolio.shared.get_portfolio().rules` zaten çalışma zamanında mutlanabilen paylaşılan tek bir nesne, `job_periodic_optimization` bunu doğrudan günceller.
     - Her hafta `optimization_runs` tablosuna hem önerilen hem UYGULANAN (kademeli adımdan sonraki gerçek) değerler kaydedilir — `applied=True` durumunda status log'u yeni canlı değerleri de yazdırır. RL/meta-öğrenme (kullanıcının Seviye 3-4'ü) hâlâ ertelendi — hem veri (yıllarca saatlik bar gerekir, elimizde ~8 ay var) hem hesaplama (bu oturumda 3 kez OOM'a yol açan XGBoost+river'dan çok daha ağır) açısından şu an için uygun bulunmadı.
  7. **Deney: LSTM tekrar denendi, KENDİ eşiğini geçti ama ensemble'ı KÖTÜLEŞTİRDİ**: `bash deploy/train-all-with-lstm.sh` (`deploy/`e eklenen, `INCLUDE_LSTM=1`'i büyük/küçük harf sorunu olmadan tek dosya adıyla geçiren kısayol) ile LSTM yeniden eğitildi — bu sefer `oos_balanced_acc=0.411` (eşik 0.37) ile GEÇTİ, sistem otomatik olarak ensemble'a kattı. Ama TAM SİSTEM backtest'inde sonuç NET ŞEKİLDE kötüleşti: kazanma oranı %64,20 → %50,35, toplam PnL +%6,43 → +%2,45 (düşüş payı biraz iyileşti, %1,41→%0,94, ama getiri düşüşünü telafi etmiyor). **Ders**: bir modelin KENDİ izole doğruluğunun kalite eşiğini geçmesi, ensemble'a EKLENMESİNİN faydalı olacağını GARANTİ ETMEZ — LSTM'in tahminleri muhtemelen XGBoost/online ile yeterince ÇELİŞİYOR, üçünü birleştiren basit kural tabanlı `DecisionEngine._combine_predictions` bu çelişkiyi net bir sinyal yerine belirsizliğe çeviriyor. Gerçek ölçüt her zaman TAM SİSTEM backtest'i olmalı, tekil model metriği değil. LSTM elle devre dışı bırakıldı (`deploy/disable-lstm.sh` — `write_model_status` ile AYNI mekanizma, `enabled=False` yazar; model dosyası SİLİNMEZ, ileride tekrar denenebilir).
  8. **Kullanıcı isteği: "iyileştirmek için gereken tüm uygulamaları yap"** — üç yeni kol birden açıldı:
     - **Çok-sembollü eğitim GERİ AÇILDI** (`ml_train_max_symbols`: 1 → **5**): BTC-only'nin OOM krizini hafifletmek için geçici bir önlem olduğu, asıl OOM kök nedeninin (online model sınırsız büyümesi) o zamandan beri bulunup düzeltildiği göz önüne alınarak geri alındı. BTC-only'de ayda yalnızca ~18-20 işlem oluyordu — bu, modelin kalitesinden BAĞIMSIZ olarak mutlak getiriyi (bileşik büyüme fırsatı) doğrudan sınırlıyordu. `select_training_symbols`'un BTC + korelasyonlu ek sembol mantığı zaten koddaydı, yalnızca üst sınır geri açıldı.
     - **XGBoost hiperparametreleri artık AYARLANABİLİR + taranabilir**: şimdiye kadar `n_estimators`/`max_depth`/`learning_rate` hiç değiştirilmemiş, hep `_XGBClassifierWrapper`'ın sabit varsayılanlarıyla (300/4/0.05) çalışılmıştı. `SignalModel`/`train_signal_model_validated`'a yeni `xgb_params` parametresi eklendi (kaydedilmiş modelin `save()`/`load()` davranışını ETKİLEMEZ — fit edilmiş pipeline zaten olduğu gibi joblib'e yazılıyor). Yeni `app.backtest.system_runner.sweep_xgboost_hyperparameters` (+ `POST /backtest/system/sweep-xgboost-hyperparameters` + `deploy/sweep-xgboost-hyperparameters.sh`) her ızgara noktası için YENİ bir aday eğitip (`persist=False`) hem kendi izole `oos_balanced_accuracy`'sini HEM DE (LSTM dersini burada da uygulayarak) TAM SİSTEM backtest metriklerini (işlem/kazanma/PnL/drawdown) birlikte raporluyor — karar PnL/drawdown'a göre verilmeli, izole doğruluğa göre değil.
     - **Etiketleme hedefi (horizon × ATR çarpanı) artık sistematik taranabilir**: mevcut hedef (3 bar/1.0×ATR) yalnızca TEK bir sentetik saf-gürültü kontrolüyle seçilmişti, gerçek veride hiç taranmamıştı. Yeni `sweep_xgboost_labeling_targets` (+ `POST /backtest/system/sweep-labeling-targets` + `deploy/sweep-labeling-targets.sh`) — AYNI desen (yeni aday eğit, tam sistem backtest'i, otomatik "en iyi" SEÇME).
     - Üçü de "veri üret, karar verme" felsefesini (`sweep_confidence_thresholds`/`sweep_position_sizing` ile AYNI) izliyor. **Bilinen bir yaklaşıklık**: `restrict_to_holdout=True` iken backtest'in holdout sınırı hâlâ o an DİSKTE KAYITLI üretim modelinin kendi holdout başlangıcından okunuyor (adayın KENDİ holdout'undan değil) — semboller/lookback/etiketleme sabit tutulduğu sürece (yalnızca hiperparametreler değiştiğinde) bu pratikte neredeyse aynı sınırı verir, ama tam kesin değildir.
     - Sonuçlar henüz sunucuda ÇALIŞTIRILMADI/gözden geçirilmedi — sıradaki adım bu üç script'i sırayla çalıştırıp verilere bakmak.
  9. **Çok-sembollü eğitim DE test edildi, sonuç yine bir GERİLEME — `ml_train_max_symbols` 1'e GERİ ALINDI**: sunucuda gerçek eğitim + backtest ile ölçüldü — BTC + 1 (uyumluluk/korelasyon filtresinden geçen, ismi `docker build` yapılmadığı için görülemedi) sembolle: 101 işlem, kazanma=%46,53, PnL **+%1,81** — BTC-only zirvesinden (176 işlem, %64,20, +%6,43) NET bir düşüş. LSTM denemesiyle TAM AYNI ders: "daha fazla veri/çeşitlilik" teoride mantıklı ama BTC'ye özgü sinyali seyreltip pratikte zarar veriyor. Kullanıcı onayıyla `ml_train_max_symbols` **1**'e geri alındı (BTC-only, şu ana kadarki en iyi doğrulanmış konfigürasyon). Sıradaki adım: hiperparametre + etiketleme taramalarını BTC-only zemininde çalıştırmak.
  10. **XGBoost hiperparametre taraması BTC-only zeminde çalıştırıldı — yeni varsayılan `n_estimators=150`/`max_depth=3`/`learning_rate=0.1`**: `bash deploy/sweep-xgboost-hyperparameters.sh` sunucuda 27 kombinasyonu (n_estimators×max_depth×learning_rate ızgarası) gerçek veriyle tam sistem backtest'inden geçirdi. Eski varsayılan (300/4/0.05) satırı 57 işlem/%59,65 kazanma/+%3,22 PnL veriyordu — bu, sweep'in kendi yaklaşıklığı içinde (adaylar `persist=False` ile ayrı eğitildiği için) önceki "en iyi bilinen" üretim backtest'inden (176 işlem/%64,20/+%6,43) düşük çıktı; ikisi arasındaki fark not edildi ama sweep'in KENDİ İÇİNDEKİ göreli sıralaması yine de anlamlı kabul edildi. En yüksek PnL'li tek nokta (500/6/0.03, 48 işlem) İLK ÖNCE seçildi, ancak `python3 -m pytest` çalıştırılınca `test_calibration_does_not_collapse_direction_to_majority_class` regresyon testini (zayıf sinyal + ağır sınıf dengesizliği senaryosu, geçmişte üretimde modelin tek sınıfa çökmesine yol açan asıl kombinasyon) KIRDIĞI görüldü — `max_depth=6` azınlık recall'ını 0.35 eşiğinin (test) altına, 0.2925'e düşürdü. Bu, derin ağaçların küçük gerçek-veri örnekleminde aşırı uyuma yatkın olduğunun somut kanıtıydı; testi gevşetmek YERİNE tablodan daha sığ bir aday seçildi: **150/3/0.1** (67 işlem, %56,72 kazanma, +%5,46 PnL — benzer/daha büyük örneklem, `max_depth` eskisinden bile SIĞ) hem bu regresyon testini geçiyor hem gerçek backtest'te iyileşme gösteriyor. `_XGBClassifierWrapper` varsayılanı buna güncellendi, tam test suite'i (378 test, 1 pre-existing skip) yeşil. Sıradaki adım: `bash deploy/sweep-labeling-targets.sh` ile etiketleme hedefini (horizon × ATR çarpanı) de taramak, sonra gerçek `train-all.sh` ile bu yeni varsayılanların TAM üretim backtest'inde doğrulanması.
  11. **Etiketleme hedefi taraması BTC-only zeminde çalıştırıldı — yeni varsayılan `horizon=8`** (`_ENSEMBLE_LABELING`, ATR çarpanı 1.0'da sabit kaldı): `bash deploy/sweep-labeling-targets.sh` sunucuda 20 kombinasyonu (5 horizon × 4 ATR çarpanı) gerçek veriyle tam sistem backtest'inden geçirdi. Eski varsayılan (horizon=3/ATR=1.0) 65 işlem/%55,38 kazanma/+%1,27 PnL veriyordu — tablonun alt sıralarında. `horizon=8/ATR=1.0` en yüksek PnL'i verdi (145 işlem, %55,17 kazanma, +%5,12 PnL, %1,46 düşüş) — büyük örneklem + güçlü PnL artışı, aşırı uyum riskini işaret eden bir uyarı (LSTM/max_depth derslerindeki gibi) yoktu. `_ENSEMBLE_LABELING["horizon"]` 3'ten 8'e güncellendi — TÜM ensemble üyeleri (XGBoost/meta-label/LSTM/online/rejim) bu tanımı PAYLAŞTIĞI için değişiklik hepsine birden yansıyor. Sıradaki adım: gerçek `train-all.sh` ile hem bu değişikliğin HEM önceki hiperparametre değişikliğinin (150/3/0.1) birlikte TAM üretim backtest'inde doğrulanması.
