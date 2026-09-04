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
- [ ] Ağır eğitim işlerinin (LSTM/XGBoost) ayrı bir process'te (subprocess/worker) çalıştırılması — şu an aynı uzun ömürlü `uvicorn` process'i içinde çalışıyor; art arda birkaç ağır eğitim isteği, Python/PyTorch bellek ayırıcısının belleği işletim sistemine tam geri vermemesi nedeniyle kümülatif bir bellek artışına ve OOM'a yol açabiliyor (bkz. BTC-only lookback sweep testinde 10K/20K başarılı, 30K'de tekrar OOM). Geçici çözüm: ağır sweep'lerde adımlar arasında backend'i yeniden başlatmak (`recreate-backend.sh`)
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
- [x] `POST /ml/train-all` (`deploy/train-all.sh`) — tüm modelleri (XGBoost → meta-label → LSTM → online → regime) TEK çağrıda, deploy script'lerinde doğrulanmış AYNI parametrelerle sırayla eğitir. Kullanıcının 5 ayrı script'i elle çalıştırması yerine — özellikle ilk kurulumda (`fourkeys_ml_artifacts` volume'ü boşken). Her adım bağımsız try/except ile sarılı: biri başarısız olursa diğerleri yine de çalışır, her adımın sonucu (`ok`/`detail`) ayrı raporlanır
- [x] Sistem backtest'i (`POST /backtest/system/run`, `GET /backtest/system/latest`, frontend "Backtest" sayfası) — DCA/JSON-strateji motorlarından FARKLI: canlı karar motorunun kullandığı AYNI eğitilmiş model (+ varsa meta-label filtresi) BTCUSDT.P futures üzerinde varsayılan olarak 10.000 saatlik mumu (`ml_train_timeframe`/`ml_train_lookback` ile aynı) bar-bar tekrar oynatır. Talep üzerine (on-demand) çalışır, otomatik/periyodik değildir. Bilinen basitleştirmeler: makro/order-book/taker-flow özellikleri hesaplanmaz (yalnızca anlık değerleri var — eğitim setindeki eski barlarla AYNI şekilde nötr kabul edilir, yeni bir yanlılık değil); LSTM/online ensemble dahil değil; kademeli alım/Kelly/VIX filtresi yok (her sinyalde tüm equity ile tek giriş/çıkış) — hepsi yanıtın `warnings` alanında da raporlanır. Sonuç `backtest_runs`/`backtest_trades` tablolarına kaydedilir; kullanılan OHLCV geçmişi `ohlcv_raw`'a backfill edilir. Grafana'da yeni "4keys — Sistem Backtest" dashboard'u (Postgres veri kaynağı): BTCUSDT.P 1h candlestick paneli + al/sat noktaları (annotation) + kümülatif equity grafiği + son çalıştırma özeti
- [x] Stop-loss zorunlu uygulama (canlı karar motoru): `DecisionEngine._open()` içinde `stop_loss_price` önceden yalnızca Kelly boyutlandırma hesabında kullanılıp atılıyordu — pozisyon üzerinde HİÇBİR ZAMAN saklanmıyor/kontrol edilmiyordu, yani model sinyalini değiştirmediği sürece fiyat o seviyeyi geçse bile pozisyon asla kapanmıyordu. Artık `PortfolioPosition.stop_loss_price` alanına kaydediliyor ve her döngüde `stop_loss_breached()` ile kontrol edilip aşılırsa modelin güncel sinyalinden BAĞIMSIZ olarak zorla kapatılıyor (`RiskRules.stop_loss_enabled`, varsayılan açık)
- [x] İşlem maliyeti (komisyon + kayma) simülasyonu — önceden sistemde HİÇBİR YERDE modellenmiyordu (canlı `PortfolioManager`, JSON-strateji `/backtest` motoru, DCA simülatörü hepsi PnL'i saf fiyat farkından hesaplıyordu; DCA simülatörünün kendi docstring'i bunu açıkça bir basitleştirme olarak belirtiyordu). Artık üç motorda da `commission_pct`/`slippage_pct` (varsayılan Binance Futures taker ücreti ~%0.04 + mütevazı kayma tahmini ~%0.02) round-trip olarak gerçekleşen PnL'den düşülüyor: `PortfolioManager.close_tranche()` (giriş+çıkış = 2 bacak), `app.strategy.engine.run_backtest()` (giriş+çıkış = 2 bacak, bariyer kontrolleri BRÜT fiyatla yapılır — gerçek emir de bu seviyelere brüt hareketle ulaşır), `app.dca.simulator.simulate_dca()` (base order + her dolan safety order + 1 çıkış = değişken sayıda bacak, DCA'nın çok-bacaklı doğasına uygun)
