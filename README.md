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

### Çalıştırma

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

| Endpoint | Açıklama |
|---|---|
| `GET /screener/top?direction=long\|short&limit=10` | Long/Short Top N tarama sonucu |
| `POST /ml/train` | Modeli eğitir (body: `{"symbols": [...], "horizon": 5, "threshold_pct": 1.0}`, `symbols` boş bırakılırsa screener top listesi kullanılır) |
| `GET /ml/predict?symbol=BTC/USDT:USDT` | Tek sembol için yön + güven tahmini |
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

## Yol haritası

- [x] Screener (Binance, teknik skor)
- [x] ML sinyal modülü (MLP tabanlı yön tahmini)
- [x] Otomatik açma/kapama karar motoru (paper-trading)
- [x] DCA optimizasyon hesaplayıcısı
- [x] JSON tabanlı strateji tanımlama motoru (TradingView'sız)
- [ ] Screener + motorları periyodik/zamanlanmış bir job'a bağlama
- [ ] Gerçek borsa emir yürütme katmanı (API anahtarı + açık onay gerektirir)
- [ ] BIST/VIOP adapter'ları
- [ ] Portföy ve risk yönetimi kuralları
