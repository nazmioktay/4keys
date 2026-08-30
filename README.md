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
- **Veri/indikatörler:** `pandas` + `pandas-ta` tabanlı teknik gösterge hesaplama.

## Durum

Şu an yalnızca **Screener (Modül 1)** iskeleti mevcut: Binance USDT-M vadeli
paritelerinde RSI, EMA trend, MACD ve hacim momentumunu birleştiren bir skor ile
Long/Short Top 10 listesi üretiyor.

### Çalıştırma

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

`GET /screener/top?direction=long&limit=10` — Long yönünde en güçlü 10 pariteyi döner.
`GET /screener/top?direction=short&limit=10` — Short yönünde en güçlü 10 pariteyi döner.

## Yol haritası

- [x] Screener iskeleti (Binance, teknik skor)
- [ ] Screener'ı gerçek zamanlı/periyodik tarama job'una bağlama
- [ ] ML sinyal modülü (screener çıktısını feature olarak kullanan model)
- [ ] Otomatik işlem açma/kapama motoru (borsa emir katmanı)
- [ ] DCA optimizasyon hesaplayıcısı
- [ ] JSON tabanlı strateji tanımlama motoru
- [ ] BIST/VIOP adapter'ları
- [ ] Portföy ve risk yönetimi kuralları
