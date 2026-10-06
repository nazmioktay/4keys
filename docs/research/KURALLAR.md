# Araştırma kuralları (sonraki TÜM oturumlar önce bunu okur)

Amaç güzel bir backtest değil, canlıda maliyetler sonrası ayakta kalan bir getiri kaynağı bulmak.
Şüpheci ol: çok iyi görünen her sonucu önce bir hata veya sızıntı olarak varsay.
Bu belgedeki kabul eşikleri ve yasaklar **önceden kayıtlıdır**; sonuç görüldükten sonra DEĞİŞTİRİLEMEZ.

Kod: `backend/research/`. Sonuçlar: `research/results/<deney_id>/`. Özet günlük: `docs/research/deneyler.md`.
Canlı/paper koduna (`backend/app/**`) araştırma amacıyla dokunulmaz.

## 1. Nihai test penceresi
- **2025-10-01 ve sonrası.** Hiçbir deney, parametre seçimi ya da grafik bu dönemi kullanamaz.
  Yalnızca Aşama 5'te BİR KEZ açılır.
- Kodla zorlanır (`research.config.FINAL_TEST_START`, `research.data.store`):
  veri yükleyici varsayılan olarak bu tarihten sonrasını KESER. Açmak için
  `allow_final_test=True` + `final_test_experiment_id=<id>` gerekir ve `docs/research/deneyler.md`
  içinde `FINAL-TEST-ACILDI <id>` satırı bulunmalıdır; yoksa `FinalTestError` fırlatılır.
  Motor (`research.engine`) ve rapor da girişte tarih kontrolü yapar (savunma derinliği).

## 2. Değerlendirme birimi
Portföyün **günlük net getiri serisi**. Sharpe / Sortino / Calmar günlük seriden yıllıklaştırılır (365 gün).

## 3. Maliyet varsayılanları (parametre; `research/config.py`)
| Kalem | Değer | Doğrulama |
|---|---|---|
| USDT-M futures taker | %0,05 | Binance destek sayfası (Regular user örneği, 2026-10): maker %0,02 / taker %0,05. Resmi ücret tablosu sayfası dinamik yüklendiği için sayısal teyit edilemedi; hesabınızdaki gerçek kademeyle teyit edin. |
| USDT-M futures maker | %0,02 | aynı |
| Spot (maker = taker) | %0,10 | binance.com/en/fee/schedule, Regular user: 0,100% / 0,100% (2026-10) |
| Kayma | BTC/ETH 2 bps; ilk 20 sembol 5 bps; diğerleri 15 bps | Kademe, noktasal-zamanlı hacim sıralamasından (`research.data.universe`) |
| Funding | gerçek geçmiş oranlar, sembolün kendi funding aralığıyla | `fundingRate` arşivi; long öder, short alır |

Ücret ×2 ve kayma ×3 duyarlılık testi zorunludur (bkz. kabul eşikleri).

## 4. Yürütme
Sinyal günlük kapanışta (00:00 UTC) üretilir; dolum **sonraki açılış ± kayma**. +1 bar gecikme
duyarlılık testi zorunludur (`engine.run(..., delay_bars=1)`).

## 5. Kabul eşikleri (önceden kayıtlı, sonradan DEĞİŞTİRİLEMEZ)
- Net Sharpe (walk-forward): **kol ≥ 0,8; portföy ≥ 1,0.**
- Deflated Sharpe ≥ **0,95** (o ana kadarki TOPLAM deneme sayısıyla).
- PBO ≤ **0,25**.
- Maks. drawdown: **kol ≤ %35, portföy ≤ %30.**
- Calmar ≥ **max(BTC al-tut Calmar'ı, 0,7).**
- Takvim yıllarının en az **%60'ı pozitif.**
- Ücret ×2 ve kayma ×3 altında Sharpe ≥ **0,5.**
- Plato: komşu parametrelerde Sharpe ≥ seçilenin **%70'i.**

`research.stats.check_acceptance` bu eşikleri koda döker; eşikler `research/config.py::ACCEPTANCE`
içinde tek yerde durur ve bu belgeyle birebir aynı olmalıdır.

## 6. Deney kaydı
- Her çalıştırma `docs/research/deneyler.md`'ye (özet) ve `research/results/<deney_id>/` altına
  (`config.yaml`, `metrics.json`, günlük getiri parquet) yazılır.
- Veri anlık görüntüsünün hash'i ve git commit'i de kaydedilir.
- Deflated Sharpe için GLOBAL bir deneme sayacı tutulur (`research/results/_registry.json`):
  her varyant bir denemedir.
- `python -m research.smoke` yalnızca altyapı kontrolüdür; deney sayılmaz, sayaç artmaz, kayıt yazmaz.

## 7. Yasaklar
- Nihai pencereye bakmak.
- Sonuç görüldükten sonra eşik değiştirmek.
- Kaydedilmeyen deneme yapmak.
- Tek "en iyi" parametreyi seçmek; **plato merkezi** seçilir.

## 8. Kod yolları (özet)
`research/data/` veri (indirme, parquet, nihai pencere kilidi, noktasal-zamanlı evren, kalite raporu, sembol limitleri) ·
`research/engine.py` vektörel portföy motoru · `research/stats.py` istatistik araç kutusu ·
`research/registry.py` deney kaydı · `research/report.py` rapor · `research/smoke.py` altyapı kontrolü.
