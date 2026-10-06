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

## 9. Deney bütçesi, ileriye dönük veri ve geçersizlik (Aşama 1 çerçevesi)
- **Deneme bütçesi:** soru (`question`, ör. "vol_tahmini") başına en fazla **40 config** (ablasyonlar dahil). Bütçe dolunca yeni deneme
  YAPILMAZ; mevcut sonuçlardan karar verilir (`research.budget`, `BudgetExceededError`). `smoke` deneyler sayılmaz.
- **forward_only veri:** geçmişi olmayan kaynaklar (tasfiye akışı, emir defteri derinliği, ayrıntılı açık pozisyon) **12 aylık veri
  birikmeden** hiçbir deneyde kullanılamaz (`research.panel.check_forward_only`).
- **Sızıntı testleri:** karıştırılmış hedef, `available_at` denetimi veya tekrarlanabilirlik testinden biri başarısızsa deney **GEÇERSİZ**
  sayılır (yine kaydedilir, sayaç artar, karar "GEÇERSİZ: ..."); +1 gün gecikme testi yalnızca UYARIDIR (kalıcı hedeflerde güçsüzdür).
- **Eşikler yine önceden kayıtlıdır (§5);** çerçeve yalnızca ölçer, sonuç görüldükten sonra değiştirilemez.
- **Smoke kaçağı kapatıldı:** `smoke` koşuları deneme sayacına sayılmaz AMA soru başına en fazla **10** kez koşturulabilir (`smoke_runs`,
  `_registry.json`); çıktıları "karar için KULLANILAMAZ" damgalıdır. Kayıtsız ablasyon yalnızca `smoke=True` ile yapılabilir
  (varsayılan: her ablasyon kayıtlı bir config). Bütçe, kayıt anında **kilit altında** zorlanır (eşzamanlı koşular 40'ı aşamaz).
- **`question` normalize edilir** (`strip` + küçük harf; yalnızca `a-z0-9_.-`): "Vol_Tahmini " ile "vol_tahmini" aynı bütçedir.
- **`final_test` kayıt satırı tam eşleşir:** `FINAL-TEST-ACILDI e1` satırı `e10`'u AÇMAZ.
- **AÇIK KARAR (kullanıcıya):** `forward_only` veri nihai test penceresinden (>= 2025-10-01) SONRA birikir, oysa deney dönemi
  (`period.end`) bu pencereye giremez. Sonuç: bu kaynaklar mevcut kurallarla hiçbir deneyde kullanılamaz — koşucu bunu sessizce boş
  özellik üretmek yerine `ForwardOnlyNoOverlapError` ile REDDEDER. Seçenekler: (a) forward_only verileri yalnızca nihai pencere açıldığında
  (Aşama 5) kullanmak; (b) KURALLAR'a ayrı, önceden kayıtlı bir "ileriye dönük doğrulama" penceresi eklemek (ör. toplama başlangıcından itibaren
  ilk 12 ay yalnızca keşif, sonrası kilitli test). Nihai pencere kilidini bu karar verilmeden gevşetmiyoruz.
- **Kod sınırı notu:** araştırma kodu `app/`'e yalnızca iki yerden bağlanır: `app/forwardcollect` (toplayıcı, opt-in) ve
  `research/sources/forward.py` (onun tablolarını `app.db.repository` üzerinden okur). Başka araştırma modülü `app/`'e dokunmaz.

