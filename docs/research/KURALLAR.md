# Araştırma kuralları (sonraki TÜM oturumlar önce bunu okur)

Amaç güzel bir backtest değil, canlıda maliyetler sonrası ayakta kalan bir getiri kaynağı bulmak.
Şüpheci ol: çok iyi görünen her sonucu önce bir hata veya sızıntı olarak varsay.
Bu belgedeki kabul eşikleri ve yasaklar **önceden kayıtlıdır**; sonuç görüldükten sonra DEĞİŞTİRİLEMEZ.

Kod: `backend/research/`. Sonuçlar: `research/results/<deney_id>/`. Özet günlük: `docs/research/deneyler.md`.
Canlı/paper koduna (`backend/app/**`) araştırma amacıyla dokunulmaz.

## 1. Nihai test penceresi
- **2025-10-01 ve sonrası** (2026-10-07 itibarıyla üst sınırı **2026-11-01**; o tarihten sonrası §10'daki ayrı ileriye dönük
  bölgedir). Hiçbir deney, parametre seçimi ya da grafik bu dönemi kullanamaz. Yalnızca Aşama 5'te BİR KEZ açılır; açıldığında da
  2026-11-01 ve sonrasını GÖSTERMEZ.
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
| Hesap (birincil) | **250 USDT, 3x kaldıraç, en yakın lot** (kullanıcının gerçek hesabı, 2026-10-07) | `research.config.ACCOUNT_NAV / ACCOUNT_LEVERAGE / ACCOUNT_LOT_ROUNDING`; min notional ve adım Binance'in GÜNCEL limitleri (`research.data.limits`, önbellekli). Koşucuda config `account: {nav, leverage, lot_rounding, limits}` ile değiştirilebilir; birincil sonuç bu hesapla raporlanır. |

**Küçük hesap ve lot kuralı (2026-10-07):**
- Min notional: sembollerin çoğunda 5 USDT, ETHUSDT'de 20, BTCUSDT'de 50 USDT. Miktar adımı da bağlayıcı: BTCUSDT'de 0,001 BTC,
  yani BTC 100 bin USDT iken en küçük lot ~100 USDT (250 USDT NAV'ın %40'ı).
- **BTC evrenden çıkarılmaz; tavanı en küçük lota göre ayarlanır** (kullanıcı kararı). Varlık başına tavan kullanan tasarımlar
  `research.sizing.asset_caps` kullanır: tavan_t = max(%20, bir lotun notional'ı_t / NAV). Yalnızca t günündeki fiyat kullanılır.
  Örnek: 114 bin USDT'de BTC tavanı %45,6 (bir lot). Böylece tavana dayanan BTC sinyali her fiyatta en az bir lot açar.
  `asset_caps` sinyal adaptörlerine OTOMATİK bağlı değildir: varlık başına tavan kullanan her tasarım onu açıkça kullanmalı ve
  ön kayıtta belirtmelidir; aksi halde düz %20 tavanla BTC (114 bin USDT'de) açılamaz.
  Sınır: tavan, deneyin BAŞLANGIÇ NAV'ına göre hesaplanır. NAV başlangıcın yarısının altına düşerse tavandaki hedef yarım lotun
  altında kalabilir ve BTC açılmaz ("min_lot").
- **En yakın lota yuvarlama** (`engine.run(lot_rounding="nearest")`):
  - Miktar en yakın lota yuvarlanır.
  - Hedef, en küçük lotun (max(min notional, 1 adım)) yarısından küçükse sonuç 0 lot olur: yeni pozisyon açılmaz ("min_lot"), mevcut pozisyon (yönü ne olursa olsun) KAPATILIR. Tam kapanış her zaman yapılabilir varsayılır.
  - Hedef yarım lottan büyükse en az bir lot açılır.
- **Yuvarlama riski büyütebilir:** hedefi lotun 0,5–1 katı olan pozisyon 1 lota, yani hedefin 2 katına kadar çıkar. Vol hedefi bu sembollerde aşılabilir. Her varyantta `lot_scale` raporlanır: GERÇEKLEŞEN emirlerde Σ|yuvarlanmış| / Σ|hedef| (atlanan emirler hariç); >1 ise yuvarlama pozisyonları büyütmüştür.
- **Kaldıraç tavanı 3x:**
  - Hedeflerin brütü NAV'ın 3 katını aşarsa önce TÜM hedefler orantılı küçültülür; bu adımda hangi sembolün açılacağı sütun sırasına
    bağlı kalmaz. Ancak lot yuvarlaması brütü yeniden tavanın üstüne çıkarırsa, tavana dayanan artışlarda sütun sırası belirleyici
    olabilir (sonraki artış tamamen atlanır). Mevcut adaptörlerin brüt tavanı 1,0–1,5x olduğu için pratik etkisi düşüktür.
  - "nearest" modunda önce küçültme ve kapanış emirleri, sonra artışlar işlenir (borsadaki sıra). Yuvarlama sonrası brütü ARTIRAN bir emir tavanı aşacaksa TAMAMEN atlanır ("leverage"; tavana sığan daha küçük bir lot denenmez). Küçültme emirleri engellenmez.
  - 250 USDT'de 750 USDT'ye kadar pozisyon taşınabilir. Sinyal adaptörlerinin kendi brüt tavanları ayrıca geçerlidir.
- **1x öneri:** her deney raporu "1x için önerilen asgari hesap"ı yazar: evrendeki her sembolün en küçük lotunun %20 referans tavana sığdığı NAV (`runner.recommended_nav_1x`).
  - Fiyat, deney penceresinin son günününkidir ve tarihi raporlanır; limitler ise güncel.
  - Örnek: BTC 114 bin USDT'de 0,001 × 114.000 / 0,20 ≈ **570 USDT**.
  - 1x ile çalışılacaksa başlangıçta hesabın en az bu tutara çıkarılması önerilir (BTC fiyatıyla değişir).
- Raporda atlanan emir denemeleri (nedenleriyle: min_lot, leverage, price) ve limiti bilinmeyen (delist) semboller gösterilir.
- Limitler okunamazsa (ağ/borsa hatası) koşu DURUR (`LimitsUnavailableError`); hata önbelleğe yazılmaz. Yalnızca borsada olmayan (delist) semboller kısıtsız varsayılır.
- BTC al-tut kıyası bilinçli olarak hesap kısıtsızdır (piyasa ölçütü).
- trend_001 bu kuraldan ÖNCE koştu (NAV 10.000, aşağı yuvarlama, kaldıraç tavanı yok); sonuçları o varsayımlarla geçerlidir.

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
- Deflated Sharpe için GLOBAL bir deneme sayacı tutulur (`research/results/_registry.json`, git'te izlenir):
  her varyant bir denemedir. Deney koşturduktan sonra sayaç değişikliği deneyle birlikte commit edilir (unutulursa sayaç
  geriler). Deneyler tek makinede koşturulur; git çakışmasında ASLA bir taraf seçilmez: `experiments` listelerinin
  birleşimi alınır, `total_trials` birleşimdeki varyant sayısına göre yeniden hesaplanır (aksi halde Deflated Sharpe
  iyimser olur).
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
- **Deneme bütçesi:** soru (`question`, ör. "vol_tahmini") başına en fazla **40 VARYANT**: deneme sayacına giren her varyant
  bütçeden düşer (ablasyonlar dahil; 2026-10-07'de "config" yerine varyant olarak netleştirildi). Tek bir deney de bütçeyi aşamaz
  (ör. 216 varyantlık bir ızgara kayda alınamaz). Bütçe dolunca yeni deneme YAPILMAZ; mevcut sonuçlardan karar verilir
  (`research.budget`, `BudgetExceededError`). `smoke` deneyler sayılmaz. *Geriye dönük:* `trend` sorusu trend_001 ile 248/40 varyant
  kullandı (kural netleşmeden önce); bu soruda yeni deneme yapılamaz, bu da trend_001'in "DUR" kararıyla tutarlı.
- **forward_only veri:** geçmişi olmayan kaynaklar (tasfiye akışı, emir defteri derinliği, ayrıntılı açık pozisyon) **12 aylık veri
  birikmeden** hiçbir deneyde kullanılamaz (`research.panel.check_forward_only`).
- **Sızıntı testleri:** karıştırılmış hedef, `available_at` denetimi veya tekrarlanabilirlik testinden biri başarısızsa deney **GEÇERSİZ**
  sayılır (yine kaydedilir, sayaç artar, karar "GEÇERSİZ: ..."); +1 gün gecikme testi yalnızca UYARIDIR (kalıcı hedeflerde güçsüzdür).
- **Eşikler yine önceden kayıtlıdır (§5);** çerçeve yalnızca ölçer, sonuç görüldükten sonra değiştirilemez.
- **Smoke kaçağı kapatıldı:** `smoke` koşuları deneme sayacına sayılmaz AMA soru başına en fazla **10** kez koşturulabilir (`smoke_runs`,
  `_registry.json`); çıktıları "karar için KULLANILAMAZ" damgalıdır. Kayıtsız ablasyon yalnızca `smoke=True` ile yapılabilir
  (varsayılan: her ablasyon kayıtlı bir varyant). Bütçe, kayıt anında **kilit altında** zorlanır (eşzamanlı koşular 40'ı aşamaz).
- **`question` normalize edilir** (`strip` + küçük harf; yalnızca `a-z0-9_.-`): "Vol_Tahmini " ile "vol_tahmini" aynı bütçedir.
- **`final_test` kayıt satırı tam eşleşir:** `FINAL-TEST-ACILDI e1` satırı `e10`'u AÇMAZ.
- **forward_only ile nihai pencere çakışması — KARAR VERİLDİ (2026-10-07, seçenek b):** forward_only veri nihai pencereden sonra
  biriktiği için ayrı, önceden kayıtlı bir ileriye dönük doğrulama bölgesi tanımlandı. Ayrıntılar §10'da.
- **Kod sınırı notu:** araştırma kodu `app/`'e yalnızca iki yerden bağlanır: `app/forwardcollect` (toplayıcı, opt-in) ve
  `research/sources/forward.py` (onun tablolarını `app.db.repository` üzerinden okur). Başka araştırma modülü `app/`'e dokunmaz.

## 10. İleriye dönük doğrulama bölgesi (forward_only veri) — 2026-10-07'de önceden kayıtlı
Bu bölüm, forward_only verinin (tasfiye akışı, emir defteri derinlik bantları, ayrıntılı açık pozisyon) hiçbir sonuç görülmeden
ÖNCE kayda geçirilmiş kullanım kuralıdır; §5 gibi sonuç görüldükten sonra DEĞİŞTİRİLEMEZ.

Takvim (`research/config.py` ile birebir aynı):
| Dönem | Aralık | Kullanım |
|---|---|---|
| Ana araştırma | < 2025-10-01 | tüm deneyler (`zone: main`, varsayılan) |
| Nihai test penceresi | [2025-10-01, 2026-11-01) | kilitli; yalnızca Aşama 5'te bir kez (§1) |
| Forward keşif (T0 = 2026-11-01) | [2026-11-01, 2027-11-01) | yalnızca `zone: forward` deneyler |
| Forward test | 2027-11-01 ve sonrası | kilitli; en erken **2028-05-01**'de bir kez açılır (>= 6 ay test verisi) |

Kurallar:
- **Bölge seçimi:** her deney config'te `zone: main | forward` taşır. forward_only kaynak YALNIZCA `zone: forward` deneyde kullanılabilir;
  `zone: forward` YALNIZCA en az bir forward_only kaynak içeren deneyde kullanılabilir. Ana araştırma (forward_only kaynak içermeyen)
  forward bölgesini ek veri olarak KULLANAMAZ (`ZoneMismatchError`).
- **Sıkı kesim:** forward bölgesindeki deney YALNIZCA [T0, forward test başlangıcı) verisini görür — fiyat, funding, türev metrikleri,
  makro ve duygu dahil. T0 öncesi (nihai pencere) özellik ısınması için bile kullanılamaz; uzun pencereli özellikler keşif döneminin
  başında NaN kalır (ısınma keşif döneminden yenir). Ana bölge forward bölgesini, forward bölgesi nihai pencereyi göremez.
  Bu kesim forward_only DB kaynaklarına da uygulanır (T0 öncesi toplanan satırlar kullanılmaz).
- **Etkin keşif süresi (bilinen sonuç):** noktasal-zamanlı evren bir sembolü en az 90 günlük geçmişten sonra alır; forward bölgesinde
  geçmiş T0'da başladığı için evren ~2027-01-30'a kadar boştur. Etkin keşif ~[2027-01-30, 2027-11-01) (~9 ay) olur; `cv.min_train_days`
  buna göre küçültülmelidir (koşucu, dönemden uzun `min_train_days`'i reddeder). Bu, sıkı kesimin bilinçli bedelidir.
- **12 ay kuralı (§9) sürer:** forward_only kaynak 365 gün birikmeden deney koşmaz. T0 ile toplama başlangıcı aynı değilse T0
  KAYDIRILMAZ (önceden kayıtlı); keşif dönemi daha az veriyle yapılır.
- **Forward testi açmak:** `allow_final_test=True` + `final_test_experiment_id=<id>` + `deneyler.md` içinde tam eşleşen
  `FORWARD-TEST-ACILDI <id>` satırı + bugün >= 2028-05-01. Bir kez açılır; açılınca [T0, ..) görünür. Kilit (`research.guard`) bunu
  zorlar; ancak nihai testte olduğu gibi bu açılışı uçtan uca yapan bir koşucu HENÜZ YOK (deney koşucusu her zaman kilitli modda
  çalışır) — Aşama 5 / forward test koşucusu ayrıca yazılacak.
- **Bütçe ve sayaç:** forward deneyleri de §9 bütçesine (soru başına 40 config) ve global deneme sayacına (Deflated Sharpe) tabidir.
- **Kabul eşikleri** §5'tekiyle aynıdır.
- **Uygulama:** `research.guard` (bölge bağlamı `guard.zone(...)`, `zone_bounds`, `cut_final_test`, `assert_no_final_test`); koşucu ve
  ablasyon deneyi config'teki bölgede çalıştırır; yükleyiciler, motor, rapor ve kaynaklar etkin bölgenin sınırına uyar.
