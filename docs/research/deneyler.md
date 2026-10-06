# Deney günlüğü

Kurallar için bkz. `KURALLAR.md`. Her çalıştırma `research.registry.register_experiment` ile buraya tek satır ekler.
Toplam deneme sayacının tek doğruluk kaynağı `research/results/_registry.json`'dır (Deflated Sharpe bunu kullanır).

**Toplam deneme: 0**

| Tarih (UTC) | Deney | Hipotez | Varyant | Net Sharpe | Maks. DD | Karar | Veri hash | Commit |
|---|---|---|---|---|---|---|---|---|

<!-- Nihai pencere açma kayıtları (yalnızca Aşama 5): "FINAL-TEST-ACILDI <deney_id>" -->
<!-- Forward test açma kayıtları (en erken 2028-05-01; KURALLAR.md §10): "FORWARD-TEST-ACILDI <deney_id>" -->

## Kural kayıtları
- 2026-10-07 — KURALLAR.md §10 (ileriye dönük doğrulama bölgesi) önceden kayda geçirildi: nihai pencere [2025-10-01, 2026-11-01);
  forward keşif [2026-11-01, 2027-11-01); forward test 2027-11-01+, en erken 2028-05-01'de açılır. Bu tarihte hiçbir forward_only
  veri toplanmamıştı ve hiçbir deney nihai pencereye bakmamıştı.

## ÖN KAYIT — `trend` (2026-10-07, sonuç görülmeden önce; dal `research/trend`)
**Hipotez:** Kripto varlıklarda zaman serisi momentumu, çok varlıklı ve volatiliteye göre boyutlandırılmış olarak, maliyetler sonrası
BTC al-tut'tan daha yüksek Calmar üretir.

**Veri ve pencereler.** USDⓈ-M perpetual günlük OHLCV (`um_1d`), yerel veri 2020-01-01'de başlıyor. Evren bir sembolü 90 gün geçmişten
sonra alır ve en uzun sinyal 250 gün ister. Bu yüzden:
- Pozisyon üretimi 2020-10-01'de başlar (ısınma).
- **Değerlendirme penceresi 2021-04-01 → 2025-09-30.** Tüm metrikler, PBO ve kabul kararı bu pencerede hesaplanır. Nihai pencere (>= 2025-10-01) kullanılmaz.
- **2020-03 stres dönemi DEĞERLENDİRİLEMEZ:** o tarihte evren boş ve 250 g sinyal yok. Raporda yalnızca BTC al-tut için gösterilir.

**Sinyaller** (gün t kapanışında hesaplanır, t+1 açılışında dolar; değer aralığı [-1, 1]):
- a) `a_L = sign(close_t / close_{t-L} - 1)`, L ∈ {20, 60, 120, 250}.
- b) `b_L = clip(r_L / (σ · √L), -2, 2) / 2`. Burada r_L, L günlük log getiri; σ, 60 günlük EWMA (span 60) günlük vol. L ∈ {20, 60, 120, 250}.
- c) `c_mix` = (8,32), (16,64), (32,128) çiftleri için `sign(EMA_f - EMA_s)` değerlerinin ortalaması.
- d) Donchian `d_20_10` ve `d_55_20`:
  - Giriş: kapanış, önceki N_giriş günün en yüksek değerinin üstündeyse long, en düşük değerinin altındaysa short.
  - Çıkış: long, önceki N_çıkış günün en düşük değerinin altında kapanırsa kapanır; short simetrik.
- **ANA ADAY `main`:** a'nın 4 hızı ve c'nin 3 çiftinin (7 bileşen) eşit ağırlıklı ortalaması. Geçmişi yetmeyen bileşen ortalamaya girmez.
- Toplam 12 sinyal: a×4, b×4, c_mix, d×2, main.

**Yön.**
- Long/flat (LF): `max(s, 0)`. Long/short (LS): `s`.
- LS'de short bacağın net katkısı ayrıca raporlanır.

**Boyutlandırma.**
- Ham ağırlık: `s_i / σ_i`. σ_i, 60 günlük EWMA yıllık vol (√365).
- Kol hedef vol: yıllık %25. Ölçek, ham portföyün geçmiş günlük getirilerinin 60 günlük EWMA vol'üyle belirlenir (yalnızca t'ye kadar veri).
- Sıra: varlık başına |w| ≤ %20 → brüt tavan (LF 1,0x; LS 1,5x) aşılırsa orantılı küçültme.

**Yeniden dengeleme ve maliyet.**
- Günlük yeniden dengeleme. No-trade bandı **göreli** tanımlanır: |hedef − mevcut| ≤ b·|hedef| ise işlem yok, b ∈ {0, 0,10, 0,25}. Çıkış (hedef 0) her zaman işlem görür. Mutlak bant, %20 varlık tavanında anlamsız kalırdı.
- Maliyet: KURALLAR §3 — taker %0,05; kayma BTC/ETH 2 bps, noktasal-zamanlı ilk-20 5 bps, diğerleri 15 bps; gerçek funding.
- Hesap: NAV 10.000 USDT birincil. Min notional ve adım, Binance'in GÜNCEL limitleri; delist olmuş semboller kısıtsız varsayılır ve raporlanır.
- NAV 1.000 USDT yalnızca ana aday için raporlanır: açılamayan pozisyon oranı ve getiri farkı.

**Evren ve walk-forward.**
- Evren: noktasal-zamanlı ilk N perpetual (30 g ort. hacim, stable/kaldıraçlı/endeks hariç), N ∈ {10, 20, 30}.
- Sinyal parametreleri SABİT. Walk-forward yalnızca (N, bant) seçer: 9 aday.
- Katmanlar 6 aylık: 2021-04-01, 2021-10-01, …, 2025-04-01 (son katman 2025-09-30'a kadar).
- Her katmanın başında, 2020-10-01'den o güne kadarki net Sharpe'ı en yüksek (N, bant) seçilir; yalnızca geçmiş veri kullanılır.
- Seçilen hedef ağırlıklar birleştirilip motor TEK kez koşturulur; katman geçiş maliyetleri dahildir.

**Deneme sayımı.** 12 sinyal × 2 yön × 9 (N, bant) = **216 temel varyant**, artı ana aday plato varyantları (aşağıda, 2 yön × 16 = 32).
- Toplam **248 deneme**, global sayaca eklenir.
- Walk-forward serileri bu varyantlardan yapılan seçimdir, ayrıca sayılmaz.
- NAV 1.000 koşuları raporlamadır, sayılmaz.
- **PBO:** 216 temel varyantın değerlendirme penceresindeki günlük getirileri üzerinden (CSCV, 16 blok).
- **Deflated Sharpe:** toplam deneme sayısı ve 248 varyantın Sharpe varyansı ile.

**Kıyaslar.** Hepsi aynı pencerede.
- BTC al-tut: perp fiyatı, funding yok (spot eşdeğeri).
- Eşit ağırlıklı ilk-10 al-tut: noktasal-zamanlı ilk-10, ay başında yeniden dengeleme, funding yok.
- BTC EMA200 long/flat: işlem maliyeti ve funding dahil.

**Stres ve duyarlılık.**
- Stres: 2021-05, 2022-05 (LUNA), 2022-11 (FTX) — dönem içi maks. DD (dönem başındaki tepeden) ve toparlanma süresi (gün; 2025-09-30'a kadar toparlanmadıysa belirtilir).
- Duyarlılık: ücret ×2 ve kayma ×3 (kabul eşiği), +1 gün gecikme (rapor).

**Plato.** Ana adayın her parametresi tek tek ×0,5 ve ×1,5 yapılır: 7 sinyal bileşeni (EMA'da çiftin iki ucu birlikte) ve vol EWMA span'i (60) → 16 komşu, her yönde.
- Komşular, ana adayın son walk-forward seçimindeki (N, bant) ile koşturulur.
- Plato oranı = en kötü komşu Sharpe / ana aday Sharpe. Eşik ≥ 0,70.

**Karar kuralı.** Kol eşikleri (KURALLAR §5): net Sharpe ≥ 0,8; DSR ≥ 0,95; PBO ≤ 0,25; maks. DD ≤ %35; Calmar ≥ max(BTC al-tut Calmar, 0,7); pozitif yıl ≥ %60; stres Sharpe ≥ 0,5; plato ≥ 0,70.
- Ana aday (LF ve LS ayrı ayrı) walk-forward serisiyle değerlendirilir. Bir yön tüm eşikleri geçerse **"trend kolu"** olarak işaretlenir.
- Geçmezse hangi eşikte kaldığı yazılır ve DURULUR: ızgara genişletilmez, yeniden deneme yapılmaz.
- Diğer varyantlar yalnızca bilgi içindir (plato hesaplanmaz): GEÇTİ/KALDI gösterilir ama karar ana adaya göre verilir.

**Çıktı.** `research/results/trend_001/` (rapor, varyant tabloları, stres, plato, turnover–getiri eğrisi). Bu günlüğe walk-forward seçilmiş 24 varyant ve kıyaslar tablosu yazılır; 216 varyantın tam tablosu sonuç klasöründedir.
