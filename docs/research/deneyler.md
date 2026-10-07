# Deney günlüğü

Kurallar için bkz. `KURALLAR.md`. Her çalıştırma `research.registry.register_experiment` ile buraya tek satır ekler.
Toplam deneme sayacının tek doğruluk kaynağı `research/results/_registry.json`'dır (Deflated Sharpe bunu kullanır).

**Toplam deneme: 248**

| Tarih (UTC) | Deney | Hipotez | Varyant | Net Sharpe | Maks. DD | Karar | Veri hash | Commit |
|---|---|---|---|---|---|---|---|---|
| 2026-10-07 07:35 | trend_001 | Kripto zaman serisi momentumu, çok varlıklı ve vol'e göre boyutlandırılmış olarak, maliyetler sonrası BTC al-tut'tan daha yüksek Calmar üretir | 248 | 0.43 | 47.9% | KALDI — main LF: (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress); main LS: (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress, plateau). DUR (ızgara genişletilmez). | 0774e2086a790a0b | unknown |

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

## SONUÇ — `trend_001` (ön kayıt: "ÖN KAYIT — trend")

**Karar:** KALDI — main LF: (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress); main LS: (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress, plateau). DUR (ızgara genişletilmez).

Pencere 2021-04-01 → 2025-09-30 · NAV 10,000 USDT · deneme: 248 varyant (toplam sayaç 248) · **PBO (216 varyant): 0.50** · BTC al-tut Calmar 0.21

### Walk-forward seçilmiş varyantlar (24) — KURALLAR §5 kol eşikleri
| Sinyal | Yön | Sharpe | Calmar | Maks. DD | Yıllık | Turnover | Maliyet/Brüt | Poz. yıl | DSR | BTC korr. | Stres SR | +1g SR | Durum |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a20 | LF | 0.40 | 0.17 | -42.1% | 7.3% | 31.2 | 29.2% | 80.0% | 0.28 | 0.62 | 0.22 | 0.45 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress) |
| a20 | LS | 0.04 | -0.06 | -34.7% | -2.0% | 47.5 | 85.7% | 40.0% | 0.09 | -0.06 | -0.25 | 0.16 | KALDI (net_sharpe, deflated_sharpe, pbo, calmar, positive_years, stress) |
| a60 | LF | 0.38 | 0.16 | -43.1% | 6.8% | 19.6 | 25.7% | 80.0% | 0.27 | 0.65 | 0.23 | 0.26 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress) |
| a60 | LS | 0.28 | 0.13 | -31.6% | 4.1% | 30.4 | 40.4% | 80.0% | 0.21 | 0.01 | 0.03 | 0.24 | KALDI (net_sharpe, deflated_sharpe, pbo, calmar, stress) |
| a120 | LF | 0.37 | 0.15 | -43.9% | 6.5% | 14.8 | 23.1% | 80.0% | 0.27 | 0.68 | 0.28 | 0.32 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress) |
| a120 | LS | 0.45 | 0.33 | -26.1% | 8.6% | 25.3 | 23.9% | 80.0% | 0.32 | 0.23 | 0.34 | 0.38 | KALDI (net_sharpe, deflated_sharpe, pbo, calmar, stress) |
| a250 | LF | -0.16 | -0.10 | -63.8% | -6.3% | 12.3 | — | 80.0% | 0.04 | 0.74 | -0.28 | 0.11 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress) |
| a250 | LS | 0.21 | 0.04 | -48.0% | 2.1% | 22.5 | 44.7% | 80.0% | 0.17 | 0.34 | 0.03 | 0.38 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress) |
| b20 | LF | 0.74 | 0.45 | -37.1% | 16.7% | 26.0 | 10.2% | 80.0% | 0.56 | 0.52 | 0.53 | 0.63 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar) |
| b20 | LS | 0.16 | 0.02 | -34.5% | 0.8% | 40.9 | 55.4% | 60.0% | 0.14 | -0.02 | -0.09 | 0.22 | KALDI (net_sharpe, deflated_sharpe, pbo, calmar, stress) |
| b60 | LF | 0.33 | 0.12 | -43.1% | 5.3% | 16.1 | 20.2% | 80.0% | 0.24 | 0.59 | 0.20 | 0.10 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress) |
| b60 | LS | -0.16 | -0.13 | -53.6% | -6.9% | 26.4 | — | 20.0% | 0.04 | 0.10 | -0.34 | -0.22 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, positive_years, stress) |
| b120 | LF | 0.57 | 0.28 | -42.4% | 12.0% | 11.8 | 12.5% | 80.0% | 0.42 | 0.61 | 0.50 | 0.52 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar) |
| b120 | LS | 0.48 | 0.28 | -32.4% | 9.2% | 19.0 | 19.7% | 80.0% | 0.34 | 0.25 | 0.37 | 0.44 | KALDI (net_sharpe, deflated_sharpe, pbo, calmar, stress) |
| b250 | LF | 0.04 | -0.04 | -47.7% | -1.7% | 10.1 | 75.0% | 40.0% | 0.09 | 0.70 | -0.04 | 0.06 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, positive_years, stress) |
| b250 | LS | -0.25 | -0.16 | -53.1% | -8.4% | 17.4 | — | 60.0% | 0.03 | 0.30 | -0.44 | -0.19 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress) |
| c_mix | LF | 0.24 | 0.06 | -51.9% | 2.9% | 11.7 | 23.9% | 80.0% | 0.18 | 0.61 | 0.16 | 0.19 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress) |
| c_mix | LS | 0.27 | 0.10 | -36.2% | 3.6% | 18.6 | 37.0% | 60.0% | 0.20 | 0.03 | 0.11 | 0.30 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress) |
| d20_10 | LF | 0.44 | 0.23 | -37.5% | 8.5% | 13.8 | 17.2% | 60.0% | 0.32 | 0.56 | 0.36 | 0.36 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress) |
| d20_10 | LS | 0.06 | -0.04 | -41.8% | -1.8% | 20.2 | 69.7% | 60.0% | 0.10 | 0.04 | -0.06 | 0.07 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress) |
| d55_20 | LF | 0.10 | -0.01 | -52.5% | -0.7% | 10.3 | 37.7% | 80.0% | 0.11 | 0.59 | 0.02 | -0.01 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress) |
| d55_20 | LS | -0.18 | -0.12 | -59.4% | -7.4% | 15.5 | — | 20.0% | 0.04 | 0.17 | -0.28 | -0.16 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, positive_years, stress) |
| main | LF | 0.43 | 0.17 | -47.9% | 8.0% | 18.1 | 15.9% | 80.0% | 0.31 | 0.60 | 0.29 | 0.41 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress) |
| main | LS | 0.34 | 0.16 | -35.2% | 5.5% | 28.3 | 36.5% | 60.0% | 0.24 | 0.07 | 0.12 | 0.37 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress, plateau) |

### Kıyaslar
| Kıyas | Sharpe | Calmar | Maks. DD | Yıllık | Turnover | BTC korr. |
|---|---|---|---|---|---|---|
| BTC al-tut | 0.54 | 0.21 | -76.7% | 15.9% | 0.0 | 1.00 |
| EW ilk-10 al-tut (aylık) | 0.15 | -0.22 | -92.6% | -19.9% | 7.9 | 0.83 |
| BTC EMA200 long/flat | 0.23 | 0.02 | -60.5% | 1.0% | 13.0 | 0.73 |

### Ana aday ayrıntıları
- **main LF**: walk-forward seçimleri 2021-04:N30/b0.25, 2021-10:N10/b0.25, 2022-04:N10/b0.25, 2022-10:N10/b0.25, 2023-04:N30/b0.25, 2023-10:N30/b0.25, 2024-04:N30/b0.25, 2024-10:N30/b0.25, 2025-04:N30/b0.25. Plato (N30/b0.25 sabit; temel Sharpe 0.52): oran **0.75**, en kötü komşu `a120×1.5` (0.39). NAV 1.000: açılamayan pozisyon 10.5% (10.000'de 0.3%), yıllık getiri 7.5% vs 8.0%.
- **main LS**: walk-forward seçimleri 2021-04:N10/b0.25, 2021-10:N10/b0.25, 2022-04:N10/b0.25, 2022-10:N10/b0.25, 2023-04:N30/b0.25, 2023-10:N30/b0.25, 2024-04:N30/b0.25, 2024-10:N30/b0.25, 2025-04:N30/b0.25. Plato (N30/b0.25 sabit; temel Sharpe 0.38): oran **0.49**, en kötü komşu `a120×1.5` (0.19). NAV 1.000: açılamayan pozisyon 8.7% (10.000'de 0.0%), yıllık getiri 6.4% vs 5.5%.
  Bacak katkısı (toplam, NAV oranı): long 38.6%, short 2.2%.

### Stres dönemleri (maks. DD / toparlanma günü)
| Seri | 2020-03 | 2021-05 | 2022-05 (LUNA) | 2022-11 (FTX) |
|---|---|---|---|---|
| main LF (WF) | değerlendirilemez | -8.8% / 84 | -23.4% / 656 | -39.9% / 479 |
| main LS (WF) | değerlendirilemez | -9.7% / 89 | -23.7% / 40 | -22.1% / 463 |
| BTC al-tut | -54.0% / 137 | -45.6% / 143 | -57.7% / 647 | -76.7% / 469 |
| EW ilk-10 al-tut (aylık) | değerlendirilemez | -51.6% / toparlanmadı | -82.8% / toparlanmadı | -89.8% / toparlanmadı |
| BTC EMA200 long/flat | değerlendirilemez | -44.5% / 1026 | -59.4% / 680 | -59.4% / 496 |

Tam tablo: `research/results/trend_001/varyantlar_216.md`; turnover-getiri: `turnover_getiri.png`. Limiti bilinmeyen (delist) semboller kısıtsız varsayıldı: 16 sembol.

> **Not ve düzeltmeler (trend_001)** — kayıtlar değiştirilmez; düzeltmeler bu notla yapılır. Karar DEĞİŞMEZ.
>
> **Kod sürümü:**
> - Kayıttaki `git_commit` "unknown" görünüyor, çünkü araştırma imajında `git` yok.
> - Sıra: ön kayıt `5de2cc8` → uygulama `dd11280` → gerçek veride tek koşu. Koşu sonrası `backend/` temizdi (HEAD = `dd11280`).
> - Koşunun son düzenlemeden sonra başladığı zaman damgalarıyla kesin olarak doğrulanamaz. Ana aday serisi `dd11280` koduyla yeniden kuruldu ve kayıttaki seriyle birebir aynı çıktı.
>
> **Raporlama düzeltmeleri** (denetimde bulundu; kod düzeltildi; değerler kayıttaki walk-forward seçimleriyle aynı koşu yeniden kurularak hesaplandı):
> - *Short bacak katkısı (main LS):* İlk hesap çıkış maliyetini hiçbir bacağa yazmıyordu. Doğru değerler: long %37,7, short **%0,8**, toplam %38,5 (raporlanan: short %2,2). Short bacağın katkısı pratikte sıfır.
> - *NAV 1.000'de atlanan emir oranı:* İlk hesabın paydası, gerçek işlem sayısı yerine pozisyon-gün sayısıydı. Doğru değerler: emir denemelerinin LF'de **%31,6**'sı, LS'de **%27,5**'i min notional yüzünden atlanıyor (raporlanan %10,5 ve %8,7). NAV 10.000'de bu oran %1,0 ve %0,01. Küçük hesapta uygulanabilirlik ciddi biçimde bozuluyor.
> - *+1 gün gecikme senaryosu:* Katman geçişlerinde bant bir gün erken hizalanıyordu. main'de tüm katmanlar b0,25 olduğu için etkisi yok; kod düzeltildi.
>
> **Ön kayıtta açık bırakılıp kodda yorumlanan noktalar** (sapma değil, yorum):
> - (a) Plato oranının paydası, walk-forward Sharpe değil, son seçimdeki sabit N30/b0,25 varyantının Sharpe'ı (0,52 ve 0,38); komşular da aynı sabit kombinasyonla koşturuldu. Walk-forward Sharpe ile oran LF 0,92, LS 0,55 olurdu; karar aynı.
> - (b) Pencerenin son getiri günü 2025-09-29. 09-30 getirisi 10-01 açılışını, yani nihai pencereyi gerektirir.
> - (c) Stres DD'si, serinin o güne kadarki kümülatif tepesinden ölçülüyor; yani dönemin kendi kaybını değil, o gün yaşanan toplam düşüşü gösteriyor. Örneğin FTX satırı 2021-11'den beri biriken düşüşü içerir.
> - (d) Donchian kanalları kapanışla değil, high/low ile kuruluyor.
> - (e) Walk-forward ısınma günleri (2020-10-01 → 2021-03-30) ilk katmanın seçimini kullanıyor; bu günler değerlendirme penceresinin dışında.
>
> **Ek kontroller:**
> - 216 varyant matrisinde boş ya da birebir aynı sütun yok. PBO tekrar hesabı 0,497; rastgele gürültü matrisinde 0,68.
> - EW ilk-10 kıyası bağımsız, maliyetsiz hesapla yıllık -%21 çıktı (raporlanan -%19,9).
>
> **Bütçe notu:** KURALLAR §9'daki "soru başına 40 config" bütçesi kodda kayıtlı deney başına sayılıyor. trend_001 bütçeden 1/40 düşüyor; deneme sayacına ise 248 varyant olarak işlendi.
