# Deney günlüğü

Kurallar için bkz. `KURALLAR.md`. Her çalıştırma `research.registry.register_experiment` ile buraya tek satır ekler.
Toplam deneme sayacının tek doğruluk kaynağı `research/results/_registry.json`'dır (Deflated Sharpe bunu kullanır).

**Toplam deneme: 321**

| Tarih (UTC) | Deney | Hipotez | Varyant | Net Sharpe | Maks. DD | Karar | Veri hash | Commit |
|---|---|---|---|---|---|---|---|---|
| 2026-10-07 07:35 | trend_001 | Kripto zaman serisi momentumu, çok varlıklı ve vol'e göre boyutlandırılmış olarak, maliyetler sonrası BTC al-tut'tan daha yüksek Calmar üretir | 248 | 0.43 | 47.9% | KALDI — main LF: (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress); main LS: (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, stress, plateau). DUR (ızgara genişletilmez). | 0774e2086a790a0b | unknown |
| 2026-10-07 16:41 | kesitsel_001 | Kripto'da kesitsel momentum (Liu, Tsyvinski & Wu 2022), maliyetler sonrası trend koluyla düşük korelasyonlu pozitif getiri üretir | 37 | 0.14 | 37.0% | KALDI — ana aday: (net_sharpe, deflated_sharpe, max_drawdown, calmar, stress, plateau). DUR (ızgara genişletilmez). | 0774e2086a790a0b | 13d10c7 |
| 2026-10-07 21:40 | carry_001 | Perpetual funding ve vadeli basis, delta-nötr toplandığında maliyetler sonrası risksiz getirinin belirgin üstünde ve trend koluyla düşük korelasyonlu getiri sağlar | 36 | 9.76 | 0.3% | KALDI — hiçbir varyant tüm koşulları geçmedi. DUR (ızgara genişletilmez). | 0774e2086a790a0b | 3a7563a |

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

## ÖN KAYIT — `kesitsel` (2026-10-07, sonuç görülmeden önce; dal `research/kesitsel`)
**Hipotez:** Kripto'da kesitsel momentum (göreli kazananların kısa vadede göreli kazanmaya devam etmesi; Liu, Tsyvinski & Wu 2022),
maliyetler sonrası trend koluyla düşük korelasyonlu pozitif getiri üretir.

**Bütçeye sığdırma (kullanıcı onayı, 2026-10-07).** Soru `kesitsel`, bütçe 40 varyant; bu deney **37** kullanır (aşağıda). Bant tek
değerde sabit; ana aday önceden belirlendi; zarar limiti %25.

**Veri ve pencereler.** trend_001 ile aynı: USDⓈ-M perpetual günlük OHLCV, pozisyon üretimi 2020-10-01'de başlar,
**değerlendirme penceresi 2021-04-01 → 2025-09-30**. Nihai pencere (>= 2025-10-01) kullanılmaz. 2020-03 stres dönemi değerlendirilemez.

**Evren.** Noktasal-zamanlı ilk N perpetual (30 g ort. hacim, stable/kaldıraçlı/endeks hariç, >= 90 g geçmiş), N ∈ {30, 50}.

**Skor** (gün t kapanışında; yalnızca <= t verisi; son 1 gün hariç):
- L ∈ {7, 14, 28} için `r_L = log(close_{t-1} / close_{t-1-L})`.
- Düz skor: her L için evren üyeleri arasında kesitsel z-skor (ortalama/std, o gün geçerli üyeler); üç z-skorun ortalaması.
  Üç bileşenden biri eksikse sembol o gün skorlanmaz.
- Vol-ayarlı skor: z-skorlardan önce `r_L / (σ · √L)`; σ = 60 günlük EWMA (span 60) günlük log getiri std'si.

**Portföyler** (dilim: skorlanmış üye sayısı n için k = max(1, ⌊n/5⌋) isim; en iyi k long, en kötü k short):
- a) **Dolar-nötr:** en iyi dilim long, en kötü dilim short. Her bacakta ters vol ağırlık (1/σ_i, σ yıllık 60 g EWMA), bacak brütü 1 (ham).
- b) **Long + BTC beta hedge:** en iyi dilim long (ters vol, ham brüt 1) + BTCUSDT'de `−β` short. β = long bacak ham portföyünün son 60 günlük
  günlük getirisinin BTC günlük getirisine OLS betası (yalnızca <= t), [0, 2]'ye kırpılır. BTC long dilimdeyse ağırlıklar netleşir.
- c) **Yalnızca long:** en iyi dilim (ters vol, ham brüt 1).

**Boyutlandırma.**
- Kol hedef vol yıllık **%20**: ölçek = 0,20 / (ham portföyün geçmiş günlük getirisinin 60 g EWMA vol'ü), yalnızca <= t (trend_001 ile aynı yöntem).
- İsim başına tavan **%10**, `research.sizing.asset_caps(base_cap=0,10)` ile: tavan_t = max(%10, bir lotun notional'ı / NAV) (KURALLAR §3).
  (b)'deki BTC hedge bacağı isim tavanına tabi değildir; yalnızca brüt tavana tabidir.
- Brüt tavan: (a) ve (b) **2,0x**, (c) **1,0x** (aşılırsa orantılı küçültme). Hesap kaldıraç tavanı ayrıca 3x.

**Yeniden dengeleme ve maliyet.**
- Haftalık: işlem pazartesi 00:00 UTC açılışında (karar pazar kapanışı). 3 günlük: 2020-10-01'den itibaren her 3. gün.
- Dengeleme günleri dışında işlem yok (zarar limiti çıkışları hariç); pozisyon drift'le tutulur.
- Göreli no-trade bandı **%25** sabit (dengeleme günlerinde |hedef − mevcut| ≤ 0,25·|hedef| ise işlem yok; çıkış her zaman).
- Maliyet: KURALLAR §3 (taker %0,05; kayma 2/5/15 bps; **gerçek funding**, short alır/öder).
- Hesap: **250 USDT, 3x, en yakın lot, Binance güncel limitleri** (KURALLAR §3). Atlanan emirler, lot_scale ve 1x önerisi raporlanır.

**Zarar limiti varyantı** (yalnızca ana aday portföyü, short isimler): dengeleme günündeki açılış fiyatı referanstır. Bir short ismin
kapanışı referansın **%25** üstüne çıkarsa o isim ertesi açılışta kapatılır ve bir sonraki dengelemeye kadar açılmaz.

**Ana aday (önceden belirlendi):** düz skor + (a) dolar-nötr. Karar YALNIZCA ana aday walk-forward serisine göre verilir.

**Walk-forward.** Skor ve portföy parametreleri SABİT. Walk-forward yalnızca (N, dengeleme sıklığı) seçer: 4 aday. Katmanlar trend_001 ile
aynı (6 aylık, 2021-04-01 … 2025-04-01); her katmanda 2020-10-01'den katman başına kadarki net Sharpe'ı en yüksek kombinasyon seçilir.
Birleştirilmiş hedef tek motor koşusunda çalışır. 6 (skor × portföy) + 1 (zarar limitli ana aday) walk-forward serisi raporlanır.

**Deneme sayımı: 37 varyant.**
- Temel ızgara: 2 evren × 2 skor × 3 portföy × 2 dengeleme = **24**.
- Zarar limitli ana aday: 2 evren × 2 dengeleme = **4**.
- Plato (ana adayın son walk-forward seçimiyle): 7, 14, 28 ufukları ve vol span'i (60; ters vol ve hedef vol) tek tek ×0,5 ve ×1,5
  (7→4/10, 14→7/21, 28→14/42, 60→30/90) = **8**. Plato oranı = en kötü komşu Sharpe / ana aday sabit-kombinasyon Sharpe; eşik ≥ 0,70.
- Trend karışımı: **1** (aşağıda).
- PBO: 24 temel varyant (CSCV, 16 blok). Deflated Sharpe: global toplam deneme sayısı ve 37 varyantın Sharpe varyansı ile.
- Walk-forward serileri ayrıca sayılmaz. Getiri–ufuk teşhisi işlem değildir, sayılmaz.

**Trend koluyla ilişki.** trend_001 bir kol üretmedi (KALDI). Karşılaştırma `research/results/trend_001/daily_returns.parquet` içindeki
`main_LF_wf` ve `main_LS_wf` serileriyle yapılır (bu seriler 10.000 USDT / aşağı yuvarlama ile koşmuştu; korelasyon için yeterli).
- Günlük getiri korelasyonu (Pearson, değerlendirme penceresi) her iki seriyle raporlanır.
- **Eşit risk karışımı:** ana aday WF serisi ile `main_LF_wf`; her gün ağırlık, serilerin t−1'e kadarki 60 g vol'üyle ters orantılı
  (toplam 1). Portföy eşikleriyle (Sharpe ≥ 1,0; DD ≤ %30) yalnızca BİLGİ için değerlendirilir.

**Teşhis — getiri–ufuk.** Ana skor, N = 50: her gün t için en iyi dilim − en kötü dilim (eşit ağırlık) ortalama ileri log getirisi,
t+1 açılışından itibaren h = 1…60 gün (brüt, maliyetsiz), değerlendirme penceresindeki günlerin ortalaması. Kümülatif farkın tepe
yaptığı ve düşmeye başladığı ufuk (momentumun geri dönüşe çevrildiği yer) raporlanır.

**Kıyaslar, stres, duyarlılık.** trend_001 ile aynı: BTC al-tut (hesap kısıtsız), EW ilk-10 al-tut (aylık), BTC EMA200 long/flat;
stres 2021-05, 2022-05 (LUNA), 2022-11 (FTX); ücret ×2 + kayma ×3 (kabul eşiği), +1 gün gecikme (rapor). Short bacak katkısı ayrıca.

**Karar kuralı.** Ana aday WF serisi KURALLAR §5 kol eşiklerinin TÜMÜNÜ (net Sharpe ≥ 0,8; DSR ≥ 0,95; PBO ≤ 0,25; maks. DD ≤ %35;
Calmar ≥ max(BTC al-tut Calmar, 0,7); pozitif yıl ≥ %60; stres Sharpe ≥ 0,5; plato ≥ 0,70) geçerse **VE** `main_LF_wf` ile `main_LS_wf`'nin
ikisiyle de korelasyonu < 0,5 ise **"kesitsel kol" adayı**. Aksi halde hangi koşulda kaldığı yazılır ve DURULUR: ızgara genişletilmez.
Diğer varyantlar bilgi içindir.

**Çıktı.** `research/results/kesitsel_001/`: sonuç, varyant tablosu, ana aday raporu, getiri–ufuk grafiği, ozet.json, günlük getiriler.

## SONUÇ — `kesitsel_001` (ön kayıt: "ÖN KAYIT — kesitsel")

**Karar:** KALDI — ana aday: (net_sharpe, deflated_sharpe, max_drawdown, calmar, stress, plateau). DUR (ızgara genişletilmez).

Pencere 2021-04-01 → 2025-09-30 · hesap 250 USDT, 3x, lot: nearest · deneme: 37 varyant (toplam sayaç 285) · **PBO (24 temel varyant): 0.24** · BTC al-tut Calmar 0.21

### Walk-forward serileri (7) — KURALLAR §5 kol eşikleri (karar yalnızca ana aday: düz | a)
| Skor | Portföy | Zarar lim. | Sharpe | Calmar | Maks. DD | Yıllık | Turnover | Maliyet/Brüt | Poz. yıl | DSR | BTC korr. | Trend LF korr. | Trend LS korr. | Stres SR | +1g SR | Durum |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| duz | a | — | 0.14 | 0.02 | -37.0% | 0.8% | 64.7 | 59.9% | 60.0% | 0.01 | -0.09 | 0.20 | 0.32 | -0.44 | 0.32 | KALDI (net_sharpe, deflated_sharpe, max_drawdown, calmar, stress, plateau) |
| duz | b | — | -0.11 | -0.09 | -51.1% | -4.3% | 33.4 | — | 40.0% | 0.00 | -0.03 | 0.40 | 0.19 | -0.53 | -0.14 | KALDI (net_sharpe, deflated_sharpe, max_drawdown, calmar, positive_years, stress) |
| duz | c | — | 0.18 | 0.04 | -42.9% | 1.6% | 20.5 | -12.5% | 80.0% | 0.01 | 0.66 | 0.75 | 0.19 | -0.09 | 0.26 | KALDI (net_sharpe, deflated_sharpe, max_drawdown, calmar, stress) |
| vol | a | — | 0.64 | 0.47 | -23.8% | 11.3% | 70.3 | 28.9% | 60.0% | 0.10 | -0.05 | 0.12 | 0.21 | 0.14 | 0.80 | KALDI (net_sharpe, deflated_sharpe, calmar, stress) |
| vol | b | — | -0.10 | -0.08 | -48.5% | -3.9% | 35.8 | — | 40.0% | 0.00 | -0.06 | 0.37 | 0.18 | -0.62 | -0.14 | KALDI (net_sharpe, deflated_sharpe, max_drawdown, calmar, positive_years, stress) |
| vol | c | — | 0.14 | 0.02 | -40.7% | 0.8% | 23.3 | 21.5% | 80.0% | 0.01 | 0.69 | 0.75 | 0.17 | -0.12 | 0.31 | KALDI (net_sharpe, deflated_sharpe, max_drawdown, calmar, stress) |
| duz | a | evet | 0.13 | 0.02 | -36.4% | 0.6% | 54.6 | 55.5% | 60.0% | 0.01 | -0.04 | 0.23 | 0.32 | -0.86 | 0.37 | KALDI (net_sharpe, deflated_sharpe, max_drawdown, calmar, stress) |

### Kıyaslar
| Kıyas | Sharpe | Calmar | Maks. DD | Yıllık | Turnover | BTC korr. |
|---|---|---|---|---|---|---|
| BTC al-tut | 0.54 | 0.21 | -76.7% | 15.9% | 0.0 | 1.00 |
| EW ilk-10 al-tut (aylık) | 0.15 | -0.22 | -92.6% | -19.9% | 7.9 | 0.83 |
| BTC EMA200 long/flat | 0.23 | 0.02 | -60.5% | 1.0% | 13.0 | 0.73 |

### Ana aday ayrıntıları
- Walk-forward seçimleri: 2021-04:N50/3D, 2021-10:N50/3D, 2022-04:N30/3D, 2022-10:N30/3D, 2023-04:N30/3D, 2023-10:N30/3D, 2024-04:N30/3D, 2024-10:N30/3D, 2025-04:N30/3D.
- Plato (N30/3D sabit; temel Sharpe 0.60): oran **0.39**, en kötü komşu `L14×0.5` (0.24). Komşular: L7×0.5 0.76, L7×1.5 0.98, L14×0.5 0.24, L14×1.5 0.37, L28×0.5 0.82, L28×1.5 0.67, vol_span×0.5 0.71, vol_span×1.5 0.43.
- Bacak katkısı (toplam, NAV oranı): long 43.4%, short -30.0% (toplam 13.4%). Zarar limitli: long 55.2%, short -43.4%.
- Hesap: atlanan emir denemesi 6.0% (nedenler: min_lot 422; zarar limitli: min_lot 192), lot_scale 1.05. 1x için önerilen asgari hesap: 570 USDT (BTCUSDT, fiyat tarihi 2025-09-30).
- Trend korelasyonu (ana aday): main_LF 0.20, main_LS 0.32 (eşik < 0.5).
- Eşit risk karışımı (ana aday + trend main_LF; bilgi, portföy eşikleri): Sharpe 0.52, maks. DD -34.8%, Calmar 0.23, yıllık 7.9%.
- Getiri–ufuk (düz skor, ilk-50, brüt): kümülatif fark h=3 günde tepe yapıyor (-0.0%); h=7: -0.1%, h=28: -0.6%, h=60: -1.7%. Grafik: `getiri_ufuk.png`.

### Stres dönemleri (maks. DD / toparlanma günü)
| Seri | 2020-03 | 2021-05 | 2022-05 (LUNA) | 2022-11 (FTX) |
|---|---|---|---|---|
| ana aday (WF) | değerlendirilemez | -9.0% / toparlanmadı | -25.1% / toparlanmadı | -15.4% / toparlanmadı |
| ana aday + zarar limiti (WF) | değerlendirilemez | -7.1% / 179 | -18.5% / toparlanmadı | -16.8% / toparlanmadı |
| BTC al-tut | -54.0% / 137 | -45.6% / 143 | -57.7% / 647 | -76.7% / 469 |
| EW ilk-10 al-tut (aylık) | değerlendirilemez | -51.6% / toparlanmadı | -82.8% / toparlanmadı | -89.8% / toparlanmadı |
| BTC EMA200 long/flat | değerlendirilemez | -44.5% / 1026 | -59.4% / 680 | -59.4% / 496 |

Tam tablo: `research/results/kesitsel_001/varyantlar.md`. Limiti bilinmeyen (delist) semboller kısıtsız varsayıldı: 20 sembol.
> **Not (kesitsel_001)** — kayıtlar değiştirilmez; notlar karar DEĞİŞTİRMEZ.
>
> **Getiri–ufuk okuması:** Eşit ağırlıklı en iyi − en kötü dilim farkı 1–60 günün HİÇBİRİNDE pozitif değil (h=1 −%0,01, h=7 −%0,09,
> h=28 −%0,59, h=60 −%1,74). "h=3'te tepe" ifadesi sıfıra en yakın değerdir, momentumun tepesi değildir. Bu evrende ve dönemde
> 7/14/28 günlük kesitsel momentum, eşit ağırlıkla bile devam etmiyor; ana adayın küçük brüt kârı ters vol ağırlıklandırmadan geliyor.
> Short bacak net −%30 (NAV oranı) kaybettirdi; zarar limiti bunu düzeltmedi (−%43).
>
> **Bilgi amaçlı (karar dışı):** vol-ayarlı dolar-nötr (vol|a) WF serisi Sharpe 0,64, maks. DD −%23,8; yine net Sharpe, DSR, Calmar ve
> stres eşiklerinde kalıyor. Ön kayıt gereği aday değildir; bu soru için yeni deneme yapılmaz (bütçe 37/40).
>
> **Ön kayıtta açık bırakılıp kodda yorumlanan noktalar** (sapma değil, yorum; denetimde listelendi):
> - (a) Zarar limiti `close ≥ 1,25 × ref` ile tetiklenir.
> - (b) Kesitsel z-skor std'si ddof=0.
> - (c) (b) portföyünde BTC long dilimdeyse hedge ile netleşir; netleşmiş BTC ağırlığı isim tavanından muaftır.
> - (d) (b)'nin betası ve tüm portföylerin hedef vol ölçeği, günlük yeniden hesaplanan ham ağırlıklarla kurulur (haftalık tutulan
>   pozisyonla değil).
> - (e) Trend serileri yalnızca değerlendirme penceresinde var; karışım ilk 60 gün eşit ağırlıklı başlar.
> - (f) +1 gün senaryosunda dengeleme günleri ve zarar limiti maskesi hedefle birlikte kaydırılır.
> - (g) Kıyaslar `trend.benchmarks` ile (10.000 NAV); BTC al-tut Calmar'ı hesap kısıtından etkilenmiyor (denetimde 0,2073–0,2074);
>   eşik zaten 0,7'ye bağlı.
> - KURALLAR §9 sızıntı testleri (karıştırılmış hedef, `available_at`, tekrarlanabilirlik) model tabanlı koşucu içindir; bu kural tabanlı
>   modülde (trend_001'de olduğu gibi) yoktur. İleri bakış birim testlerle sınandı (skor, beta, seçim, zarar limiti).

## ÖN KAYIT — `carry` (2026-10-07, sonuç görülmeden önce; dal `research/carry`)
**Hipotez:** Perpetual funding ve vadeli kontrat basis'i, kaldıraçlı long talebinin yapısal bir primidir. Delta-nötr toplandığında
maliyetler sonrası risksiz getirinin belirgin üstünde ve trend koluyla düşük korelasyonlu getiri sağlar.

**Kullanıcı kararları (2026-10-07):** bütçe taslağı (36 varyant) onaylandı; risksiz getiri **yıllık %4**; boşta kalan USDT **%0** kazanır
(faizli sürüm yalnızca bilgi); A stratejisinde plato uygulanamaz sayılır.

**Veri.** USDⓈ-M perpetual günlük OHLCV + gerçek funding olayları (her sembolün kendi aralığı: 8/4/2/1 saat; olaylar ödeme zamanına
göre günlüğe toplanır, `engine.daily_funding`), aynı adlı spot çifti günlük OHLCV, BTC/ETH USDⓈ-M quarterly günlük OHLCV, perp 1 saatlik
OHLCV (bacak riski için). Mark/index fiyatı indirilmedi: basis = perp (ya da quarterly) − spot, aynı zaman damgasıyla.
Lot/asgari emir: Binance güncel perp limitleri (quarterly'de aynı varlığın perp limiti; spot bacak perp miktarına eşitlenir).

**Pencere.** Değerlendirme **2021-04-01 → 2025-09-30** (öncekilerle aynı); nihai pencere kullanılmaz. A ayrıca 2020-01-01'den koşar;
2020 ve 2021-Q1 yalnızca yıllık tabloda bilgi olarak gösterilir.

**Ortak yürütme ve muhasebe.**
- Karar gün t kapanışında, işlem t+1 açılışında. Pozisyon = q coin spot long + q coin perp/quarterly short (1x hedge, eşit miktar).
- **Bağlanan sermaye:** notional N için spot cüzdanında N, vadeli cüzdanında marj N/kaldıraç. Portföy Marjı (PM) varyantında spot teminattır:
  bağlanan sermaye 1,10·N (%10 USDT tampon). Getiri, 250 USDT hesabın günlük NAV'ından ölçülür (boştaki nakit %0); ayrıca
  "sermaye verimliliği" = ortalama bağlanan sermaye / NAV ve bağlanan sermayeye göre yıllık getiri raporlanır.
- **Hesap:** 250 USDT; q, perp miktar adımına EN YAKIN lota yuvarlanır; asgari emir altındaki pozisyon açılmaz (KURALLAR §3).
- **Maliyetler:** spot ücreti %0,10, perp/quarterly taker %0,05; kayma her iki bacakta KURALLAR §3 kademeleri (2/5/15 bps).
  **Bacak riski:** her bacak çifti işleminde ek maliyet = E|ΔP| over τ dakika = σ_1s·√(τ/60)·√(2/π); σ_1s = son 7 günün 1 saatlik
  perp log getiri std'si (1 saatlik veri yoksa günlük σ/√24). Birincil τ = 1 dk; τ = 5 dk duyarlılık olarak raporlanır.
- **Cüzdan dengeleme ve tasfiye:** her gün açılışta vadeli cüzdan özsermayesi E_p, hedef marjın (q·P/kaldıraç) %50'sinin altına
  düşmüş ya da %150'sinin üstüne çıkmışsa iki bacak birlikte yeniden boyutlanır (aynı q), nakit cüzdanlar arasında taşınır
  (transfer ücretsiz; bacak işlemleri ücretli). Gün içi en yüksek fiyatta E_p − q·(H − P_açılış) ≤ %1·q·H ise perp bacağı **tasfiye**:
  vadeli cüzdan özsermayesi 0'a iner, spot kalır, ertesi açılışta kalan özsermayeyle yeniden hedge'lenir. PM'de tasfiye simüle edilmez
  (yalnızca stres raporu). Ayrıca: dönem içi en kötü günlük yükseliş ve +%30 tek gün şoku için her kaldıraçta marj tamponunun yetip
  yetmediği raporlanır.
- **Funding:** short bacak, (t açılışı, t+1 açılışı] aralığındaki ödemeleri q·P_açılış üzerinden alır/öder.
- **PnL kalemleri ayrı:** funding geliri, basis PnL (iki bacağın fiyat PnL toplamı), ücret+kayma, bacak riski, tasfiye kaybı.
- **Delist:** perp ya da spot fiyatı biterse iki bacak son mevcut kapanıştan kapatılır (maliyetler dahil). Delist tarihinden önceki 30 gün
  içinde açık carry pozisyonu olan semboller ve PnL'leri raporlanır. Funding tavanı: tutulan pozisyonlarda |oran| ≥ %0,3 / olay
  sayısı raporlanır. ADL simüle edilemez (veri yok); not düşülür.

**Stratejiler ve varyantlar (36).**
- **A — BTC/ETH sürekli carry (8):** tüm sermaye tek varlıkta, her zaman açık. {BTC, ETH} × kaldıraç {1x, 2x, 3x} = 6; PM {BTC, ETH} = 2.
- **B — koşullu çok coinli funding carry (12):**
  - Evren: noktasal-zamanlı ilk 30 perpetual (KURALLAR universe), aynı adlı spot çifti olanlar (spotu olmayan `1000…` vb. hariç).
  - Beklenen getiri (yıllık, notional'a göre) = son W günün günlük funding toplamlarının ortalaması × 365, W ∈ {3, 7}.
  - Giriş eşiği = gidiş-dönüş maliyet (2 × (spot ücreti + perp ücreti + iki bacak kayması + bacak riski)) × 365 / **14** (beklenen tutma
    süresi, gün) + **%10** (güvenlik payı). Yalnızca pozitif funding (perp short).
  - Çıkış (histerezis): beklenen getiri < **%3** (negatif dahil). Tutulan pozisyon çıkış koşuluna kadar tutulur; boş slotlar en yüksek
    beklenen getirili uygun sembollerle doldurulur.
  - En fazla K ∈ {3, 5, 8} sembol; slot başına sermaye NAV/K; slot notional'ı ≤ son 30 günün ortalama günlük quote hacminin %0,1'i
    (perp ve spotun küçüğü). Kaldıraç {1x, 3x}. 3 × 2 × 2 = 12.
- **C — vadeli basis, cash-and-carry (8):**
  - Kontrat: kalan gün ≥ 30 olan en yakın BTC/ETH USDⓈ-M quarterly.
  - Yıllık basis = (F/S − 1) × 365 / kalan gün. Giriş eşiği = gidiş-dönüş maliyet × 365 / kalan gün + %10.
  - "Vadeye kadar tut": vade günü açılışında iki bacak kapanır (quarterly'nin uzlaşma fiyatı ≈ spot açılışı varsayılır; ücretler dahil);
    ertesi karar gününde yeni kontrat eşiği aşıyorsa açılır.
  - "Daralınca geç": yıllık basis < %3 olunca erken kapanır; sonraki uygun kontrat eşiği aşıyorsa ona geçilir.
  - {BTC, ETH} × {vadeye kadar, daralınca geç} × kaldıraç {1x, 3x} = 8. Funding yok.
- **Plato (8):** B ana adayı (K5, W7, 1x): güvenlik payı %10 → %5/%15, çıkış eşiği %3 → %1,5/%4,5, beklenen tutma 14 → 7/21 gün (6).
  C ana adayı (BTC, vadeye kadar, 1x): güvenlik payı %10 → %5/%15 (2). Plato oranı = en kötü komşu Sharpe / ana aday Sharpe ≥ 0,70;
  bir stratejinin plato sonucu o stratejinin TÜM varyantlarına uygulanır. **A'nın ayarlanabilir parametresi yoktur; plato uygulanamaz
  (geçti sayılır).**
- **D — kıyas (deneme değil):** USDT'yi boşta tutup yıllık %4 almak. Ayrıca BTC al-tut (Calmar eşiği için).

**Deneme sayımı:** 8 + 12 + 8 + 8 = **36** varyant (soru `carry`). Walk-forward yok: serbest parametre seçimi yapılmaz, her varyant sabit
koşar. PBO: 28 temel varyant (CSCV, 16 blok). DSR: global toplam deneme sayısı ve 36 varyantın Sharpe varyansı. τ = 5 dk, faizli boş
nakit, +1 gün gecikme ve stres koşuları raporlamadır, sayılmaz.

**Çıktı.** Strateji × varyant tablosu: NAV'a ve bağlanan sermayeye göre yıllık getiri, vol, Sharpe, maks. DD, en uzun su altı süresi
(gün), sermaye verimliliği, işlem sayısı, tasfiye sayısı, PnL kalemleri, 2020–2025 yıllık getiriler, BTC ve trend_001 (main_LF_wf,
main_LS_wf) korelasyonu. Stres dönemleri ve duyarlılık (ücret ×2 + kayma ×3, bacak riski τ = 5 dk, +1 gün) öncekilerle aynı.

**Karar kuralı.** Bir varyant "carry kolu" adayıdır ancak KURALLAR §5 kol eşiklerinin TÜMÜNÜ (net Sharpe ≥ 0,8; DSR ≥ 0,95; PBO ≤ 0,25;
maks. DD ≤ %35; Calmar ≥ max(BTC al-tut Calmar, 0,7); pozitif yıl ≥ %60; ücret ×2 + kayma ×3 Sharpe ≥ 0,5; plato ≥ 0,70) geçerse **VE**
NAV'a göre yıllık net getirisi ≥ **%8** (risksiz %4'ün 2 katı) ise. Geçenlerden **EN BASİT** olan aday ilan edilir; sadelik sırası:
A-1x (BTC, ETH) → A-2x → A-3x → A-PM → C (1x önce; vadeye kadar önce; BTC önce) → B (1x önce; K3 → K5 → K8; W7 → W3).
Hiçbiri geçmezse DURULUR ve hangi koşullarda kalındığı yazılır. Trend korelasyonu raporlanır, karar koşulu değildir.

> **ÖN KAYIT EKİ — `carry` (2026-10-08, gerçek koşudan önce; kayıt değil, olay notu).** İlk kod denetiminde denetçi ajan, muhasebeyi
> doğrulamak için gerçek veride ~40 simülasyon çalıştırdı. Bir kontrol çıktısında bazı varyantların pencere içi yıllık getiri ve Sharpe
> değerleri basıldı. Bu sayılar uygulayıcıya iletilmedi ve hiçbir karara girmedi; kayıt ve sayaç değişmedi. Ön kayıt (`e3b5ebc`) bu
> bakıştan ÖNCE commit edilmişti; sonrasında eşik, varyant ya da parametre DEĞİŞMEDİ. Denetimden sonra yapılan değişiklikler yalnızca
> ön kayda uyum ve muhasebe düzeltmeleridir: B ve C de pencere başında 250 USDT ile başlar (A ayrıca 2020'den, yalnızca yıllık tablo);
> tasfiye kaybı ayrı kalem; atlanan emir / lot_scale / 1x önerisi raporlanır; tek bacak fiyatsızken MTM ve işlem yapılmaz, > 5 gün
> senkron boşlukta son kapanıştan kapatılır; lot yuvarlaması marjı hedefe getiremezse boştaki nakit vadeli cüzdana aktarılır (ön kayıt:
> "nakit cüzdanlar arasında taşınır"); vade günü yeni kontrat açılmaz (ertesi karar günü); plato oranında NaN komşu = başarısız;
> stres ve τ = 5 dk koşularında giriş/çıkış eşikleri birincil maliyetle kalır. Kullanıcı kararı (2026-10-08): not düşülerek devam edilir.
> Sonraki denetimlerde denetçi gerçek veride metrik üretmez.

## SONUÇ — `carry_001` (ön kayıt: "ÖN KAYIT — carry")

**Karar:** KALDI — hiçbir varyant tüm koşulları geçmedi. DUR (ızgara genişletilmez).

Pencere 2021-04-01 → 2025-09-30 · hesap 250 USDT · deneme: 36 varyant (toplam sayaç 321) · **PBO (28 temel varyant): 0.07** · BTC al-tut Calmar 0.21 · risksiz %4 (getiri eşiği %8)

### Varyantlar (sadelik sırasıyla) — KURALLAR §5 kol eşikleri + yıllık getiri ≥ %8
| Varyant | Yıllık (NAV) | Yıllık (bağlı serm., aritm.) | Vol | Sharpe | Maks. DD | Calmar | Su altı (g) | Serm. verim. | İşlem | Tasfiye | Funding | Basis | Maliyet | Bacak r. | Tasfiye kaybı | Atlanan | lot_scale | DSR | Stres SR | τ5 SR | +1g SR | BTC korr. | Trend LF/LS | Durum |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A|BTC|1x | 4.3% | 4.6% | 0.4% | 9.76 | -0.3% | 16.36 | 74 | 89.1% | 22 | 0 | 21.9% | 0.0% | -0.9% | -0.3% | -0.0% | — | 0.94 | 1.00 | 7.92 | 9.46 | 9.76 | -0.11 | -0.05 / 0.01 | KALDI (min_return) |
| A|ETH|1x | 4.9% | 4.8% | 0.6% | 8.22 | -0.9% | 5.27 | 165 | 98.7% | 26 | 0 | 25.4% | -0.0% | -0.9% | -0.5% | -0.0% | — | 0.98 | 0.54 | 7.07 | 7.95 | 8.22 | -0.04 | 0.00 / 0.02 | KALDI (deflated_sharpe, min_return) |
| A|BTC|2x | 2.4% | 4.0% | 3.6% | 0.67 | -7.7% | 0.31 | 993 | 59.6% | 11 | 1 | 19.4% | 0.1% | -0.3% | -0.1% | -7.8% | — | 0.71 | 0.00 | 0.64 | 0.67 | 0.67 | -0.05 | 0.00 / 0.01 | KALDI (net_sharpe, deflated_sharpe, calmar, min_return) |
| A|ETH|2x | 6.0% | 5.8% | 0.8% | 7.67 | -1.4% | 4.25 | 263 | 99.4% | 67 | 0 | 33.2% | -0.1% | -2.0% | -1.0% | -0.0% | — | 0.99 | 0.22 | 6.24 | 7.33 | 7.67 | -0.04 | 0.01 / 0.02 | KALDI (deflated_sharpe, min_return) |
| A|BTC|3x | 3.3% | 3.7% | 5.9% | 0.58 | -12.9% | 0.26 | 944 | 92.1% | 51 | 1 | 31.1% | 0.0% | -1.6% | -0.7% | -13.0% | — | 0.94 | 0.00 | 0.20 | 0.20 | 0.58 | -0.06 | 0.00 / 0.01 | KALDI (net_sharpe, deflated_sharpe, calmar, stress, min_return) |
| A|ETH|3x | 6.6% | 6.3% | 0.8% | 7.54 | -1.7% | 3.91 | 338 | 99.6% | 103 | 0 | 37.6% | -0.0% | -2.8% | -1.4% | -0.0% | — | 0.99 | 0.16 | 5.50 | 7.02 | 7.54 | -0.04 | 0.01 / 0.03 | KALDI (deflated_sharpe, min_return) |
| A|BTC|PM | 7.2% | 8.6% | 0.7% | 10.63 | -0.3% | 22.94 | 60 | 79.1% | 41 | 0 | 38.2% | 0.1% | -1.0% | -0.4% | -0.0% | — | 0.78 | 1.00 | 8.76 | 10.26 | 10.63 | -0.11 | -0.06 / -0.00 | KALDI (min_return) |
| A|ETH|PM | 7.0% | 6.6% | 1.0% | 6.73 | -2.6% | 2.74 | 388 | 99.9% | 398 | 0 | 44.9% | -0.1% | -6.2% | -3.1% | -0.0% | — | 1.00 | 0.01 | 3.92 | 5.90 | 6.73 | -0.03 | 0.01 / 0.02 | KALDI (deflated_sharpe, min_return) |
| C|BTC|vade|1x | 2.1% | 7.7% | 1.8% | 1.14 | -2.3% | 0.91 | 742 | 26.4% | 19 | 0 | 0.0% | 11.4% | -1.2% | -0.4% | -0.0% | — | 0.77 | 0.00 | 0.85 | 1.07 | 1.14 | -0.30 | -0.21 / -0.21 | KALDI (deflated_sharpe, min_return) |
| C|ETH|vade|1x | 2.2% | 6.3% | 2.4% | 0.95 | -2.2% | 1.02 | 743 | 34.9% | 25 | 0 | 0.0% | 12.9% | -1.7% | -0.7% | -0.0% | — | 1.00 | 0.00 | 0.60 | 0.86 | 0.96 | -0.27 | -0.22 / -0.21 | KALDI (deflated_sharpe, min_return) |
| C|BTC|gec|1x | 2.1% | 8.3% | 1.8% | 1.15 | -2.3% | 0.92 | 742 | 24.8% | 18 | 0 | 0.0% | 11.4% | -1.2% | -0.4% | -0.0% | — | 0.77 | 0.00 | 0.88 | 1.09 | 1.15 | -0.29 | -0.21 / -0.21 | KALDI (deflated_sharpe, min_return) |
| C|ETH|gec|1x | 2.2% | 6.7% | 2.4% | 0.94 | -2.2% | 1.01 | 743 | 32.9% | 24 | 0 | 0.0% | 12.8% | -1.7% | -0.7% | -0.0% | — | 1.00 | 0.00 | 0.59 | 0.85 | 0.94 | -0.27 | -0.22 / -0.21 | KALDI (deflated_sharpe, min_return) |
| C|BTC|vade|3x | 3.4% | 11.5% | 2.8% | 1.20 | -3.4% | 0.99 | 737 | 28.8% | 28 | 0 | 0.0% | 19.4% | -2.2% | -0.8% | -0.0% | — | 0.87 | 0.00 | 0.86 | 1.12 | 1.20 | -0.29 | -0.22 / -0.22 | KALDI (deflated_sharpe, min_return) |
| C|ETH|vade|3x | 3.0% | 8.7% | 3.4% | 0.91 | -3.3% | 0.92 | 743 | 34.5% | 45 | 0 | 0.0% | 18.6% | -2.9% | -1.3% | -0.0% | — | 1.00 | 0.00 | 0.50 | 0.79 | 0.91 | -0.28 | -0.23 / -0.21 | KALDI (deflated_sharpe, min_return) |
| C|BTC|gec|3x | 3.4% | 12.1% | 2.8% | 1.20 | -3.4% | 1.00 | 736 | 27.3% | 26 | 0 | 0.0% | 19.4% | -2.2% | -0.8% | -0.0% | — | 0.87 | 0.00 | 0.88 | 1.13 | 1.20 | -0.29 | -0.22 / -0.22 | KALDI (deflated_sharpe, min_return) |
| C|ETH|gec|3x | 3.0% | 9.1% | 3.3% | 0.91 | -3.3% | 0.91 | 743 | 32.5% | 43 | 0 | 0.0% | 18.4% | -2.8% | -1.3% | -0.0% | — | 1.00 | 0.00 | 0.50 | 0.79 | 0.90 | -0.28 | -0.23 / -0.21 | KALDI (deflated_sharpe, min_return) |
| B|K3|W7|1x | 3.1% | 10.9% | 1.3% | 2.34 | -1.8% | 1.66 | 688 | 26.6% | 90 | 0 | 21.4% | -1.7% | -3.3% | -1.8% | -0.0% | — | 0.98 | 0.00 | 0.80 | 1.92 | 2.04 | 0.00 | 0.04 / 0.04 | KALDI (deflated_sharpe, min_return) |
| B|K3|W3|1x | 3.1% | 8.8% | 1.3% | 2.43 | -2.4% | 1.33 | 407 | 33.9% | 114 | 0 | 23.0% | -1.7% | -4.1% | -2.4% | -0.0% | price 1 | 0.99 | 0.00 | 0.56 | 1.87 | 2.33 | -0.01 | 0.01 / 0.00 | KALDI (deflated_sharpe, min_return) |
| B|K5|W7|1x | 3.2% | 10.9% | 1.0% | 3.24 | -1.1% | 2.90 | 685 | 28.0% | 154 | 0 | 21.1% | -0.9% | -3.1% | -1.8% | -0.0% | cash 2 | 0.99 | 0.00 | 1.29 | 2.70 | 2.99 | -0.01 | 0.03 / 0.03 | KALDI (deflated_sharpe, min_return) |
| B|K5|W3|1x | 3.1% | 10.2% | 1.0% | 3.23 | -1.4% | 2.22 | 412 | 29.6% | 166 | 0 | 21.4% | -1.0% | -3.5% | -2.0% | -0.0% | price 1 | 0.98 | 0.00 | 0.96 | 2.54 | 3.13 | -0.01 | 0.01 / 0.01 | KALDI (deflated_sharpe, min_return) |
| B|K8|W7|1x | 3.3% | 12.6% | 1.0% | 3.42 | -0.7% | 4.81 | 700 | 25.2% | 201 | 2 | 19.3% | -0.5% | -2.4% | -1.4% | 0.9% | min_lot 22 | 1.03 | 0.00 | 1.36 | 2.60 | 3.12 | -0.02 | 0.02 / 0.02 | KALDI (deflated_sharpe, min_return) |
| B|K8|W3|1x | 3.2% | 11.4% | 0.8% | 3.82 | -0.9% | 3.58 | 402 | 26.7% | 227 | 0 | 20.4% | -0.5% | -3.0% | -1.7% | -0.0% | cash 2, price 1 | 1.02 | 0.00 | 1.46 | 3.15 | 3.64 | -0.02 | 0.01 / -0.00 | KALDI (deflated_sharpe, min_return) |
| B|K3|W7|3x | 3.5% | 11.3% | 7.8% | 0.47 | -13.4% | 0.26 | 1433 | 26.9% | 202 | 21 | 32.4% | -2.9% | -7.6% | -4.7% | -0.7% | — | 1.00 | 0.00 | 0.39 | 0.39 | 0.42 | -0.00 | 0.01 / 0.01 | KALDI (net_sharpe, deflated_sharpe, calmar, positive_years, stress, min_return) |
| B|K3|W3|3x | 2.7% | 7.4% | 7.8% | 0.38 | -11.0% | 0.24 | 1595 | 33.9% | 240 | 22 | 31.5% | -2.9% | -8.3% | -5.2% | -2.4% | price 1 | 0.99 | 0.00 | -0.23 | 0.14 | 0.73 | 0.03 | 0.04 / 0.03 | KALDI (net_sharpe, deflated_sharpe, calmar, positive_years, stress, min_return) |
| B|K5|W7|3x | 6.2% | 21.0% | 5.2% | 1.19 | -4.7% | 1.33 | 778 | 27.1% | 333 | 28 | 30.1% | -1.9% | -6.6% | -4.2% | 13.9% | — | 0.98 | 0.00 | 0.51 | 0.99 | 0.85 | 0.00 | 0.02 / 0.02 | KALDI (deflated_sharpe, min_return) |
| B|K5|W3|3x | 2.5% | 8.0% | 5.2% | 0.50 | -7.5% | 0.33 | 1029 | 29.7% | 349 | 29 | 29.6% | -1.6% | -6.9% | -4.3% | -5.1% | price 1 | 0.99 | 0.00 | -0.34 | 0.15 | 0.83 | 0.01 | 0.03 / 0.01 | KALDI (net_sharpe, deflated_sharpe, calmar, positive_years, stress, min_return) |
| B|K8|W7|3x | 0.8% | 3.2% | 7.5% | 0.15 | -13.8% | 0.06 | 1599 | 25.6% | 529 | 105 | 24.5% | -0.8% | -6.0% | -4.8% | -9.1% | — | 1.00 | 0.00 | -0.28 | -0.02 | -0.01 | 0.15 | 0.14 / 0.14 | KALDI (net_sharpe, deflated_sharpe, calmar, stress, min_return) |
| B|K8|W3|3x | 2.2% | 8.1% | 3.7% | 0.60 | -4.4% | 0.50 | 1019 | 26.4% | 496 | 37 | 27.8% | -0.9% | -5.8% | -3.6% | -7.2% | price 1 | 0.99 | 0.00 | -0.37 | 0.38 | -0.26 | 0.02 | 0.04 / 0.02 | KALDI (net_sharpe, deflated_sharpe, calmar, stress, min_return) |

PnL kalemleri pencere toplamıdır, pencere başı NAV'ına (250 USDT) oranla. Basis = iki bacağın fiyat PnL'i; tasfiye kaybı = hedge'li kalsaydı oluşacak vadeli PnL'e göre fark (ayrı kalem). Stres (ücret ×2, kayma ve bacak riski ×3) ve τ = 5 dk koşularında giriş/çıkış eşikleri birincil maliyetle kalır; yalnızca yürütme pahalanır. Bağlanan sermayeye göre getiri aritmetiktir (Σ PnL / Σ bağlı sermaye × 365).
1x için önerilen asgari hesap (KURALLAR §3 tanımı: her sembolün en küçük lotu %20 tavana sığar; carry'de gereken hesap yaklaşık lot × (1 + 1/kaldıraç) × K): 570 USDT (BTCUSDT, fiyat tarihi 2025-09-30).

### Yıllık getiri (NAV). A: 2020-01-01'den ayrı koşu (2020 ve 2021-Q1 bilgi); B ve C: 2021-04-01'den (2021 kısmi yıl)
| Varyant | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Boş nakit %4 ile yıllık |
|---|---|---|---|---|---|---|---|
| A|BTC|1x | 9.0% | 18.3% | 1.9% | 4.5% | 6.7% | 2.0% | 4.7% |
| A|ETH|1x | -27.9% | 23.0% | 0.2% | 4.4% | 7.0% | 1.7% | 5.0% |
| A|BTC|2x | 11.7% | -1.5% | 2.3% | 5.5% | 7.6% | 2.6% | 4.0% |
| A|ETH|2x | -0.1% | 28.5% | -0.2% | 5.6% | 9.1% | 2.3% | 6.0% |
| A|BTC|3x | 12.9% | 10.5% | 1.5% | 2.6% | 7.3% | 2.4% | 3.7% |
| A|ETH|3x | 14.3% | 30.9% | -0.4% | 6.2% | 9.9% | 2.4% | 6.6% |
| A|BTC|PM | 14.1% | 28.3% | 2.1% | 4.9% | 10.0% | 3.4% | 8.0% |
| A|ETH|PM | 24.3% | 35.9% | -1.5% | 7.0% | 11.2% | 2.3% | 7.0% |
| C|BTC|vade|1x | — | 4.7% | 0.0% | -0.5% | 4.2% | 1.1% | 5.3% |
| C|ETH|vade|1x | — | 2.9% | 0.0% | -0.8% | 6.7% | 1.4% | 4.9% |
| C|BTC|gec|1x | — | 4.8% | 0.0% | -0.5% | 4.2% | 1.0% | 5.4% |
| C|ETH|gec|1x | — | 2.9% | 0.0% | -0.8% | 6.7% | 1.3% | 5.0% |
| C|BTC|vade|3x | — | 6.8% | 0.0% | -1.1% | 7.8% | 2.0% | 6.5% |
| C|ETH|vade|3x | — | 3.5% | 0.0% | -1.1% | 9.5% | 2.1% | 5.8% |
| C|BTC|gec|3x | — | 7.0% | 0.0% | -1.1% | 7.8% | 2.0% | 6.6% |
| C|ETH|gec|3x | — | 3.6% | 0.0% | -1.1% | 9.5% | 2.0% | 5.8% |
| B|K3|W7|1x | — | 9.5% | -0.0% | 0.5% | 5.3% | -1.2% | 6.1% |
| B|K3|W3|1x | — | 9.1% | -0.1% | 1.5% | 5.5% | -1.6% | 5.9% |
| B|K5|W7|1x | — | 9.4% | -0.1% | 0.6% | 5.7% | -0.7% | 6.2% |
| B|K5|W3|1x | — | 9.0% | -0.0% | 1.5% | 5.0% | -1.0% | 6.1% |
| B|K8|W7|1x | — | 9.6% | -0.0% | 1.7% | 4.4% | -0.4% | 6.1% |
| B|K8|W3|1x | — | 8.9% | -0.0% | 1.5% | 4.9% | -0.6% | 6.3% |
| B|K3|W7|3x | — | 29.1% | -0.1% | -9.3% | -1.0% | 0.7% | 9.7% |
| B|K3|W3|3x | — | 12.6% | -0.1% | -2.3% | 3.1% | -0.7% | 5.4% |
| B|K5|W7|3x | — | 20.9% | -0.1% | -3.4% | 12.3% | 0.4% | 9.1% |
| B|K5|W3|3x | — | 8.1% | -0.0% | -0.4% | 4.5% | -0.6% | 5.8% |
| B|K8|W7|3x | — | 2.6% | -0.1% | -4.0% | 5.2% | 0.3% | 4.2% |
| B|K8|W3|3x | — | 5.0% | -0.0% | 0.4% | 5.0% | -0.4% | 5.6% |

### Plato
- B ana aday `B|K5|W7|1x` (Sharpe 3.24): oran **0.87**; komşular: B|margin×0.5 3.38, B|margin×1.5 2.96, B|exit×0.5 3.26, B|exit×1.5 3.07, B|hold×0.5 2.82, B|hold×1.5 3.21.
- C ana aday `C|BTC|vade|1x` (Sharpe 1.14): oran **1.02**; komşular: C|margin×0.5 1.28, C|margin×1.5 1.16.
- A: ayarlanabilir parametre yok; uygulanamaz (geçti sayılır).

### Marj tamponu (tasfiye eşiği ≈ marj oranı − %1)
| Varlık/kaldıraç | En kötü günlük yükseliş | Hedef marjda | Bant altında (%50) | +%30 hedefte | +%30 bant altında |
|---|---|---|---|---|---|
| BTCUSDT|1x | 36.2% (2021-07-26) | yeter | yeter | yeter | yeter |
| BTCUSDT|2x | 36.2% (2021-07-26) | yeter | YETMEZ | yeter | YETMEZ |
| BTCUSDT|3x | 36.2% (2021-07-26) | YETMEZ | YETMEZ | yeter | YETMEZ |
| ETHUSDT|1x | 27.6% (2021-05-24) | yeter | yeter | yeter | yeter |
| ETHUSDT|2x | 27.6% (2021-05-24) | yeter | YETMEZ | yeter | YETMEZ |
| ETHUSDT|3x | 27.6% (2021-05-24) | yeter | YETMEZ | yeter | YETMEZ |
| BTCUSDT|PM (günlük basis değişimi vs %10 tampon) | 0.2% (2021-04-18) | yeter | yeter | yeter | yeter |
| ETHUSDT|PM (günlük basis değişimi vs %10 tampon) | 0.2% (2021-05-20) | yeter | yeter | yeter | yeter |

### Stres dönemleri (maks. DD / toparlanma günü)
| Seri | 2020-03 | 2021-05 | 2022-05 (LUNA) | 2022-11 (FTX) |
|---|---|---|---|---|
| A|BTC|1x | değerlendirilemez | -0.1% / 6 | -0.1% / 9 | -0.2% / 40 |
| C|BTC|vade|1x | değerlendirilemez | -1.0% / 4 | -0.1% / 620 | -0.1% / 436 |
| B|K5|W7|1x | değerlendirilemez | -0.6% / 96 | -0.1% / 586 | -0.1% / 402 |
| BTC al-tut | değerlendirilemez | -45.6% / 143 | -57.7% / 647 | -76.7% / 469 |

### Delist ve kuyruk riski
- B varyantlarında delist edilen sembolde (bitişten önceki 30 gün) açık carry pozisyonu: 24 kayıt: B|K3|W7|1x MATICUSDT 2024-09-05→2024-09-16 PnL -0.56 USDT; B|K3|W7|1x BNXUSDT 2025-03-12→2025-03-24 PnL -3.81 USDT; B|K3|W3|1x MATICUSDT 2024-09-05→2024-09-16 PnL -0.56 USDT; B|K3|W3|1x BNXUSDT 2025-03-11→2025-03-24 PnL -3.59 USDT; B|K5|W7|1x MATICUSDT 2024-09-05→2024-09-16 PnL -0.34 USDT; B|K5|W7|1x BNXUSDT 2025-03-12→2025-03-24 PnL -2.30 USDT; B|K5|W3|1x MATICUSDT 2024-09-05→2024-09-16 PnL -0.34 USDT; B|K5|W3|1x BNXUSDT 2025-03-11→2025-03-24 PnL -2.13 USDT; B|K8|W7|1x MATICUSDT 2024-09-05→2024-09-16 PnL -0.21 USDT; B|K8|W7|1x BNXUSDT 2025-03-12→2025-03-24 PnL -1.43 USDT; B|K8|W3|1x MATICUSDT 2024-09-05→2024-09-16 PnL -0.21 USDT; B|K8|W3|1x BNXUSDT 2025-03-11→2025-03-24 PnL -1.34 USDT.
- Funding tavanı yakınlığı (tutulan pozisyonda |oran| ≥ %0,3/olay): tablo sütunu yok; `ozet.json` → `funding_cap_events`.
- ADL simüle edilemez (veri yok).
- BTC al-tut (pencere): Sharpe 0.54, yıllık 15.9%; risksiz %4 kıyası (D) yıllık 4.0%.
- Limiti bilinmeyen (delist) semboller kısıtsız varsayıldı: 16 sembol.
> **Not (carry_001)** — kayıtlar değiştirilmez; notlar karar DEĞİŞTİRMEZ.
>
> **Karar neden sağlam:** Tüm 28 varyant "NAV'a göre yıllık getiri ≥ %8" koşulunda kalıyor. En yüksek: A|BTC|PM %7,2 (tasfiye yok), A|ETH|PM
> %7,0, A|ETH|3x %6,6. Tasfiye kayıpları tamamen yok sayılsa bile hiçbir varyant %8'e ulaşmıyor (A|BTC|3x ≈ %6, B|K5|W7|3x'te tasfiyenin
> net etkisi zaten pozitif). Funding geliri yıllara göre hızla düşüyor: A|BTC|1x 2021 %18,3 → 2025 %2,0.
>
> **Gözlemler (bilgi):**
> - A (sürekli carry) çok düşük oynaklıkla çalışıyor (Sharpe 7–10, maks. DD < %1 at 1x) ama getirisi risksiz faizin yaklaşık 1–1,8 katı;
>   hipotezdeki "belirgin üstünde" iddiası tutmuyor. A|BTC|1x ve A|BTC|PM yalnızca getiri koşulunda kalıyor.
> - Boştaki nakit %4 kazansaydı (bilgi; ön kayıtta karar dışı): A|BTC|PM %8,0, B|K3|W7|3x %9,7, B|K5|W7|3x %9,1. Bu sürümler karar
>   ölçütü değildir.
> - DSR: aynı deneydeki A varyantlarının çok yüksek Sharpe'ları deneme Sharpe varyansını büyütüyor; B ve C varyantlarında DSR bu yüzden
>   ~0. Karar bundan bağımsızdır (getiri koşulu herkes için bağlayıcı).
> - C'de sermaye zamanın ~%70'inde boşta (basis eşiği nadiren aşılıyor); bağlanan sermayeye göre %6–12.
> - B'de bağlanan sermayeye göre %8–13 (1x), ama sermaye verimliliği ~%27; NAV getirisi ~%3.
>
> **Veri ve modelleme notları:**
> - Tasfiye, ön kayıt gereği perp'in gün içi SON İŞLEM en yüksek fiyatıyla kontrol edildi. Binance tasfiyeleri mark fiyatıyla tetikler;
>   bu yüzden yöntem kısa iğnelerde kötümserdir. Örnek: 2021-07-26 BTCUSDT perp 48.168,6 (01:00 mumu; spot en yüksek 40.550) — A|BTC|2x
>   ve 3x'teki tek tasfiye budur. 3x B varyantlarındaki çok sayıdaki tasfiyenin bir kısmı da benzer perp iğneleri olabilir (ayrıştırılmadı).
> - ETHUSDT perp 2020-03-13 en yüksek 323 (spot 139,7): veri/iğne kaynaklı; A|ETH|1x'in 2020 yılı −%27,9 bu sahte tasfiyeden. 2020 yalnızca
>   bilgi; pencere dışı.
> - Koşu sonrası bu iki tarihe bakmak için kayıtlı varyantlar yeniden koşturuldu (A|ETH|1x 2020'den); yeni varyant/parametre denenmedi.

## ML turnuvası — KOŞTURULMADI (2026-10-08; deneme sayılmaz)
Kullanıcı talimatı (3 soru: volatilite tahmini, kesitsel sıralama, trend meta-label) kabul edilmiş kolların varlığına dayanıyor:
"ML yalnızca kabul edilmiş bir kolun net performansını ... iyileştirirse portföye girer." Bu tarihte kabul edilmiş kol YOK
(trend_001, kesitsel_001, carry_001: KALDI).
- Soru 2 talimatın kendi koşuluyla uygulanamaz ("yalnızca kesitsel kol kabul edildiyse").
- Soru 3: filtrelenecek kabul edilmiş trend kolu yok; başarısız bir kolu meta-label ile kurtarmak ilkeye aykırı (aşırı uyum riski).
- Soru 1: tahmin kısmı yapılabilir ama ikinci başarı ölçütü (kolun vol hedeflemesinde net Sharpe artışı) kol gerektirir; ML portföye
  giremez.
- Portföye giremeyecek deneyler global deneme sayacını (321) büyütüp gelecekteki kolların Deflated Sharpe eşiğini zorlaştıracağından
  **kullanıcı kararıyla (2026-10-08) koşturulmadı.** Tasarım `docs/research/sablonlar/ml_turnuva.md`'de bekliyor; bir kol kabul edilince
  her soru için config listesi ön kayıt olarak yazılıp uygulanır.

## Portföy ve nihai test — KOŞTURULMADI (2026-10-08; deneme sayılmaz, nihai pencere AÇILMADI)
Kullanıcı talimatı (kabul edilen kolların portföyü → aday_v1 kilidi → nihai pencerede tek seferlik test) kabul edilmiş kolların
varlığına dayanıyor. Bu tarihte kabul edilmiş kol YOK (trend_001, kesitsel_001, carry_001: KALDI).
- Başarısız kollardan portföy kurmak sonuca göre seçim olur (KURALLAR §7).
- Nihai pencere (KURALLAR §1) tek kullanımlıktır; kabul edilmemiş bir adayla açılması onu gelecekteki gerçek adaylar için tüketirdi.
- **Kullanıcı kararıyla (2026-10-08) koşturulmadı; nihai pencere KİLİTLİ ve KULLANILMAMIŞ.** Tasarım
  `docs/research/sablonlar/portfoy_nihai_test.md`'de bekliyor.

## Portföy paper modu — UYGULANMADI (2026-10-08)
Talimat "aday_v1 nihai testi geçti" varsayımıyla geldi; `research/configs/aday_v1.yaml` yok ve nihai test yapılmadı (kabul edilmiş kol
yok). Kullanıcı kararıyla (2026-10-08) kod yazılmadı; saatlik ML motoru varsayılan olarak kalıyor. Tasarım
`docs/research/sablonlar/portfoy_paper.md`'de bekliyor.

## ÖN KAYIT — `ml_kol` (2026-10-08, sonuç görülmeden önce; dal `research/ml-kol`)
**Kullanıcı kararları (2026-10-08):** (1) ML'yi tek başına bir kol adayı olarak sınamak için TEK bir ön kayıtlı soru açılır; ML turnuvası
şablonundaki "ML yalnızca kabul edilmiş kolu büyütür" ilkesine bu soru için açık istisna. Kabul eşikleri aynı (KURALLAR §5). (2) Tahmin
hedefi düşüş/nötr/yükseliş sınıfı değil, **maliyet sonrası beklenen getiri**.

**Hipotez:** Günlük çok varlıklı veride, vol'e göre ölçeklenmiş 7 günlük ileri getiriyi tahmin eden bir model, yalnızca beklenen
getirinin işlem maliyetini aştığı sembollerde pozisyon alarak maliyetler sonrası kol eşiklerini geçen getiri üretir.

**Veri, evren, doğrulama.** Noktasal-zamanlı ilk-30 perpetual; dönem 2021-01-01 → 2025-09-30 (nihai pencere kapalı). Koşucu
(`research.runner`): tarih bazlı purge + embargo'lu walk-forward, 5 katman, embargo = h, `min_train_days` 365, tohum 0. Sızıntı testleri
(karıştırılmış hedef, `available_at`, +1 gün gecikme uyarısı, tekrarlanabilirlik) açık. Hesap: 250 USDT, 3x, en yakın lot, Binance limitleri.

**Hedef.** `vol_adj_return(h)` = ileri h gün açılıştan açılışa getiri / (t'ye kadar bilinen 20 günlük günlük vol × √h). Birincil h = 7; ek h = 3.

**Kaynak kademeleri.** a: `ohlcv_core`; b: a + `funding` + `taker_flow`; d: b + `macro` + `sentiment_fng`. (`derivatives_metrics` yalnızca
BTC/ETH'yi kapsadığı için evrende kullanılamaz; dışarıda.)

**Modeller.** `ridge` (doğrusal kıyas), `lightgbm_reg`, `xgboost_reg` (çerçevenin varsayılan parametreleri).

**Sinyal adaptörü `edge_threshold` (yeni; karar t kapanışında, işlem t+1 açılışında):**
- p = model tahmini (σ birimi). Maliyet eşiği c_i = 2 × (taker ücreti + sembolün kayma kademesi) / (σ_i · √h); σ_i = t'ye kadar bilinen
  20 günlük günlük vol (hedefteki ile aynı).
- |p| > k · c_i ise yön = işaret(p) (long ya da short), değilse 0. Sabit **k = 1,0**.
- Ham ağırlık ∝ işaret(p) · |p| / σ_i. Kol hedef vol yıllık **%20** (ham portföyün geçmiş günlük getirisinin 60 g EWMA vol'ü; yalnızca ≤ t).
- İsim tavanı `sizing.asset_caps(base_cap = 0,10)` (BTC için en küçük lota göre yükseltilir, KURALLAR §3); brüt tavan **2,0x**.
- Yürütme: motorda göreli işlem bandı **%25** (|hedef − mevcut| ≤ 0,25·|hedef| ise işlem yok; çıkış her zaman). Hesap kaldıracı ayrıca 3x.

**Varyantlar (soru `ml_kol`; bütçe 40; bu deney 18):**
- Temel: 3 model × kademe {a, b, d} × h = 7 → **9**.
- Ek ufuk: 3 model × kademe b × h = 3 → **3**.
- Plato (ana aday ÖNCEDEN: `lightgbm_reg`, kademe b, h = 7): k ×0,5 / ×1,5; lightgbm `num_leaves` (varsayılan) ×0,5 / ×1,5;
  `learning_rate` ×0,5 / ×1,5 → **6**. Plato oranı = en kötü komşu Sharpe / ana aday Sharpe ≥ 0,70; plato sonucu tüm varyantlara uygulanır.
- PBO: 12 temel varyant (9 + 3), CSCV 16 blok. DSR: global toplam deneme sayısı ve 18 varyantın Sharpe varyansı.

**Karar kuralı.** Bir varyant **"ML kolu adayı"**dır ancak: (1) tüm sızıntı testleri geçti; (2) KURALLAR §5 kol eşiklerinin TÜMÜ (net Sharpe
≥ 0,8; DSR ≥ 0,95; PBO ≤ 0,25; maks. DD ≤ %35; Calmar ≥ max(BTC al-tut Calmar, 0,7); pozitif yıl ≥ %60; ücret ×2 + kayma ×3 Sharpe ≥ 0,5;
plato ≥ 0,70) geçti; (3) model `ridge` değilse, aynı kademe ve ufuktaki `ridge`'e karşı günlük net getiri Sharpe farkının eşleştirilmiş
durağan blok bootstrap %95 aralığı 0'ın üstünde. Birden çok geçen varsa EN BASİT olan aday ilan edilir: ridge → lightgbm → xgboost; az
kaynak önce (a → b → d); h = 7 önce. Hiçbiri geçmezse DURULUR ve nedeni yazılır. Kazanan varsa config'i
`backend/research/configs/ml_kol_kazanan.yaml` olarak kaydedilir.

**Raporlanan ek bilgiler (karar dışı):** günlük rank IC ve t (her varyant), kaynak kademelerinin marjinal katkısı (a→b, b→d; IC ve net Sharpe
farkı, eşleştirilmiş bootstrap aralığı), turnover ve maliyet/brüt oranı, trend_001 main_LF/LS serileriyle korelasyon.
