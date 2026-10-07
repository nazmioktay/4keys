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
