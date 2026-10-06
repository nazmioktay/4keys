# Kaynak kapsamı

| Kaynak | Sürüm | Kapsam | Aralık | Satır | Sembol kapsamı | Karar satırı eksik | Hücre NaN | Yayın gecikmesi | forward_only |
|---|---|---|---|---|---|---|---|---|---|
| derivatives_metrics | 1 | symbol | 2020-09-01 → 2025-09-30 | 3256 | 2/182 (1%) | 99.1% | 99.2% | 0 days 00:00:00 | False |
| funding | 1 | symbol | 2020-01-01 → 2025-09-30 | 228552 | 182/182 (100%) | 40.2% | 40.2% | 0 days 00:00:00 | False |
| macro | 1 | market | 2019-12-02 → 2025-09-30 | 1815 | — | 0.0% | 34.3% | 0 days 01:00:00 | False |
| ohlcv_core | 1 | symbol | 2020-01-01 → 2025-09-30 | 229429 | 182/182 (100%) | 39.9% | 41.2% | 0 days 00:00:00 | False |
| sentiment_fng | 1 | market | 2019-12-02 → 2025-09-30 | 2129 | — | 0.0% | 0.0% | 0 days 06:00:00 | False |
| taker_flow | 1 | symbol | 2020-01-01 → 2025-09-30 | 224139 | 182/182 (100%) | 41.3% | 41.5% | 0 days 00:00:00 | False |

Üretim: 2026-10-06 23:41 UTC · `python -m research.coverage_report --n 20 --start 2020-01-01 --end 2025-09-30` · veri hash `0774e2086a790a0b`

## Nasıl okunur
- **Dönem** 2020-01-01 → 2025-09-30 (ana bölge; nihai pencere >= 2025-10-01 hariç). **Evren:** noktasal-zamanlı ilk-20; dönem boyunca evrene en az bir gün girmiş **182** aday sembol.
- **Karar satırı eksik:** (gün × aday sembol) karar ızgarasında o kaynaktan HİÇ özellik olmayan satır oranı (noktasal-zamanlı birleştirmeden sonra). Sembolün listelenmediği günler de ızgarada olduğu için oran, gerçek kullanılabilir eksiklikten yüksektir.
- **Hücre NaN:** aynı ızgarada tüm özellik hücrelerinin NaN oranı (uzun pencereli özelliklerin ısınması dahil).
- Piyasa-geneli kaynaklarda (`market`) sembol kapsamı anlamsızdır (—).

## Türev metrikleri arşivi
- Binance `metrics` arşivinde (S3 listesi) **900** sembol var; en erken gün 2020-09-01.
- Yerelde indirilmiş (`um_metrics_1d`): **2** sembol: BTCUSDT, ETHUSDT. Diğer semboller için `derivatives_metrics` özellikleri NaN'dır; genişletmek için `python -m research.data.download_metrics --symbols ...`.

## İleriye dönük (forward_only) kaynaklar
- `depth_bands`, `liquidations`, `oi_detail`: VPS veritabanında 2026-10-06'dan beri birikiyor; burada ölçülmez. Durum: `bash deploy/forward-durum.sh`. Kullanım kuralı: KURALLAR.md §10 (yalnızca `zone: forward`, keşif 2026-11-01 → 2027-11-01).
