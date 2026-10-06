# Veri kapsamı (Aşama 0 indirmesi)

Anlık görüntü hash: `9a3db8906f6a547f` · tam indirme: 2026-10-05T23:58 → 2026-10-06T01:31 (UTC), 3091 görev, 0 hata; ardından KLAYUSDT spot zaman damgası düzeltmesi sonrası yeniden indirme (105 görev, 0 hata)

Kaynak: data.binance.vision (aylık/günlük zip) + REST kuyruğu (yalnızca halen listeli semboller). Nihai pencere (>= 2025-10-01) önbellekte VAR ama yükleyiciler varsayılan olarak keser.

## Tür başına kapsam

| Tür | Sembol | İlk tarih | Son tarih |
|---|---|---|---|
| um_1d | 900 | 2020-01-01 | 2026-10-05 |
| um_1h | 289 | 2020-01-01 | 2026-10-06 |
| spot_1d | 474 | 2017-08-17 | 2026-10-05 |
| um_funding | 900 | 2020-01-01 | 2026-10-06 |
| delivery_um_1d | 50 | 2021-02-03 | 2026-10-04 |
| delivery_cm_1d | 52 | 2020-06-11 | 2026-10-04 |

## Perpetual evren (USDT-M günlük)

- Arşivdeki USDT-M perpetual sembol: 900; indirilen: 900
- Halen işlem gören (son bar <= 3 gün eski): 869
- **Delist edilmiş** (son veri 2026-10-05'den >3 gün önce): 31
- 1 saatlik veri indirilen sembol (hiç ilk-60 hacme girmiş): 289
- BTC/ETH teslimli: USDT-M 50 kontrat, COIN-M 52 kontrat

<details><summary>Delist edilmiş semboller (31)</summary>

1000BTTCUSDT, AERGOUSDT, AKROUSDT, ANCUSDT, ANTUSDT, AUDIOUSDT, BDXNUSDT, BLUEBIRDUSDT, BTCSTUSDT, BTSUSDT, BTTUSDT, BZRXUSDT, COCOSUSDT, DODOUSDT, DOTECOUSDT, EOSUSDT, FOOTBALLUSDT, FRONTUSDT, GALUSDT, HNTUSDT, KEEPUSDT, LENDUSDT, LUNAUSDT, MATICUSDT, MBLUSDT, NUUSDT, RNDRUSDT, SRMUSDT, SXPUSDT, TOMOUSDT, YFIIUSDT

</details>

## Bilinen veri kusurları (düzeltilmedi, tüketici bilmeli)

- **Yeniden adlandırmalar ayrı sembol görünür**: ör. MATICUSDT (delist) → POLUSDT, RNDRUSDT (delist) → RENDERUSDT. "Delist" listesi bunları da içerir; bir varlığın geçmişi iki sembole bölünür. Evren/ağırlık kodu bunu varlık düzeyinde birleştirmiyor.
- **Arşiv boşlukları**: 2022-02-26..28 ve 2022-04-01.. gibi bazı günler birçok sembolde eksik (Binance arşiv kesintileri). Doldurulmadı; motor eksik fiyatlı günde pozisyonu tutmaz ve raporlar.
- **Sıfır hacimli günler** (21.952) çoğunlukla düşük likiditeli/ölmekte olan sembollerde; evren fonksiyonu hacim ortalamasında zaten elenir.
- **Endeks perpetual'ları** (BLUEBIRDUSDT, FOOTBALLUSDT) delist listesinde görünür; evren fonksiyonu bunları dışlar.
- **Spot** serisi yalnızca perpetual'ın aynı adlı spot çifti için indirildi (474 sembol); bazı perpetual'ların spot karşılığı yok.
- **Zaman damgası birimi**: arşiv dosyaları saniye/ms/µs karışık olabilir (KLAYUSDT spot'ta görüldü); ayrıştırıcı değer bazında tespit eder ve olanaksız tarihte hata verir.

## Veri kalitesi (günlük, nihai pencere hariç)


- Sembol sayısı: 900
- İlk tarih: 2020-01-01 00:00:00  ·  Son tarih: 2025-09-30 00:00:00
- Eksik günü olan sembol: 52 (toplam 323 gün)
- Sıfır hacimli gün: 21952
- |log getiri| > 5σ sıçrama: 1181

Eksik günler ve sıçramalar DOLDURULMAZ/DÜZELTİLMEZ; tüketici (motor/evren) NaN'ı olduğu gibi görür.

| Sembol | İlk | Son | Satır | Eksik gün | Sıfır hacim | Sıçrama | Örnek eksik günler |
|---|---|---|---|---|---|---|---|
| TLMUSDT | 2021-07-16 | 2025-09-30 | 1509 | 29 | 264 | 5 | 2023-03-01, 2023-03-02, 2023-03-03, 2023-03-04, 2023-03-05,  |
| ICPUSDT | 2021-05-11 | 2025-09-30 | 1578 | 26 | 82 | 6 | 2022-09-01, 2022-09-02, 2022-09-03, 2022-09-04, 2022-09-05,  |
| BNXUSDT | 2022-04-01 | 2025-09-30 | 1253 | 26 | 197 | 2 | 2022-04-17, 2022-06-09, 2022-08-10, 2022-08-11, 2022-08-12,  |
| XRPUSDT | 2020-01-06 | 2025-09-30 | 2090 | 5 | 0 | 12 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| FILUSDT | 2020-10-16 | 2025-09-30 | 1806 | 5 | 0 | 10 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| WAVESUSDT | 2020-08-12 | 2025-09-30 | 1871 | 5 | 476 | 10 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| LRCUSDT | 2020-10-19 | 2025-09-30 | 1803 | 5 | 0 | 8 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| MASKUSDT | 2021-08-27 | 2025-09-30 | 1491 | 5 | 0 | 8 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| OMGUSDT | 2020-07-02 | 2025-09-30 | 1912 | 5 | 242 | 7 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| PEOPLEUSDT | 2021-12-24 | 2025-09-30 | 1372 | 5 | 0 | 7 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| SANDUSDT | 2021-01-25 | 2025-09-30 | 1705 | 5 | 0 | 7 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| SKLUSDT | 2020-12-08 | 2025-09-30 | 1753 | 5 | 0 | 7 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| STORJUSDT | 2020-09-16 | 2025-09-30 | 1836 | 5 | 0 | 7 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| TRBUSDT | 2020-09-03 | 2025-09-30 | 1849 | 5 | 0 | 7 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| TRXUSDT | 2020-01-15 | 2025-09-30 | 2081 | 5 | 0 | 7 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| UNFIUSDT | 2021-02-19 | 2025-09-30 | 1680 | 5 | 335 | 7 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| XEMUSDT | 2021-03-03 | 2025-09-30 | 1668 | 5 | 295 | 7 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| XLMUSDT | 2020-01-20 | 2025-09-30 | 2076 | 5 | 0 | 7 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| FTMUSDT | 2020-09-24 | 2025-09-30 | 1828 | 5 | 267 | 6 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| GRTUSDT | 2020-12-19 | 2025-09-30 | 1742 | 5 | 0 | 6 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| HOTUSDT | 2021-03-30 | 2025-09-30 | 1641 | 5 | 0 | 6 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| KNCUSDT | 2020-06-22 | 2025-09-30 | 1922 | 5 | 0 | 6 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| MANAUSDT | 2021-03-15 | 2025-09-30 | 1656 | 5 | 0 | 6 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| MKRUSDT | 2020-08-13 | 2025-09-30 | 1870 | 5 | 22 | 6 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| OGNUSDT | 2021-04-01 | 2025-09-30 | 1639 | 5 | 0 | 6 | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 |
| … | | | | | | | (+343 sembol daha) |