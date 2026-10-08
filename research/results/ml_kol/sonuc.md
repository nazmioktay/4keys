## SONUÇ — `ml_kol` (ön kayıt: "ÖN KAYIT — ml_kol")

**Karar:** KALDI — hiçbir varyant tüm koşulları geçmedi. DUR (ızgara genişletilmez).

Dönem 2021-01-01 → 2025-09-30 (OOF) · deneme: 18 varyant (toplam sayaç 339) · **PBO (12 temel varyant): 0.36** · BTC al-tut Calmar 1.86

### Temel varyantlar (sadelik sırasıyla) — KURALLAR §5 kol eşikleri + sızıntı testleri + ridge'e karşı fark
| Varyant | Rank IC (t) | Sharpe | Calmar | Maks. DD | Yıllık | Poz. yıl | Turnover | Maliyet/Brüt | DSR | Stres SR | +1g SR | Ridge'e göre ΔSharpe [%95] | BTC korr. | Trend LF/LS | Durum |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ridge|a|h7 | -0.023 (-1.9) | 0.28 | 0.20 | -17.9% | 3.5% | 50.0% | 51.6 | 58.2% | 0.06 | -0.29 | 0.94 | — (kıyas) | -0.04 | 0.16 / 0.23 | KALDI (net_sharpe, deflated_sharpe, pbo, calmar, positive_years, stress, plateau) |
| ridge|b|h7 | -0.033 (-2.8) | 0.29 | 0.16 | -22.8% | 3.8% | 50.0% | 51.6 | 68.0% | 0.06 | -0.29 | 0.71 | — (kıyas) | 0.03 | 0.20 / 0.21 | KALDI (net_sharpe, deflated_sharpe, pbo, calmar, positive_years, stress, plateau) |
| ridge|b|h3 | -0.028 (-3.3) | 0.57 | 0.63 | -15.6% | 9.8% | 50.0% | 55.2 | 52.3% | 0.15 | -0.09 | 0.13 | — (kıyas) | -0.08 | 0.14 / 0.29 | KALDI (net_sharpe, deflated_sharpe, pbo, calmar, positive_years, stress, plateau) |
| ridge|d|h7 | -0.043 (-3.4) | -0.12 | -0.10 | -29.8% | -3.0% | 50.0% | 49.4 | 139.7% | 0.01 | -0.87 | -0.25 | — (kıyas) | -0.10 | -0.04 / 0.11 | KALDI (net_sharpe, deflated_sharpe, pbo, calmar, positive_years, stress, plateau) |
| lightgbm_reg|a|h7 | 0.013 (1.0) | 0.64 | 0.53 | -19.6% | 10.4% | 75.0% | 76.7 | 47.2% | 0.18 | -0.27 | 0.05 | 0.36 [-0.86, 1.55] | -0.12 | -0.04 / 0.01 | KALDI (net_sharpe, deflated_sharpe, pbo, calmar, stress, plateau, beats_ridge) |
| lightgbm_reg|b|h7 | 0.006 (0.4) | -0.04 | -0.07 | -31.8% | -2.1% | 50.0% | 69.9 | 107.7% | 0.02 | -0.78 | -0.23 | -0.33 [-1.54, 0.94] | -0.07 | 0.02 / 0.02 | KALDI (net_sharpe, deflated_sharpe, pbo, calmar, positive_years, stress, plateau, beats_ridge) |
| lightgbm_reg|b|h3 | 0.009 (1.1) | 1.10 | 1.13 | -17.4% | 19.5% | 100.0% | 92.1 | 39.4% | 0.46 | 0.04 | 0.21 | 0.54 [-0.78, 1.90] | -0.18 | -0.10 / 0.03 | KALDI (deflated_sharpe, pbo, calmar, stress, plateau, beats_ridge) |
| lightgbm_reg|d|h7 | 0.019 (1.5) | -0.39 | -0.19 | -37.2% | -7.0% | 50.0% | 49.2 | 2294.2% | 0.01 | -0.86 | -0.84 | -0.27 [-1.78, 1.07] | -0.08 | -0.04 / 0.05 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, positive_years, stress, plateau, beats_ridge) |
| xgboost_reg|a|h7 | 0.012 (0.9) | 0.52 | 0.35 | -22.1% | 7.8% | 50.0% | 73.9 | 52.4% | 0.13 | -0.68 | 0.22 | 0.24 [-1.00, 1.42] | -0.14 | -0.05 / 0.00 | KALDI (net_sharpe, deflated_sharpe, pbo, calmar, positive_years, stress, plateau, beats_ridge) |
| xgboost_reg|b|h7 | 0.003 (0.2) | 0.11 | 0.02 | -29.2% | 0.5% | 50.0% | 69.2 | 84.9% | 0.03 | -0.71 | -0.25 | -0.18 [-1.41, 1.15] | -0.06 | -0.02 / -0.05 | KALDI (net_sharpe, deflated_sharpe, pbo, calmar, positive_years, stress, plateau, beats_ridge) |
| xgboost_reg|b|h3 | 0.010 (1.2) | 0.78 | 0.58 | -22.4% | 12.9% | 75.0% | 89.8 | 47.4% | 0.25 | -0.13 | 0.40 | 0.21 [-1.11, 1.54] | -0.21 | -0.10 / 0.08 | KALDI (net_sharpe, deflated_sharpe, pbo, calmar, stress, plateau, beats_ridge) |
| xgboost_reg|d|h7 | -0.001 (-0.1) | -0.27 | -0.14 | -43.2% | -6.1% | 50.0% | 47.9 | 394.5% | 0.01 | -0.69 | -1.06 | -0.15 [-2.00, 1.41] | -0.07 | -0.00 / 0.07 | KALDI (net_sharpe, deflated_sharpe, pbo, max_drawdown, calmar, positive_years, stress, plateau, beats_ridge) |

### Plato
- Ana aday `lightgbm_reg|b|h7` (Sharpe -0.04): oran **—**; komşular: k×0.5 -0.05, k×1.5 -0.08, num_leaves×0.5 -0.05, num_leaves×1.5 0.26, learning_rate×0.5 0.11, learning_rate×1.5 -0.13. Plato sonucu tüm varyantlara uygulanır.

### Kaynak kademelerinin marjinal katkısı (h=7; eşleştirilmiş blok bootstrap %95; bilgi)
| Model | Adım | ΔSharpe [%95] | ΔRank IC [%95] |
|---|---|---|---|
| ridge | a→b | 0.01 [-0.42, 0.42] | -0.0100 [-0.0221, 0.0003] |
| ridge | b→d | -0.41 [-1.83, 0.95] | -0.0102 [-0.0259, 0.0056] |
| lightgbm_reg | a→b | -0.69 [-1.39, 0.00] | -0.0077 [-0.0282, 0.0095] |
| lightgbm_reg | b→d | -0.34 [-2.04, 1.17] | 0.0132 [-0.0166, 0.0411] |
| xgboost_reg | a→b | -0.41 [-1.04, 0.24] | -0.0094 [-0.0283, 0.0065] |
| xgboost_reg | b→d | -0.38 [-2.06, 1.00] | -0.0037 [-0.0339, 0.0260] |

Sızıntı testleri: hepsi geçti. Varyant ayrıntıları: `research/results/ml_kol/varyantlar.csv`; her config'in kendi raporu `research/results/ml_kol_*/`.
