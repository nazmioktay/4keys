#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Hibrit rejim+ML: BTC-oncelikli + uyumlu semboller (symbols=None ->
# select_training_symbols, bkz. app/ml/symbol_selection.py) uzerinde
# GMM ile volatilite/trend uzayinda rejimlere ayirir, HER REJIM ICIN
# AYRI bir XGBoost modeli egitir ve out-of-sample sonuclarini
# karsilastirir.
#
# ILK IKI DENEME de (once BTC-only, sonra coklu-sembol ama varsayilan
# horizon=5/threshold_pct=1.0 ile) balanced_accuracy'nin ~1/3'e
# (rastgele seviye) yakin kaldigini gosterdi. Sinif agirliklandirma ve
# coklu-sembol duzeltmeleri (bkz. git gecmisi) kok nedeni COZMEDI —
# asil sorun LSTM'de de bulunan AYNI sey: zayif etiketleme kalibrasyonu
# (horizon=5/threshold_pct=1.0). Bu script artik LSTM etiketleme
# taramasinin (sweep-labeling-lstm-btc.sh) buldugu EN IYI kombinasyonu
# (horizon=3, threshold_pct=1.0) kullaniyor.
#
# UYARI: canli karar motoruna HENUZ baglanmadi — bu, "rejime ayirmak
# tek global modelden daha mi iyi?" sorusuna cevap vermek icin offline
# bir karsilastirmadir. persist=True (varsayilan) oldugu icin
# model_regime_0/1/2.joblib ve regime_model.joblib diske kaydedilir,
# ama hicbir canli tahmin/karar akisi bunlari henuz kullanmiyor.
# ============================================================

curl -sS -X POST http://127.0.0.1:8000/ml/train-regime \
  -H "Content-Type: application/json" \
  -d '{"n_regimes": 3, "walk_forward_splits": 3, "horizon": 3, "threshold_pct": 1.0}'
echo
echo "Tamamlandi. Her rejimin out_of_sample_balanced_accuracy'sini, mevcut global XGBoost modelinin sonucuyla (bkz. train-xgboost.sh) karsilastir."
