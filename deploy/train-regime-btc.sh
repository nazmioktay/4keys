#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Hibrit rejim+ML: BTC/USDT'yi (n_regimes=3, varsayilan) GMM ile
# volatilite/trend uzayinda rejimlere ayirir, HER REJIM ICIN AYRI bir
# XGBoost modeli egitir ve out-of-sample sonuclarini karsilastirir.
#
# UYARI: canli karar motoruna HENUZ baglanmadi — bu, "rejime ayirmak
# tek global modelden daha mi iyi?" sorusuna cevap vermek icin offline
# bir karsilastirmadir. persist=True (varsayilan) oldugu icin
# model_regime_0/1/2.joblib ve regime_model.joblib diske kaydedilir,
# ama hicbir canli tahmin/karar akisi bunlari henuz kullanmiyor.
# ============================================================

curl -sS -X POST http://127.0.0.1:8000/ml/train-regime \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["BTC/USDT:USDT"], "n_regimes": 3, "walk_forward_splits": 3}'
echo
echo "Tamamlandi. Her rejimin out_of_sample_balanced_accuracy'sini, mevcut global XGBoost modelinin sonucuyla (bkz. train-xgboost.sh) karsilastir."
