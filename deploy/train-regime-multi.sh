#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Hibrit rejim+ML: BTC-oncelikli + uyumlu semboller (symbols=None ->
# select_training_symbols, bkz. app/ml/symbol_selection.py) uzerinde
# GMM ile volatilite/trend uzayinda rejimlere ayirir, HER REJIM ICIN
# AYRI bir XGBoost modeli egitir ve out-of-sample sonuclarini
# karsilastirir.
#
# ILK DENEME (yalnizca BTC-only, symbols=["BTC/USDT:USDT"]) her uc
# rejimde de balanced_accuracy'nin tam olarak 1/3'e (rastgele seviye)
# oturdugunu gosterdi — kucuk veri + siniflandirici sinif dengesizligi
# birlesince coktu. Iki duzeltme yapildi: (1) XGBoost'a (SignalModel)
# LSTM'dekiyle AYNI ters-frekans sinif agirliklandirmasi eklendi,
# (2) burada artik coklu-sembol (BTC + korele semboller) kullaniliyor
# ki rejim basina yeterli ornek olsun.
#
# UYARI: canli karar motoruna HENUZ baglanmadi — bu, "rejime ayirmak
# tek global modelden daha mi iyi?" sorusuna cevap vermek icin offline
# bir karsilastirmadir. persist=True (varsayilan) oldugu icin
# model_regime_0/1/2.joblib ve regime_model.joblib diske kaydedilir,
# ama hicbir canli tahmin/karar akisi bunlari henuz kullanmiyor.
# ============================================================

curl -sS -X POST http://127.0.0.1:8000/ml/train-regime \
  -H "Content-Type: application/json" \
  -d '{"n_regimes": 3, "walk_forward_splits": 3}'
echo
echo "Tamamlandi. Her rejimin out_of_sample_balanced_accuracy'sini, mevcut global XGBoost modelinin sonucuyla (bkz. train-xgboost.sh) karsilastir."
