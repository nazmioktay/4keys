#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# PatchTST'ten esinlenilmis, patch-tabanli Transformer siniflandiriciyi
# YALNIZCA BTC/USDT uzerinde egitip test eder — LSTM'in BTC-only
# sinamalarda hem lookback artirma (10K->20K) hem model kucultme ile
# ~%38-39 balanced_accuracy tavanina takili kalmasi uzerine, farkli bir
# mimarinin (dikkat mekanizmasi) daha fazlasini cikarip cikaramayacagini
# test etmek icin eklendi.
#
# Kullanim:
#   bash train-patchtst-btc.sh                # LOOKBACK=10000 (varsayilan)
#   LOOKBACK=20000 bash train-patchtst-btc.sh
# ============================================================

LOOKBACK="${LOOKBACK:-10000}"

echo "PatchTST egitimi (BTC/USDT, lookback=$LOOKBACK) baslatiliyor (bu birkac dakika surebilir)..."
curl -sS -X POST http://127.0.0.1:8000/ml/train-patchtst \
  -H "Content-Type: application/json" \
  -d "{\"symbols\": [\"BTC/USDT:USDT\"], \"lookback\": $LOOKBACK}"
echo
echo "Tamamlandi."
