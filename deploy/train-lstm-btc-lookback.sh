#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LSTM'i YALNIZCA BTC/USDT uzerinde, ozel bir lookback (mum sayisi)
# degeriyle egitip test eder. `train-lstm-btc.sh`'tan farki: lookback
# `settings.ml_train_lookback` varsayilanina degil, LOOKBACK ortam
# degiskenine (varsayilan 20000) gore ayarlanir.
#
# Hetzner konsolunda inline JSON icindeki tirnak karakterleri bozulabildigi
# icin (bkz. proje notlari), bu deger komut satirinda degil bu dosyada
# tanimlaniyor — sadece indirip calistirman yeterli.
#
# Kullanim:
#   bash train-lstm-btc-lookback.sh          # LOOKBACK=20000 (varsayilan)
#   LOOKBACK=30000 bash train-lstm-btc-lookback.sh
# ============================================================

LOOKBACK="${LOOKBACK:-20000}"

echo "LSTM egitimi (yalnizca BTC/USDT, lookback=$LOOKBACK) baslatiliyor (bu birkac dakika surebilir)..."
curl -sS -X POST http://127.0.0.1:8000/ml/train-lstm \
  -H "Content-Type: application/json" \
  -d "{\"symbols\": [\"BTC/USDT:USDT\"], \"lookback\": $LOOKBACK}"
echo
echo "Tamamlandi."
