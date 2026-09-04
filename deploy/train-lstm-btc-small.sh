#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LSTM'i YALNIZCA BTC/USDT uzerinde, KUCUK model kapasitesiyle
# (hidden_size=32, num_layers=1) egitip test eder.
#
# Neden: BTC-only sinama sinif agirlikli LSTM ile balanced_accuracy'yi
# rastgele seviyeden (%33) %38.7'ye cikardi ama lookback'i artirmak
# (10K->20K) iyilesme saglamadi — sinirlayici faktorun veri miktari
# degil, veri boyutuna gore FAZLA BUYUK model kapasitesi olabilecegi
# hipotezini test ediyoruz. Varsayilan (hidden_size=64, num_layers=2)
# ile karsilastir.
#
# Kullanim:
#   bash train-lstm-btc-small.sh                # LOOKBACK=10000 (varsayilan)
#   LOOKBACK=20000 bash train-lstm-btc-small.sh
# ============================================================

LOOKBACK="${LOOKBACK:-10000}"

echo "LSTM egitimi (BTC/USDT, lookback=$LOOKBACK, KUCUK model: hidden_size=32 num_layers=1) baslatiliyor..."
curl -sS -X POST http://127.0.0.1:8000/ml/train-lstm \
  -H "Content-Type: application/json" \
  -d "{\"symbols\": [\"BTC/USDT:USDT\"], \"lookback\": $LOOKBACK, \"hidden_size\": 32, \"num_layers\": 1}"
echo
echo "Tamamlandi."
