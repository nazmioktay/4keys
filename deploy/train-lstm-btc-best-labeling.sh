#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LSTM'i YALNIZCA BTC/USDT uzerinde, etiketleme taramasinin
# (sweep-labeling-lstm-btc.sh) bulguna gore EN IYI kombinasyonla
# (horizon=3, threshold_pct=1.0 -> out_of_sample_balanced_accuracy ~%44.1,
# rastgele seviye %33) KALICI (persist=True, varsayilan) olarak egitir.
#
# Varsayilan horizon=5, threshold_pct=1.0 ile ayni tarama ~%37 veriyordu;
# bu ayarlar LSTM'i XGBoost'un ulastigi %41-44 seviyesine tasiyor.
# ============================================================

echo "LSTM egitimi (BTC/USDT, horizon=3, threshold_pct=1.0 - en iyi etiketleme) baslatiliyor..."
curl -sS -X POST http://127.0.0.1:8000/ml/train-lstm \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["BTC/USDT:USDT"], "horizon": 3, "threshold_pct": 1.0}'
echo
echo "Tamamlandi."
