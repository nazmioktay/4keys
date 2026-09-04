#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# XGBoost'u (Faz A) varsayilan horizon=5/threshold_pct=1.0 yerine,
# LSTM etiketleme taramasinin (sweep-labeling-lstm-btc.sh) bulgusuna
# gore EN IYI kombinasyonla (horizon=3, threshold_pct=1.0) egitir.
#
# Varsayilan etiketlemeyle /ml/train, BTC-only veride out_of_sample_
# balanced_accuracy'nin tam olarak 1/3'e (rastgele seviye) oturdugunu
# gosterdi — sinif agirliklandirma bunu DUZELTMEDI, cunku kok neden
# sinif dengesizligi degil, zayif etiketleme kalibrasyonuydu (LSTM'de
# de ayni sorun ayni sekilde cozulmustu).
# ============================================================

curl -sS -X POST http://127.0.0.1:8000/ml/train \
  -H "Content-Type: application/json" \
  -d '{"horizon": 3, "threshold_pct": 1.0}'
echo
echo "Tamamlandi."
