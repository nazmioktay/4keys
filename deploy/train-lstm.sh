#!/usr/bin/env bash
set -euo pipefail

echo "LSTM egitimi baslatiliyor (bu birkac dakika surebilir)..."
curl -sS -X POST http://127.0.0.1:8000/ml/train-lstm \
  -H "Content-Type: application/json" \
  -d '{}'
echo
echo "Tamamlandi."
