#!/usr/bin/env bash
set -euo pipefail

echo "XGBoost (Faz A) egitimi baslatiliyor..."
curl -sS -X POST http://127.0.0.1:8000/ml/train \
  -H "Content-Type: application/json" \
  -d '{}'
echo
echo "Tamamlandi."
