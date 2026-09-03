#!/usr/bin/env bash
set -euo pipefail

echo "Meta-label modeli egitimi baslatiliyor (once XGBoost egitilmis olmali)..."
curl -sS -X POST http://127.0.0.1:8000/ml/train-meta \
  -H "Content-Type: application/json" \
  -d '{}'
echo
echo "Tamamlandi."
