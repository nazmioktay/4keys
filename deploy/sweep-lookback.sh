#!/usr/bin/env bash
set -euo pipefail

echo "Lookback taramasi baslatiliyor (bu UZUN surebilir, birden fazla egitim yapilacak)..."
curl -sS -X POST http://127.0.0.1:8000/ml/sweep-lookback \
  -H "Content-Type: application/json" \
  -d '{"lookback_values": [2000, 4000, 6000, 8000, 10000]}'
echo
echo "Tamamlandi."
