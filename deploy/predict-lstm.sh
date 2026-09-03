#!/usr/bin/env bash
set -euo pipefail

SYMBOL="BTC/USDT:USDT"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "$SYMBOL")

echo "LSTM tahmini isteniyor (symbol=$SYMBOL)..."
curl -sS "http://127.0.0.1:8000/ml/predict-lstm?symbol=${ENCODED}"
echo
echo "Tamamlandi."
