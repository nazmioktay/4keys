#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LSTM'i YALNIZCA BTC/USDT üzerinde eğitir ve test eder — çoklu sembol
# (BTC-öncelikli + uyumlu semboller) eğitimi değil, saf tek-sembol
# testi. `train-lstm.sh` varsayılan (çoklu sembol) davranışı içindir;
# bu script kullanıcı isteğiyle eklendi: "LSTM trainingi yalnızca
# BTCUSDT için gerçekleştirip, BTCUSDT çifti için sonuçlarını test
# edelim".
# ============================================================

echo "LSTM egitimi (yalnizca BTC/USDT) baslatiliyor (bu birkac dakika surebilir)..."
curl -sS -X POST http://127.0.0.1:8000/ml/train-lstm \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["BTC/USDT:USDT"]}'
echo
echo "Tamamlandi."
