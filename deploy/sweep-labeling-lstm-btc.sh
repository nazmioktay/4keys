#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LSTM icin farkli (horizon, threshold_pct) etiketleme kombinasyonlarini
# BTC/USDT uzerinde tarar. Neden: lookback artirma, model kucultme ve
# mimari degistirme (LSTM->PatchTST) HICBIRI ~%38-39 balanced_accuracy
# tavanini asamadi — dordunun de ayni noktada tikanmasi, sinirlayici
# faktorun etiketleme (sabit horizon=5/threshold_pct=1.0'in piyasa
# gurultusune gore kotu kalibre olmasi) olabilecegini gosteriyor.
#
# UYARI: varsayilan izgara 4x4=16 kombinasyon, her biri sifirdan egitim —
# birkac dakika surebilir. Production LSTM modelini DEGISTIRMEZ.
# ============================================================

echo "LSTM etiketleme taramasi (BTC/USDT) baslatiliyor (birkac dakika surebilir)..."
curl -sS -X POST http://127.0.0.1:8000/ml/sweep-labeling-lstm \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["BTC/USDT:USDT"]}'
echo
echo "Tamamlandi. out_of_sample_balanced_accuracy en yuksek olan (horizon, threshold_pct) ciftini karsilastir."
