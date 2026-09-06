#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Guven esigi taramasi: farkli open_confidence degerleriyle ard arda
# backtest calistirip her biri icin islem sayisi/kazanma orani/PnL/max
# drawdown doner. Egitim YAPMAZ, model degistirmez — yalnizca AYNI
# egitilmis modelle mevcut backtest'i birden fazla esikte tekrar oynatir.
# Ara denemeler backtest_runs tablosuna/Grafana panellerine YAZILMAZ.
#
# Otomatik "en iyi" esigi SECMEZ: az islemle gorulen yuksek bir kazanma
# orani, cok islemle gorulen daha dusuk bir orandan DAHA GUVENILIR
# degildir — karar operatore kalir. Denenecek degerleri asagidan
# degistirebilirsin.
# ============================================================

response=$(curl -sS -X POST http://127.0.0.1:8000/backtest/system/sweep-confidence \
  -H "Content-Type: application/json" \
  -d '{"open_confidence_values": [0.5, 0.55, 0.6, 0.65, 0.7]}')

echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
echo
echo "Tamamlandi. Her nokta icin trades_closed / win_rate_pct / total_pnl_pct / max_drawdown_pct karsilastir."
