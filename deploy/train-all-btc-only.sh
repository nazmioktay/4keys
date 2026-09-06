#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# deploy/train-all.sh ile AYNI, ama YALNIZCA BTC/USDT:USDT ile - coklu-sembol
# veri hazirlamanin (BTC + ml_train_max_symbols kadar korelasyonlu digerleri,
# her biri icin ~10.000 mumluk/~50+ ozellikli bir tablo) bellek zirvesinin
# OOM'un asil nedeni olup olmadigini test etmek icin (bkz. README "OOM
# uretim olayi" - egitim konteyneri Grafana/Prometheus durdurulsa bile
# tekrarlanan bir noktada (~1GB) olduruluyordu, bu tek-sembol testi kaynagi
# netlestirmek icin eklendi).
#
# Calistirma (sunucuda, root olarak, /opt/4keys icinde): bash deploy/train-all-btc-only.sh
# ============================================================

exec "$(dirname "$0")/train-all.sh" "BTC/USDT:USDT"
