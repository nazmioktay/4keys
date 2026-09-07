#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# train-all.sh'i LSTM DAHIL calistiran kisayol - konsol harf
# buyuk/kucuk sorunu yuzunden "INCLUDE_LSTM=1 bash deploy/train-all.sh"
# gibi buyuk harfli bir ortam degiskeni yazmak zorunda kalmamak icin
# TEK, tamamen kucuk harfli bir dosya adi olarak eklendi.
#
# Calistirma (sunucuda, root olarak, /opt/4keys icinde):
#   bash deploy/train-all-with-lstm.sh
# ============================================================

export INCLUDE_LSTM=1
exec bash "$(dirname "$0")/train-all.sh"
