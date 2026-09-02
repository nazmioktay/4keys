#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 4keys — FRED (ABD Merkez Bankası) API anahtarını backend/.env'e
# ekler/günceller ve backend'i yeniden başlatır.
#
# Çalıştırma (sunucuda, root olarak):
#   curl -fsSL raw.githubusercontent.com/nazmioktay/4keys/main/deploy/set-fred-key.sh -o set-fred-key.sh
#   bash set-fred-key.sh <FRED_API_KEY>
# ============================================================

APP_DIR="/opt/4keys"
ENV_FILE="$APP_DIR/backend/.env"
KEY="${1:-}"

if [ -z "$KEY" ]; then
  echo "Kullanim: bash set-fred-key.sh <FRED_API_KEY>"
  exit 1
fi

if grep -q "^FOURKEYS_FRED_API_KEY=" "$ENV_FILE" 2>/dev/null; then
  sed -i "s#^FOURKEYS_FRED_API_KEY=.*#FOURKEYS_FRED_API_KEY=${KEY}#" "$ENV_FILE"
else
  echo "FOURKEYS_FRED_API_KEY=${KEY}" >> "$ENV_FILE"
fi

docker restart 4keys-backend

echo "Tamamlandi: FRED API key eklendi, backend yeniden baslatildi."
