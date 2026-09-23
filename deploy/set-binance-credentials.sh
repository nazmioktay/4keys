#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# `.env`e Binance API anahtarlarini (FOURKEYS_BINANCE_API_KEY/SECRET)
# GUVENLI sekilde ekler/gunceller — `set-auth-credentials.sh` ile AYNI
# desen: ozel karakterleri (alt cizgi, yonlendirme) kullanicinin elle
# yazmasina GEREK BIRAKMAZ, sadece anahtar/secret degerlerini argument
# olarak alir.
#
# Kullanim (backend/ dizininde, /opt/4keys/backend):
#   bash ../deploy/set-binance-credentials.sh API_ANAHTARI API_SECRET
# ============================================================

if [ "$#" -ne 2 ]; then
  echo "Kullanim: bash set-binance-credentials.sh API_ANAHTARI API_SECRET"
  exit 1
fi

API_KEY="$1"
API_SECRET="$2"
ENV_FILE=".env"

if [ ! -f "$ENV_FILE" ]; then
  echo "HATA: $ENV_FILE bulunamadi. Bu script'i backend/ dizininde calistirin (/opt/4keys/backend)."
  exit 1
fi

grep -v -E "^FOURKEYS[_-]BINANCE[_-]API[_-](KEY|SECRET)=" "$ENV_FILE" > "$ENV_FILE.tmp" || true
mv "$ENV_FILE.tmp" "$ENV_FILE"

{
  printf 'FOURKEYS_BINANCE_API_KEY=%s\n' "$API_KEY"
  printf 'FOURKEYS_BINANCE_API_SECRET=%s\n' "$API_SECRET"
} >> "$ENV_FILE"

echo "Tamamlandi. Eklenen satirlar (secret degeri gizlendi):"
grep "^FOURKEYS_BINANCE_API_KEY=" "$ENV_FILE"
echo "FOURKEYS_BINANCE_API_SECRET=***gizli***"
echo
echo "Simdi backend'i yeniden baslatin (repo KOKUNDEN /opt/4keys):"
echo "  cd .. && bash deploy/recreate-backend.sh"
