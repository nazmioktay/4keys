#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Binance CANLI (gercek para) islemini acar — iki guvenlik kapisini
# birden `.env`de ayarlar:
#   FOURKEYS_BINANCE_TESTNET=false   (ccxt artik futures testnet/sandbox
#     modunu desteklemiyor — bkz. https://t.me/ccxt_announcements/92 —
#     bu yuzden gercek hesaba baglanmak GEREKIYOR)
#   FOURKEYS_ENABLE_LIVE_TRADING=true (ikinci kapi: bu acilmadan HICBIR
#     emir gonderilmez, sadece bakiye/pozisyon OKUNUR)
#
# ONEMLI: Bu script emir gondermeyi MUMKUN kilar, otomatik emir
# GONDERMEZ — frontend (Canli Islem ekrani) her emirde hala elle,
# iki adimli onay ister. Kaldirac tavani (MAX_LEVERAGE=3, app/security/
# safety.py) .env ile bile ASILAMAZ.
#
# Kullanim (backend/ dizininde, /opt/4keys/backend):
#   bash ../deploy/enable-live-trading.sh
# ============================================================

ENV_FILE=".env"

if [ ! -f "$ENV_FILE" ]; then
  echo "HATA: $ENV_FILE bulunamadi. Bu script'i backend/ dizininde calistirin (/opt/4keys/backend)."
  exit 1
fi

grep -v -E "^FOURKEYS[_-](BINANCE[_-]TESTNET|ENABLE[_-]LIVE[_-]TRADING)=" "$ENV_FILE" > "$ENV_FILE.tmp" || true
mv "$ENV_FILE.tmp" "$ENV_FILE"

{
  printf 'FOURKEYS_BINANCE_TESTNET=false\n'
  printf 'FOURKEYS_ENABLE_LIVE_TRADING=true\n'
} >> "$ENV_FILE"

echo "Tamamlandi. Eklenen satirlar:"
grep -E "^FOURKEYS_(BINANCE_TESTNET|ENABLE_LIVE_TRADING)=" "$ENV_FILE"
echo
echo "UYARI: artik GERCEK PARA ile emir gonderilebilir (frontend'in kendi elle onay adimina kadar)."
echo "Simdi backend'i yeniden baslatin (repo KOKUNDEN /opt/4keys):"
echo "  cd .. && bash deploy/recreate-backend.sh"
