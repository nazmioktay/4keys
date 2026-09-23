#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# `.env`e giris kontrolu (FOURKEYS_AUTH_*) satirlarini GUVENLI sekilde
# ekler/gunceller — bu script yazildi cunku bazi terminal/klavye
# uygulamalari elle yazilan "_" karakterini "-"ye, ">>" karakterini
# ".."ye cevirebiliyor; bu script sadece KULLANICI ADI ve SIFREYI
# argument olarak alir, tum alt-cizgi/yonlendirme karakterlerini
# KENDI icinde (kod olarak) uretir — kullanicinin bunlari elle
# yazmasi hic gerekmez.
#
# Kullanim (backend/ dizininde, /opt/4keys/backend):
#   bash ../deploy/set-auth-credentials.sh KULLANICI_ADI SIFRE
# ============================================================

if [ "$#" -ne 2 ]; then
  echo "Kullanim: bash set-auth-credentials.sh KULLANICI_ADI SIFRE"
  exit 1
fi

USERNAME="$1"
PASSWORD="$2"
ENV_FILE=".env"

if [ ! -f "$ENV_FILE" ]; then
  echo "HATA: $ENV_FILE bulunamadi. Bu script'i backend/ dizininde calistirin (/opt/4keys/backend)."
  exit 1
fi

# Var olan (bozuk veya eski) AUTH satirlarini temizle — hem alt cizgili
# hem yanlislikla tireli yazilmis olabilecek varyantlari.
grep -v -E "^FOURKEYS[_-]AUTH[_-]" "$ENV_FILE" > "$ENV_FILE.tmp" || true
mv "$ENV_FILE.tmp" "$ENV_FILE"

SECRET=$(openssl rand -hex 32)

{
  printf 'FOURKEYS_AUTH_USERNAME=%s\n' "$USERNAME"
  printf 'FOURKEYS_AUTH_PASSWORD=%s\n' "$PASSWORD"
  printf 'FOURKEYS_AUTH_SECRET_KEY=%s\n' "$SECRET"
} >> "$ENV_FILE"

echo "Tamamlandi. Eklenen satirlar:"
grep "^FOURKEYS_AUTH_" "$ENV_FILE"
echo
echo "Simdi backend'i yeniden baslatin (repo KOKUNDEN /opt/4keys):"
echo "  cd .. && bash deploy/recreate-backend.sh"
