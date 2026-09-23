#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# `docker build` + `recreate-backend.sh`i TEK adımda yapar — bu script
# yazildi cunku VPS terminaline elle yazilan ":" karakteri bazen ";"ye
# donusuyor, bu da "docker build -t 4keys-backend:latest ..." komutunu
# bozuyordu (image YENIDEN BUILD EDILMIYOR, recreate-backend.sh eski
# image'i kullanmaya devam ediyordu). Bu script'i calistirmak icin
# ozel karakter yazmaya GEREK YOK.
#
# Calistirma (sunucuda, root olarak, /opt/4keys icinde):
#   bash deploy/rebuild-and-recreate-backend.sh
# ============================================================

APP_DIR="/opt/4keys"
cd "$APP_DIR"

echo "==> Backend image'i yeniden build ediliyor (birkaç dakika surebilir)..."
docker build --tag 4keys-backend:latest --file backend/Dockerfile backend

echo "==> Backend container'i yeniden olusturuluyor..."
bash deploy/recreate-backend.sh

echo "==> Dogrulama: /auth/login rotasi imajda var mi?"
if docker exec 4keys-backend grep -q "auth.router" /app/app/main.py; then
  echo "OK: auth.router main.py icinde bulundu — yeni kod calisiyor."
else
  echo "UYARI: auth.router bulunamadi — build/kod pull adiminda bir sorun olabilir."
fi
