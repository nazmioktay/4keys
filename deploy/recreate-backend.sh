#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 4keys — backend container'ını, en son build edilen "4keys-backend"
# image'ıyla, doğru ağ ve port ayarlarıyla yeniden oluşturur.
#
# `docker restart`, container'ı YENİ build edilen image'la değil,
# oluşturulduğu ANDAKİ eski image'la yeniden başlatır — bu yüzden kod
# güncellemesi sonrası her zaman recreate (rm + run) gerekir.
#
# Çalıştırma (sunucuda, root olarak, /opt/4keys içinde):
#   curl -fsSL raw.githubusercontent.com/nazmioktay/4keys/main/deploy/recreate-backend.sh -o recreate-backend.sh
#   bash recreate-backend.sh
# ============================================================

APP_DIR="/opt/4keys"
NETWORK="4keys-net"
ENV_FILE="$APP_DIR/backend/.env"

docker rm -f 4keys-backend 2>/dev/null || true

if docker network inspect "$NETWORK" >/dev/null 2>&1; then
  echo "==> 4keys-net ağı bulundu, backend ona bağlanacak (veritabanı erişimi için)."
  docker run -d \
    --name 4keys-backend \
    --network "$NETWORK" \
    --restart unless-stopped \
    --publish 127.0.0.1:8000:8000 \
    --env-file "$ENV_FILE" \
    4keys-backend
else
  echo "==> 4keys-net ağı yok, veritabansız çalışılıyor."
  docker run -d \
    --name 4keys-backend \
    --restart unless-stopped \
    --publish 127.0.0.1:8000:8000 \
    --env-file "$ENV_FILE" \
    4keys-backend
fi

echo "Tamamlandi: backend yeniden olusturuldu."
