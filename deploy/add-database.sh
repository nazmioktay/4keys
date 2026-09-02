#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 4keys — Kalıcı veritabanını (TimescaleDB/PostgreSQL) sunucuya ekler.
#
# setup.sh backend'i düz `docker run` ile (docker-compose olmadan)
# çalıştırdığı için, DB container'ının backend'e "4keys-db" adıyla
# ulaşılabilmesi amacıyla ikisini de aynı özel Docker ağına (4keys-net)
# bağlar ve backend container'ını bu ağla yeniden oluşturur.
#
# Şifre bu script tarafından otomatik üretilir ve backend/.env'e
# yazılır — elle bir şey girmene gerek yok.
#
# Çalıştırma (sunucuda, root olarak):
#   curl -fsSL raw.githubusercontent.com/nazmioktay/4keys/main/deploy/add-database.sh -o add-database.sh
#   bash add-database.sh
# ============================================================

APP_DIR="/opt/4keys"
NETWORK="4keys-net"
DB_CONTAINER="4keys-db"
DB_NAME="fourkeys"
DB_USER="fourkeys"
ENV_FILE="$APP_DIR/backend/.env"

echo "==> Docker ağı oluşturuluyor (varsa atlanır)..."
docker network create "$NETWORK" 2>/dev/null || true

if [ -f "$APP_DIR/.db_password" ]; then
  DB_PASSWORD=$(cat "$APP_DIR/.db_password")
  echo "==> Mevcut veritabanı şifresi kullanılıyor."
else
  DB_PASSWORD=$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 32)
  echo "$DB_PASSWORD" > "$APP_DIR/.db_password"
  chmod 600 "$APP_DIR/.db_password"
  echo "==> Yeni veritabanı şifresi üretildi ve kaydedildi."
fi

echo "==> TimescaleDB container'ı başlatılıyor..."
docker rm -f "$DB_CONTAINER" 2>/dev/null || true
docker volume create fourkeys_db_data >/dev/null
docker run -d \
  --name "$DB_CONTAINER" \
  --network "$NETWORK" \
  --restart unless-stopped \
  -e POSTGRES_DB="$DB_NAME" \
  -e POSTGRES_USER="$DB_USER" \
  -e POSTGRES_PASSWORD="$DB_PASSWORD" \
  -v fourkeys_db_data:/var/lib/postgresql/data \
  timescale/timescaledb:latest-pg16

echo "==> Veritabanının hazır olması bekleniyor..."
for i in $(seq 1 30); do
  if docker exec "$DB_CONTAINER" pg_isready -U "$DB_USER" >/dev/null 2>&1; then
    echo "    Veritabanı hazır."
    break
  fi
  sleep 2
done

echo "==> backend/.env güncelleniyor..."
if grep -q "^FOURKEYS_DATABASE_URL=" "$ENV_FILE" 2>/dev/null; then
  sed -i "s#^FOURKEYS_DATABASE_URL=.*#FOURKEYS_DATABASE_URL=postgresql+psycopg2://${DB_USER}:${DB_PASSWORD}@${DB_CONTAINER}:5432/${DB_NAME}#" "$ENV_FILE"
else
  echo "FOURKEYS_DATABASE_URL=postgresql+psycopg2://${DB_USER}:${DB_PASSWORD}@${DB_CONTAINER}:5432/${DB_NAME}" >> "$ENV_FILE"
fi

echo "==> Backend container'ı aynı ağa bağlı olarak yeniden oluşturuluyor..."
docker rm -f 4keys-backend 2>/dev/null || true
docker run -d \
  --name 4keys-backend \
  --network "$NETWORK" \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  --env-file "$ENV_FILE" \
  4keys-backend

echo ""
echo "============================================================"
echo "Veritabanı kuruldu ve backend'e bağlandı."
echo "Doğrulamak için: curl -s api.acromer.com/db/status"
echo "============================================================"
