#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 4keys — Hetzner sunucusuna tek seferlik kurulum script'i
# Ubuntu 24.04 üzerinde root olarak çalıştırılmak üzere yazıldı.
#
# Çalıştırma (sunucuda, root olarak):
#   curl -fsSL https://raw.githubusercontent.com/nazmioktay/4keys/main/deploy/setup.sh | bash
# ============================================================

APP_DOMAIN="app.4kyonetim.com.tr"
API_DOMAIN="api.4kyonetim.com.tr"
REPO_URL="https://github.com/nazmioktay/4keys.git"
APP_DIR="/opt/4keys"
LETSENCRYPT_EMAIL="nazmioktay@gmail.com"

echo "==> Paketler güncelleniyor..."
apt-get update -y
apt-get install -y ca-certificates curl gnupg git nginx ufw

echo "==> Docker kuruluyor..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

echo "==> Node.js 20 kuruluyor (frontend build için)..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

echo "==> Firewall (ufw) ayarlanıyor..."
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw --force enable

echo "==> Repo klonlanıyor: $REPO_URL -> $APP_DIR"
if [ -d "$APP_DIR/.git" ]; then
  cd "$APP_DIR" && git pull
else
  git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

echo "==> Backend .env oluşturuluyor (paper-trading güvenli varsayılanlar)..."
cat > backend/.env <<EOF
FOURKEYS_EXCHANGE_ID=binance
FOURKEYS_ENABLE_LIVE_TRADING=false
FOURKEYS_ENABLE_BIST_TRADING=false
FOURKEYS_BINANCE_TESTNET=true
FOURKEYS_CORS_ORIGINS=https://${APP_DOMAIN}
EOF
echo "    (backend/.env oluşturuldu — gerçek Binance API anahtarlarını daha sonra elle eklersin)"

echo "==> Backend Docker image build ediliyor..."
docker build -t 4keys-backend ./backend

echo "==> Backend container başlatılıyor (yalnızca localhost:8000, dışarı Nginx üzerinden)..."
docker rm -f 4keys-backend 2>/dev/null || true
docker run -d \
  --name 4keys-backend \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  --env-file backend/.env \
  4keys-backend

echo "==> Frontend build ediliyor..."
cd "$APP_DIR/frontend"
echo "VITE_API_BASE_URL=https://${API_DOMAIN}" > .env
npm install
npm run build
mkdir -p /var/www/4keys-frontend
rm -rf /var/www/4keys-frontend/*
cp -r dist/* /var/www/4keys-frontend/

echo "==> Nginx site tanımları yazılıyor..."
cat > /etc/nginx/sites-available/4keys-frontend <<EOF
server {
    listen 80;
    server_name ${APP_DOMAIN};
    root /var/www/4keys-frontend;
    index index.html;
    location / {
        try_files \$uri \$uri/ /index.html;
    }
}
EOF

cat > /etc/nginx/sites-available/4keys-api <<EOF
server {
    listen 80;
    server_name ${API_DOMAIN};
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

ln -sf /etc/nginx/sites-available/4keys-frontend /etc/nginx/sites-enabled/4keys-frontend
ln -sf /etc/nginx/sites-available/4keys-api /etc/nginx/sites-enabled/4keys-api
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

echo ""
echo "============================================================"
echo "Temel kurulum tamamlandı."
echo ""
echo "ŞİMDİ SIRADA (sen yapman gereken):"
echo "1) DNS: ${APP_DOMAIN} ve ${API_DOMAIN} için A kaydını"
echo "   bu sunucunun IP adresine yönlendir (domain sağlayıcında)."
echo "2) DNS yayıldıktan sonra (birkaç dakika-birkaç saat), HTTPS için:"
echo "   apt-get install -y certbot python3-certbot-nginx"
echo "   certbot --nginx -d ${APP_DOMAIN} -d ${API_DOMAIN} -m ${LETSENCRYPT_EMAIL} --agree-tos"
echo "============================================================"
