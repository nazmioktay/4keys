#!/usr/bin/env bash
set -euo pipefail

cd /opt/4keys/frontend

echo "npm install calisiyor..."
npm install

echo "Frontend build ediliyor..."
npm run build

echo "Nginx'in servis ettigi klasore kopyalaniyor..."
rm -rf /var/www/4keys-frontend/*
cp -r dist/* /var/www/4keys-frontend/

echo "Tamamlandi. Yeni frontend yayinda."
