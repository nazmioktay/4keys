#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 4keys — Nginx proxy zaman aşımını uzatır (screener taraması
# yüzlerce sembolü Binance'ten tek tek çektiği için varsayılan
# 60 saniyelik Nginx zaman aşımını aşabiliyor) ve askıda kalmış
# olabilecek arka plan görevlerini temizlemek için backend
# container'ını yeniden başlatır.
#
# Çalıştırma (sunucuda, root olarak):
#   curl -fsSL raw.githubusercontent.com/nazmioktay/4keys/main/deploy/fix_timeout.sh -o fix_timeout.sh
#   bash fix_timeout.sh
# ============================================================

cat > /etc/nginx/conf.d/timeouts.conf <<'EOF'
proxy_read_timeout 300s;
proxy_connect_timeout 300s;
proxy_send_timeout 300s;
EOF

nginx -t && systemctl reload nginx

docker restart 4keys-backend

echo "Tamamlandı: Nginx zaman aşımı 300 saniyeye çıkarıldı, backend yeniden başlatıldı."
