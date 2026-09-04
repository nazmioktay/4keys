#!/usr/bin/env bash
set -euo pipefail

cd /opt/4keys

if ! docker network inspect 4keys-net >/dev/null 2>&1; then
  echo "4keys-net agi bulunamadi, once add-database.sh calistirilmis olmali."
  exit 1
fi

echo "Prometheus baslatiliyor..."
docker rm -f 4keys-prometheus >/dev/null 2>&1 || true
docker run -d --name 4keys-prometheus \
  --network 4keys-net \
  --restart unless-stopped \
  --publish 127.0.0.1:9090:9090 \
  --volume /opt/4keys/monitoring/prometheus-prod.yml:/etc/prometheus/prometheus.yml:ro \
  --volume 4keys-prometheus-data:/prometheus \
  prom/prometheus:latest

echo "Grafana baslatiliyor..."
docker rm -f 4keys-grafana >/dev/null 2>&1 || true
docker run -d --name 4keys-grafana \
  --network 4keys-net \
  --restart unless-stopped \
  --publish 3001:3000 \
  --env GF_SECURITY_ADMIN_PASSWORD=admin \
  --volume /opt/4keys/monitoring/grafana/provisioning-prod:/etc/grafana/provisioning:ro \
  --volume /opt/4keys/monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro \
  --volume 4keys-grafana-data:/var/lib/grafana \
  grafana/grafana:latest

echo "Tamamlandi."
echo "Prometheus: sadece sunucu icinden (127.0.0.1:9090) erisilebilir."
echo "Grafana: http://SUNUCU_IP:3001 (kullanici: admin, sifre: admin -- ILK GIRISTE DEGISTIRIN)."
