#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# 4keys-backend'in son loglarindan hata/traceback satirlarini gosterir —
# konsola uzun/ozel karakterli komut yapistirmanin bozulma sorununu
# (bkz. onceki oturum notlari) asmak icin TEK DOSYA olarak eklendi.
#
# Calistirma (sunucuda, root olarak, /opt/4keys icinde):
#   1) Once Backtest sayfasinda "Backtest Calistir"a bas, hata cikmasini bekle
#   2) Sonra bunu calistir:
#      bash deploy/show-backend-errors.sh
# ============================================================

OUT_FILE="/tmp/backend-errors-$(date +%Y%m%d-%H%M%S).txt"
SINCE="${SINCE:-5m}"

{
  echo "=== Son ${SINCE} icinde /metrics ve /health DISINDA KALAN TUM satirlar ==="
  echo "--- (bu, gurultuyu (Prometheus'un saniyede bir /metrics taramasi) eleyip gercek istek/hata satirlarini one cikarir) ---"
  docker logs --since "$SINCE" 4keys-backend 2>&1 | grep -v -E "GET /metrics|GET /health" || echo "(bu surede metrics/health disinda hicbir satir yok)"
} | tee "$OUT_FILE"

echo
echo "Tamamlandi. Tam metin de kaydedildi: $OUT_FILE"
echo "Yukaridaki TUM ciktiyi kopyalayip gonder."
