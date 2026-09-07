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

{
  echo "=== Son 300 log satirindan hata/traceback icerenler ==="
  docker logs --tail 300 4keys-backend 2>&1 | grep -B2 -A 40 -iE "unhandled exception|traceback|error" || echo "(hicbir hata/traceback satiri bulunamadi)"
  echo
  echo "=== Ham son 60 satir (referans icin) ==="
  docker logs --tail 60 4keys-backend 2>&1
} | tee "$OUT_FILE"

echo
echo "Tamamlandi. Tam metin de kaydedildi: $OUT_FILE"
echo "Yukaridaki TUM ciktiyi kopyalayip gonder."
