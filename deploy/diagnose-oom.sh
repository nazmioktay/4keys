#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# OOM (bellek yetersizligi) teshis paketi: 4keys-backend konteynerinin
# bellekten dolayi neden/nasil oldurulduysa nedenini anlamak icin gereken
# butun bilgiyi TEK KOMUTLA toplar. Hicbir sey DEGISTIRMEZ/YENIDEN
# BASLATMAZ — yalnizca okur/raporlar.
#
# Calistirma (sunucuda, root olarak, /opt/4keys icinde):
#   bash deploy/diagnose-oom.sh
# ============================================================

OUT_FILE="/tmp/diagnose-oom-$(date +%Y%m%d-%H%M%S).txt"

{
  echo "=== RAM/SWAP ==="
  free -h
  echo
  echo "=== Konteyner durumu ==="
  docker ps -a --filter name=4keys-backend
  echo
  echo "=== Restart sayisi / durum ==="
  docker inspect 4keys-backend --format 'RestartCount={{.RestartCount}} Status={{.State.Status}} StartedAt={{.State.StartedAt}}' 2>&1
  echo
  echo "=== Anlik bellek kullanimi ==="
  docker stats --no-stream 4keys-backend 2>&1
  echo
  echo "=== Docker bellek limiti (0 = limitsiz) ==="
  docker inspect 4keys-backend --format 'Memory={{.HostConfig.Memory}} MemorySwap={{.HostConfig.MemorySwap}}' 2>&1
  echo
  echo "=== Son OOM olaylari ==="
  dmesg 2>&1 | grep -i "out of memory\|killed process" | tail -10
} | tee "$OUT_FILE"

echo
echo "Tamamlandi. Tam metin de kaydedildi: $OUT_FILE (kaybolursa: cat $OUT_FILE)"
echo "Yukaridaki TUM ciktiyi kopyalayip gonder."
