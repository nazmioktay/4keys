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
  echo "=== TUM konteynerler ==="
  docker ps -a
  echo
  echo "=== TUM konteynerlerin anlik bellek kullanimi ==="
  docker stats --no-stream 2>&1
  echo
  echo "=== Docker disk/imaj kullanimi (birikim var mi) ==="
  docker system df 2>&1
  echo
  echo "=== 4keys-backend: restart sayisi / durum ==="
  docker inspect 4keys-backend --format 'RestartCount={{.RestartCount}} Status={{.State.Status}} StartedAt={{.State.StartedAt}}' 2>&1
  echo
  echo "=== 4keys-backend: bellek limiti (0 = limitsiz) ==="
  docker inspect 4keys-backend --format 'Memory={{.HostConfig.Memory}} MemorySwap={{.HostConfig.MemorySwap}}' 2>&1
  echo
  echo "=== Son OOM olaylari ==="
  dmesg 2>&1 | grep -i "out of memory\|killed process" | tail -10
  echo
  echo "=== DB tablo boyutlari (buyukten kucuge) ==="
  echo "--- (macro/orderbook/open_interest_snapshots: egitim/canli karar dongusu bunlarin TAMAMINI (200.000 satira kadar, lookback'ten BAGIMSIZ) yukluyor - buyukse muhtemel OOM kaynagi) ---"
  echo "--- (feature_snapshots/signals buyukse ENDISELENME: incelendi, feature_snapshots SADECE YAZILIYOR - egitim/canli kod bunu OKUMUYOR (bkz. README) - artik varsayilan olarak da yazilmiyor (ml_persist_feature_snapshots=False); eskiden birikmis satirlari temizlemek icin: bash deploy/truncate-feature-snapshots.sh) ---"
  docker exec 4keys-db bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT relname AS tablo, n_live_tup AS satir_sayisi, pg_size_pretty(pg_total_relation_size(relid)) AS boyut FROM pg_stat_user_tables ORDER BY n_live_tup DESC;"' 2>&1
} | tee "$OUT_FILE"

echo
echo "Tamamlandi. Tam metin de kaydedildi: $OUT_FILE (kaybolursa: cat $OUT_FILE)"
echo "Yukaridaki TUM ciktiyi kopyalayip gonder."
