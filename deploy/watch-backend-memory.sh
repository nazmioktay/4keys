#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# Canli 4keys-backend'in (uvicorn) bellegini ZAMAN icinde izler —
# bkz. README "canli API'nin bellegi zamanla buyuyor" bulgusu: uretimde
# taze bir restart'ta ~590MB olan bu surec, gozlemlenen bir OOM olayinda
# anon-rss:2657744kB'a (~2.66GB) kadar cikip --oom-score-adj=-500
# korumasina RAGMEN oldurulmustu (dmesg: "Out of memory: Killed process
# ... (uvicorn) ... oom_score_adj:-500") — bu, buyumenin cok HIZLI
# (tek bir egitim dongusu suresi icinde) olabildigini gosteriyor.
#
# Bu script HICBIR SEYI DEGISTIRMEZ/YENIDEN BASLATMAZ — yalnizca
# DURATION_SECONDS boyunca INTERVAL_SECONDS'ta bir anlik bellek + restart
# sayisi + canli/olu durumunu kaydeder. Amac: buyumenin SUREKLI mi yoksa
# SICRAMALI mi oldugunu, ve varsa hangi zamanlayici isiyle (ör. 5 dakikalik
# karar dongusu) CAKISTIGINI gormek.
#
# Calistirma (sunucuda, root olarak, /opt/4keys icinde, ~10 dakika surer):
#   bash deploy/watch-backend-memory.sh
# Daha uzun/kisa/sik izlemek icin:
#   DURATION_SECONDS=1800 INTERVAL_SECONDS=30 bash deploy/watch-backend-memory.sh
# ============================================================

DURATION_SECONDS="${DURATION_SECONDS:-600}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-15}"
OUT_FILE="/tmp/watch-backend-memory-$(date +%Y%m%d-%H%M%S).txt"

echo "==> ${DURATION_SECONDS} saniye boyunca ${INTERVAL_SECONDS} saniyede bir izlenecek. Cikti: $OUT_FILE"

{
  echo "=== Baslangic: $(date -Iseconds) ==="
  elapsed=0
  while [ "$elapsed" -lt "$DURATION_SECONDS" ]; do
    ts="$(date -Iseconds)"
    restart_count="$(docker inspect 4keys-backend --format '{{.RestartCount}}' 2>&1)"
    status="$(docker inspect 4keys-backend --format '{{.State.Status}}' 2>&1)"
    mem_line="$(docker stats --no-stream --format '{{.MemUsage}} ({{.MemPerc}})' 4keys-backend 2>&1)"
    health="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8000/health 2>&1)"
    echo "[$ts] status=$status restart_count=$restart_count mem=$mem_line health_http=$health"
    sleep "$INTERVAL_SECONDS"
    elapsed=$((elapsed + INTERVAL_SECONDS))
  done
  echo "=== Bitis: $(date -Iseconds) ==="
} | tee "$OUT_FILE"

echo
echo "Tamamlandi. Tam metin de kaydedildi: $OUT_FILE"
echo "Yukaridaki TUM ciktiyi kopyalayip gonder — restart_count arttiysa (veya status/health degistiyse) izleme sirasinda OOM/cokme olmus demektir."
