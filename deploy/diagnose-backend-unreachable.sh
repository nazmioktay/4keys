#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# Frontend'de "Failed to fetch" hatası (bkz. Backtest sayfası) —
# tarayıcının backend'e HİÇ HTTP yanıtı ALAMADIĞI (nginx çökmüş/durmuş,
# backend container ayakta değil, ya da bağlantı reddedildi) durumlarda
# çıkan bir hata; API'nin kendi döndürdüğü bir hata mesajı DEĞİLDİR.
# Bu script backend<->nginx<->tarayıcı zincirindeki HER halkayı tek
# komutla kontrol eder. Hiçbir şeyi DEĞİŞTİRMEZ/YENİDEN BAŞLATMAZ.
#
# Calistirma (sunucuda, root olarak, /opt/4keys icinde):
#   bash deploy/diagnose-backend-unreachable.sh
# ============================================================

OUT_FILE="/tmp/diagnose-backend-$(date +%Y%m%d-%H%M%S).txt"

{
  echo "=== 4keys-backend konteyner durumu ==="
  docker inspect 4keys-backend --format 'Status={{.State.Status}} RestartCount={{.RestartCount}} OOMKilled={{.State.OOMKilled}} StartedAt={{.State.StartedAt}} ExitCode={{.State.ExitCode}}' 2>&1
  echo
  echo "=== Tum konteynerler ==="
  docker ps -a
  echo
  echo "=== Backend'e SUNUCUNUN KENDISINDEN erisim (localhost:8000) ==="
  curl -sS -o /dev/null -w "HTTP kodu: %{http_code}, sure: %{time_total}s\n" http://127.0.0.1:8000/health 2>&1
  curl -sS http://127.0.0.1:8000/health 2>&1
  echo
  echo "=== 4keys-backend son 80 log satiri (cokme/hata var mi) ==="
  docker logs --tail 80 4keys-backend 2>&1
  echo
  echo "=== Nginx durumu ==="
  systemctl status nginx --no-pager 2>&1 | head -20
  echo
  echo "=== Nginx config testi ==="
  nginx -t 2>&1
  echo
  echo "=== Nginx son 40 hata log satiri ==="
  tail -n 40 /var/log/nginx/error.log 2>&1
  echo
  echo "=== Dinlenen portlar (80/443/8000 nginx VE backend'de gorunmeli) ==="
  ss -tlnp 2>&1 | grep -E ':80 |:443 |:8000 ' || echo "(ss bulunamadi veya eslesme yok)"
} | tee "$OUT_FILE"

echo
echo "Tamamlandi. Tam metin de kaydedildi: $OUT_FILE"
echo "Yukaridaki TUM ciktiyi kopyalayip gonder."
