#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# İleriye dönük veri toplayıcılarının (tasfiye akışı, derinlik bantları,
# ayrıntılı açık pozisyon) durumunu gösterir: ayar açık mı, her tabloda
# kaç satır var, son kayıt ne zaman yazıldı. YALNIZCA OKUR — hiçbir şeyi
# değiştirmez, emir göndermez.
#
# Bu script, VPS terminaline elle yazılan uzun/tırnaklı komutların
# bozulmasını (":" -> ";", akıllı tırnak) önlemek için yazıldı; özel
# karakter yazmaya GEREK YOK.
#
# Çalıştırma (sunucuda, /opt/4keys içinde):
#   bash deploy/forward-durum.sh
# ============================================================

CONTAINER="4keys-backend"

echo "==> Ayar (konteyner ortamı):"
docker exec "$CONTAINER" env | grep FORWARD || echo "   FORWARD ayarı yok -> toplayıcılar KAPALI (varsayılan)"

echo
echo "==> Tablolar:"
docker exec -i "$CONTAINER" python - <<'PY'
# Tabloları belleğe YÜKLEMEZ: yalnızca count(*) ve max(time) (canlı backend konteynerinde bellek/DB yükü olmasın).
from sqlalchemy import func

from app.db.models import DepthBandSnapshot, LiquidationEvent, OIDetailSnapshot
from app.db.session import session_scope

with session_scope() as s:
    for label, model in [("tasfiye  ", LiquidationEvent), ("derinlik ", DepthBandSnapshot), ("açık poz.", OIDetailSnapshot)]:
        n, last = s.query(func.count(), func.max(model.time)).select_from(model).one()
        print(f"   {label}  satır: {n:>7}   son kayıt (UTC): {last or '-'}")
PY

echo
echo "Not: derinlik ve açık pozisyon her 5 dakikada sembol başına 1 satır artar."
echo "     Tasfiye yalnızca seçili semboller (BTC/ETH) için yazılır; bu ikisinde sakin dönemde saatlerce"
echo "     yeni satır gelmeyebilir. Akışın kendisi çalışıyor mu: bash deploy/forward-tani.sh"
