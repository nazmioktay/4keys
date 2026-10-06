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
from app.db import repository as r

tables = [
    ("tasfiye  ", r.get_liquidation_events),
    ("derinlik ", r.get_depth_band_snapshots),
    ("açık poz.", r.get_oi_detail_snapshots),
]
for label, reader in tables:
    df = reader()
    last = df["time"].max() if len(df) else "-"
    print(f"   {label}  satır: {len(df):>7}   son kayıt (UTC): {last}")
PY

echo
echo "Not: derinlik ve açık pozisyon her 5 dakikada sembol başına 1 satır artar."
echo "     Tasfiye piyasaya bağlıdır; sakin dönemde saatlerce 0 kalabilir."
