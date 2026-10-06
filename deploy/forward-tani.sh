#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# İleriye dönük veri toplayıcıları için TEŞHİS: (1) veri tazeliği — her
# tablonun son kaydı kaç dakika önce, (2) tasfiye akışı kopma logları,
# (3) CANLI test — 60 sn boyunca Binance tasfiye akışına (tüm piyasa)
# bağlanıp gelen mesajları sayar. YALNIZCA OKUR: hiçbir şeyi
# değiştirmez, emir göndermez, DB'ye yazmaz.
#
# Çalıştırma (sunucuda, /opt/4keys içinde):
#   bash deploy/forward-tani.sh
# Hızlı özet için: bash deploy/forward-durum.sh
# ============================================================

CONTAINER="4keys-backend"

echo "==> 1) Veri tazeliği (sunucu saati: $(date -u '+%Y-%m-%d %H:%M:%S') UTC)"
docker exec -i "$CONTAINER" python - <<'PY' || echo "   <-- tazelik okunamadı (yukarıdaki hataya bakın); diğer kontroller sürüyor"
# Tabloları belleğe YÜKLEMEZ: yalnızca count(*) ve max(time).
import pandas as pd
from sqlalchemy import func

from app.db.models import DepthBandSnapshot, LiquidationEvent, OIDetailSnapshot
from app.db.session import session_scope

now = pd.Timestamp.now(tz="UTC")
with session_scope() as s:
    stats = [(label, *s.query(func.count(), func.max(m.time)).select_from(m).one(), limit) for label, m, limit in [
        ("tasfiye  ", LiquidationEvent, None),
        ("derinlik ", DepthBandSnapshot, 15),
        ("açık poz.", OIDetailSnapshot, 15),
    ]]
for label, n, last, limit in stats:
    if not n:
        print(f"   {label}  satır: {0:>7}   hiç kayıt yok")
        continue
    last = pd.Timestamp(last)
    last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
    age = (now - last).total_seconds() / 60
    flag = ""
    if limit is not None and age > limit:
        flag = f"   <-- UYARI: {limit} dakikadan eski, toplama durmuş olabilir"
    print(f"   {label}  satır: {n:>7}   son kayıt {age:6.1f} dk önce{flag}")
PY

echo
echo "==> 2) Tasfiye akışı kopma logları (son 24 saat)"
logs=$(docker logs --since 24h "$CONTAINER" 2>&1 || true)  # log bir kez okunur
count=$(printf '%s\n' "$logs" | grep -c "tasfiye akışı koptu" || true)
echo "   kopma sayısı: $count"
if [ "$count" -gt 0 ]; then
  echo "   son hata(lar):"
  printf '%s\n' "$logs" | grep -A 6 "tasfiye akışı koptu" | tail -14 | sed 's/^/      /'
fi
flush=$(printf '%s\n' "$logs" | grep -c "liquidation flush başarısız" || true)
echo "   DB'ye yazma hatası sayısı: $flush"

echo
echo "==> 3) Canlı test: 60 sn Binance tasfiye akışı (tüm piyasa) dinleniyor..."
docker exec -i "$CONTAINER" python - <<'PY'
import asyncio
import json
import time

import websockets

from app.forwardcollect.liquidations import URL


async def main():
    counts = {}
    try:
        async with websockets.connect(URL, ping_interval=20, ping_timeout=20, open_timeout=15) as ws:
            print("   bağlantı: AÇILDI")
            end = time.monotonic() + 60
            while (left := end - time.monotonic()) > 0:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=left)
                except asyncio.TimeoutError:
                    break
                sym = json.loads(msg).get("o", {}).get("s", "?")
                counts[sym] = counts.get(sym, 0) + 1
    except Exception as exc:  # noqa: BLE001 - teşhis: hatayı olduğu gibi göster
        print(f"   bağlantı: HATA -> {type(exc).__name__}: {exc}")
        return
    total = sum(counts.values())
    print(f"   60 sn'de gelen tasfiye mesajı: {total} (tüm piyasa)")
    for s in ("BTCUSDT", "ETHUSDT"):
        print(f"   {s}: {counts.get(s, 0)}")
    if total == 0:
        print("   <-- UYARI: bağlantı açık ama hiç mesaj gelmedi; akış adresi/erişim sorunu olabilir")
    else:
        print("   akış çalışıyor. BTC/ETH sayısı 0 ise yalnızca bu iki sembolde tasfiye olmamıştır.")


asyncio.run(main())
PY
