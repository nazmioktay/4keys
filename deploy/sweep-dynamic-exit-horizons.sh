#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Dinamik cikis (kar-al/zarar-durdur) regresyon modelinin `horizon`
# (giristen sonra kac bar ileriye bakilacagi) izgarasinda ard arda
# calistirilip izole MAE/R^2 raporlanmasi — bkz. app.ml.dynamic_exit.
#
# Hicbir model KAYDEDILMEZ, canli/backtest'e KABLOLANMAZ — yalnizca
# hangi horizon degerinde model daha anlamli bir sinyal (R^2 daha
# yuksek) yakaliyor, onu gormek icin.
#
# Calistirma (VPS'te veya lokal ortamda, backend container'i ayaktayken):
#   bash deploy/sweep-dynamic-exit-horizons.sh
# ============================================================

CONTAINER="${FOURKEYS_BACKEND_CONTAINER:-4keys-backend}"
OUT_FILE="/tmp/sweep-dynamic-exit-horizons-$(date +%Y%m%d-%H%M%S).json"

docker exec "$CONTAINER" python3 -c "
import json
from app.core.config import settings
from app.exchanges import get_exchange
from app.ml.dynamic_exit import sweep_dynamic_exit_horizons

exchange = get_exchange(settings.exchange_id)
points = sweep_dynamic_exit_horizons(exchange, [settings.ml_primary_symbol], [5, 10, 15, 20, 25, 30, 40])
print(json.dumps([p.__dict__ for p in points]))
" > "$OUT_FILE"

echo "Tam JSON kaydedildi: $OUT_FILE (konsolda kaybolursa: cat $OUT_FILE)"
echo

python3 - "$OUT_FILE" <<'PYEOF' 2>/dev/null || cat "$OUT_FILE"
import json, sys
with open(sys.argv[1]) as f:
    points = json.load(f)
header = f"{'horizon':>7} {'satir':>6} {'peak_mae':>9} {'peak_r2':>8} {'trough_mae':>10} {'trough_r2':>9}  hata"
print(header)
print("-" * len(header))
for p in points:
    err = p.get("error") or ""
    print(
        f"{p['horizon']:>7} {p['rows_used']:>6} {p['peak_mae']:>9.4f} {p['peak_r2']:>8.4f} "
        f"{p['trough_mae']:>10.4f} {p['trough_r2']:>9.4f}  {err}"
    )
PYEOF

echo
echo "Tamamlandi. Hicbir model kaydedilmedi/canliya baglanmadi. R^2 en yuksek horizon, sonraki adim icin en umut verici aday."
