#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Guven esigi taramasi: farkli open_confidence degerleriyle ard arda
# backtest calistirip her biri icin islem sayisi/kazanma orani/PnL/max
# drawdown doner. Egitim YAPMAZ, model degistirmez — yalnizca AYNI
# egitilmis modelle mevcut backtest'i birden fazla esikte tekrar oynatir.
# Ara denemeler backtest_runs tablosuna/Grafana panellerine YAZILMAZ.
#
# Otomatik "en iyi" esigi SECMEZ: az islemle gorulen yuksek bir kazanma
# orani, cok islemle gorulen daha dusuk bir orandan DAHA GUVENILIR
# degildir — karar operatore kalir. Denenecek degerleri asagidan
# degistirebilirsin.
#
# Tam JSON'u bir dosyaya da yazar — konsol gecmisinde (ozellikle Hetzner
# web konsolu gibi kaydirma gecmisi sinirli ortamlarda) yukari kaydirip
# eski satirlara ulasamama sorununu onlemek icin.
# ============================================================

OUT_FILE="/tmp/sweep-confidence-$(date +%Y%m%d-%H%M%S).json"

curl -sS -X POST http://127.0.0.1:8000/backtest/system/sweep-confidence \
  -H "Content-Type: application/json" \
  -d '{"open_confidence_values": [0.5, 0.55, 0.6, 0.65, 0.7]}' \
  -o "$OUT_FILE"

echo "Tam JSON kaydedildi: $OUT_FILE (konsolda kaybolursa: cat $OUT_FILE)"
echo

python3 - "$OUT_FILE" <<'PYEOF' 2>/dev/null || cat "$OUT_FILE"
import json, sys
with open(sys.argv[1]) as f:
    points = json.load(f)["points"]
header = f"{'open':>6} {'close':>6} {'islem':>6} {'kazanma%':>9} {'pnl%':>8} {'gunluk%':>8} {'drawdown%':>10}  hata"
print(header)
print("-" * len(header))
for p in points:
    err = p.get("error") or ""
    print(
        f"{p['open_confidence']:>6.2f} {p['close_confidence']:>6.2f} {p['trades_closed']:>6} "
        f"{p['win_rate_pct']:>9.2f} {p['total_pnl_pct']:>8.3f} {p['daily_pnl_pct']:>8.4f} "
        f"{p['max_drawdown_pct']:>10.3f}  {err}"
    )
PYEOF

echo
echo "Tamamlandi. Otomatik 'en iyi' esik SECILMEZ — ozellikle <30 islemli noktalarda sonuc guvenilirligi dusuktur, karar sana kalir."
