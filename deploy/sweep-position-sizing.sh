#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Pozisyon boyutlandirma taramasi: kelly_min_trades x kelly_multiplier
# izgarasinda ard arda backtest calistirip her kombinasyon icin islem
# sayisi/kazanma orani/PnL/max drawdown doner. Egitim YAPMAZ, model
# degistirmez — yalnizca AYNI egitilmis modelle mevcut backtest'i birden
# fazla boyutlandirma ayariyla tekrar oynatir. Ara denemeler
# backtest_runs tablosuna/Grafana panellerine YAZILMAZ.
#
# NEDEN: gercek uretim backtest'inde (bkz. README "karlilik") ilk ~40
# islemlik gurultulu pencerede kazanma orani gecici olarak %50'nin
# altina dusunce Kelly formulu "tam Kelly=%0" hesapladi ve o donemdeki
# ~25-30 islem TAMAMEN SIFIR boyutla acildi (sermaye o donem hic
# kullanilmadi) — nihai kazanma orani (%63) cok daha iyi olsa bile.
# kelly_min_trades'i artirmak bunu onleyebilir; kelly_multiplier'i
# artirmak (dusuk drawdown'lu donemlerde) getiriyi buyutebilir.
#
# Otomatik "en iyi" kombinasyonu SECMEZ — karar operatore kalir. Denenecek
# degerleri asagidan degistirebilirsin.
# ============================================================

OUT_FILE="/tmp/sweep-position-sizing-$(date +%Y%m%d-%H%M%S).json"

curl -sS -X POST http://127.0.0.1:8000/backtest/system/sweep-position-sizing \
  -H "Content-Type: application/json" \
  -d '{"kelly_min_trades_values": [10, 20, 40, 60], "kelly_multiplier_values": [0.5, 0.75, 1.0]}' \
  -o "$OUT_FILE"

echo "Tam JSON kaydedildi: $OUT_FILE (konsolda kaybolursa: cat $OUT_FILE)"
echo

python3 - "$OUT_FILE" <<'PYEOF' 2>/dev/null || cat "$OUT_FILE"
import json, sys
with open(sys.argv[1]) as f:
    points = json.load(f)["points"]
header = f"{'min_trd':>7} {'mult':>5} {'islem':>6} {'kazanma%':>9} {'pnl%':>8} {'gunluk%':>8} {'drawdown%':>10}  hata"
print(header)
print("-" * len(header))
for p in points:
    err = p.get("error") or ""
    print(
        f"{p['kelly_min_trades']:>7} {p['kelly_multiplier']:>5.2f} {p['trades_closed']:>6} "
        f"{p['win_rate_pct']:>9.2f} {p['total_pnl_pct']:>8.3f} {p['daily_pnl_pct']:>8.4f} "
        f"{p['max_drawdown_pct']:>10.3f}  {err}"
    )
PYEOF

echo
echo "Tamamlandi. Otomatik 'en iyi' kombinasyon SECILMEZ — ozellikle az islemli noktalarda sonuc guvenilirligi dusuktur, karar sana kalir."
