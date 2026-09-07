#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Etiketleme hedefi taramasi: horizon (ATR-triple-barrier zaman bariyeri,
# bar) x atr_multiplier (kar/zarar bariyerinin ATR carpani) izgarasinda
# YENI XGBoost adaylari egitip her biri icin TAM SISTEM backtest
# metriklerini doner.
#
# NEDEN: su anki hedef (3 bar/1.0xATR) yalnizca TEK bir sentetik
# saf-gurultu kontroluyle (sinif dagilimi saglikli mi) secilmisti -
# gercek veride/backtest'te sistematik olarak TARANMADI. Bu, o
# bosluyu kapatir.
#
# Uretim modelinin UZERINE YAZMAZ, otomatik "en iyi"yi SECMEZ - karar
# operatore kalir.
#
# UYARI: her kombinasyon TAM BIR EGITIM + BACKTEST demek - varsayilan
# 5x4=20 kombinasyonla BIRKAC DAKIKA surebilir.
# ============================================================

OUT_FILE="/tmp/sweep-labeling-targets-$(date +%Y%m%d-%H%M%S).json"

curl -sS -X POST http://127.0.0.1:8000/backtest/system/sweep-labeling-targets \
  -H "Content-Type: application/json" \
  -d '{"horizon_values": [2, 3, 5, 8, 12], "atr_multiplier_values": [0.75, 1.0, 1.5, 2.0]}' \
  -o "$OUT_FILE"

echo "Tam JSON kaydedildi: $OUT_FILE (konsolda kaybolursa: cat $OUT_FILE)"
echo

python3 - "$OUT_FILE" <<'PYEOF' 2>/dev/null || cat "$OUT_FILE"
import json, sys
with open(sys.argv[1]) as f:
    points = json.load(f)["points"]
points.sort(key=lambda p: p["total_pnl_pct"], reverse=True)
header = f"{'horizon':>7} {'atr_mult':>8} {'oos_acc':>8} {'islem':>6} {'kazanma%':>9} {'pnl%':>8} {'drawdown%':>10}  hata"
print(header)
print("-" * len(header))
for p in points:
    err = p.get("error") or ""
    print(
        f"{p['horizon']:>7} {p['atr_multiplier']:>8.2f} {p['oos_balanced_accuracy']:>8.3f} "
        f"{p['trades_closed']:>6} {p['win_rate_pct']:>9.2f} {p['total_pnl_pct']:>8.3f} {p['max_drawdown_pct']:>10.3f}  {err}"
    )
PYEOF

echo
echo "Tamamlandi (PnL'e gore en iyiden en kotuye siralandi). Otomatik 'en iyi' SECILMEZ, karar sana kalir."
