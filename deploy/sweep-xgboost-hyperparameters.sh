#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# XGBoost hiperparametre taramasi: n_estimators x max_depth x
# learning_rate izgarasinda YENI adaylar egitip her biri icin TAM
# SISTEM backtest metriklerini (islem sayisi/kazanma orani/PnL/max
# drawdown) doner - kendi izole oos_balanced_accuracy'si de raporlanir
# ama KARAR ICIN ASIL GUVENILECEK OLAN PnL/drawdown'dir (bkz. README
# "LSTM denendi" bulgusu - izole dogruluk ensemble/gercek performansa
# katkiyi GARANTI ETMEZ).
#
# Uretim modelinin UZERINE YAZMAZ, ara denemeler backtest_runs tablosunu
# KIRLETMEZ. Otomatik "en iyi" kombinasyonu SECMEZ - karar operatore
# kalir. Denenecek degerleri asagidan degistirebilirsin.
#
# UYARI: her kombinasyon TAM BIR EGITIM + BACKTEST demek - varsayilan
# 3x3x3=27 kombinasyonla BIRKAC DAKIKA surebilir.
# ============================================================

OUT_FILE="/tmp/sweep-xgboost-hyperparameters-$(date +%Y%m%d-%H%M%S).json"

curl -sS -X POST http://127.0.0.1:8000/backtest/system/sweep-xgboost-hyperparameters \
  -H "Content-Type: application/json" \
  -d '{"n_estimators_values": [150, 300, 500], "max_depth_values": [3, 4, 6], "learning_rate_values": [0.03, 0.05, 0.1]}' \
  -o "$OUT_FILE"

echo "Tam JSON kaydedildi: $OUT_FILE (konsolda kaybolursa: cat $OUT_FILE)"
echo

python3 - "$OUT_FILE" <<'PYEOF' 2>/dev/null || cat "$OUT_FILE"
import json, sys
with open(sys.argv[1]) as f:
    points = json.load(f)["points"]
points.sort(key=lambda p: p["total_pnl_pct"], reverse=True)
header = f"{'n_est':>6} {'depth':>6} {'lr':>6} {'oos_acc':>8} {'islem':>6} {'kazanma%':>9} {'pnl%':>8} {'drawdown%':>10}  hata"
print(header)
print("-" * len(header))
for p in points:
    err = p.get("error") or ""
    print(
        f"{p['n_estimators']:>6} {p['max_depth']:>6} {p['learning_rate']:>6.3f} {p['oos_balanced_accuracy']:>8.3f} "
        f"{p['trades_closed']:>6} {p['win_rate_pct']:>9.2f} {p['total_pnl_pct']:>8.3f} {p['max_drawdown_pct']:>10.3f}  {err}"
    )
PYEOF

echo
echo "Tamamlandi (PnL'e gore en iyiden en kotuye siralandi). Otomatik 'en iyi' SECILMEZ, karar sana kalir."
