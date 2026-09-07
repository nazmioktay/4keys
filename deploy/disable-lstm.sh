#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LSTM'i canli/backtest ensemble'indan devre disi birakir — model
# dosyasini SILMEZ, yalnizca .status.json'daki "enabled" alanini False
# yapar (write_model_status ile AYNI mekanizma, is_model_enabled bunu
# okur). Geri acmak icin: INCLUDE_LSTM=1 ile tekrar egitip esigi
# gecmesi yeterli (bkz. deploy/train-all-with-lstm.sh).
#
# NEDEN: LSTM kendi izole kalite esigini (oos_balanced_acc=0.411 > 0.37)
# gecti ama TAM SISTEM backtest'inde ensemble'a katilinca sonucu
# KOTULESTIRDI (kazanma orani %64.20 -> %50.35, PnL +%6.43 -> +%2.45) -
# bkz. README "karlilik". Demek ki izole dogruluk, ensemble'a katkida
# bulunacagini GARANTI ETMIYOR - gercek olcut tam sistem backtest'i.
#
# Calistirma (sunucuda, root olarak, /opt/4keys icinde):
#   bash deploy/disable-lstm.sh
# ============================================================

docker exec 4keys-backend python3 -c "
from app.ml.model_paths import DEFAULT_LSTM_MODEL_PATH
from app.ml.model_status import write_model_status, read_model_status

prev = read_model_status(DEFAULT_LSTM_MODEL_PATH) or {}
write_model_status(
    DEFAULT_LSTM_MODEL_PATH,
    enabled=False,
    balanced_accuracy=prev.get('balanced_accuracy', 0.0),
    reason='Elle devre disi birakildi - kendi esigini gecti ama tam sistem backtest sonucunu kotulestirdi (bkz. README karlilik).',
)
print('LSTM devre disi birakildi.')
"

echo "Tamamlandi. Dogrulamak icin: bash deploy/diagnose-oom.sh (Model durumlari bolumu LSTM icin enabled:false gostermeli)."
