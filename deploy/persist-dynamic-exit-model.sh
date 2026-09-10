#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Izole olcumler (bkz. deploy/sweep-dynamic-exit-horizons.sh) horizon=5'in
# diger degerlerden daha anlamli bir sinyal (R^2~0.11-0.12) yakaladigini
# gosterdi. Bu script AYNI modeli horizon=5 ile egitip DISKE KAYDEDER
# (persist=True, bkz. app.ml.dynamic_exit.DEFAULT_DYNAMIC_EXIT_MODEL_PATH)
# — boylece backend, SystemBacktestRequest.use_dynamic_exit=true ile
# gelen /backtest/system/run isteklerinde bu dosyayi yukleyip kullanabilir
# (bkz. app.api.routes.backtest.run_system).
#
# DIKKAT: bu script'i calistirmak modeli CANLI karar dongusune KABLOLAMAZ
# — yalnizca backtest'in use_dynamic_exit bayragi bu dosyayi okuyabilir
# hale gelir. Gercek etki, ayni backtest'i once use_dynamic_exit=false
# sonra true ile calistirip trades/win-rate/PnL/drawdown karsilastirarak
# olculmelidir (bkz. README) — izole R^2 zayif oldugu icin (~0.11) bu
# karsilastirma yapilmadan hicbir sonuc "iyilesme" sayilmamali.
#
# Calistirma (VPS'te, backend container'i ayaktayken):
#   bash deploy/persist-dynamic-exit-model.sh
# ============================================================

CONTAINER="${FOURKEYS_BACKEND_CONTAINER:-4keys-backend}"

docker exec "$CONTAINER" python3 -c "
from app.core.config import settings
from app.exchanges import get_exchange
from app.ml.dynamic_exit import train_dynamic_exit_model

exchange = get_exchange(settings.exchange_id)
result = train_dynamic_exit_model(
    exchange,
    [settings.ml_primary_symbol],
    horizon=5,
    persist=True,
)
r = result.out_of_sample
print('satir sayisi:', result.rows_used)
print('future_peak_pct  MAE:', round(r.peak_mae, 4), ' R^2:', round(r.peak_r2, 4))
print('future_trough_pct MAE:', round(r.trough_mae, 4), ' R^2:', round(r.trough_r2, 4))
"

echo
echo "Tamamlandi. Model KAYDEDILDI (dynamic_exit_model.joblib) — backtest'te use_dynamic_exit=true ile kullanilabilir."
echo "Sonraki adim: ayni backtest'i once use_dynamic_exit=false, sonra true ile calistirip GERCEK sistem etkisini karsilastirin."
