#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Kullanici onerisi "Seviye 1: Supervizeli Ogrenme (Regresyon)" — sabit
# bir ATR carpani yerine, giris anindaki piyasa kosullarina (mevcut 63
# ozellik) bakip "bu giriste fiyat ne kadar yukselip/dusecek" sorusunu
# REGRESYON ile tahmin eden AYRI bir model (bkz. app.ml.dynamic_exit).
#
# Bu script SADECE IZOLE olcum yapar (MAE/R^2) — canli karar dongusune/
# backtest'e KABLOLAMAZ, model dosyasi KAYDEDILMEZ (persist=False).
# NEDEN: bugunku oturumda tekrar tekrar gorulen ders — izole dogruluk
# artisi gercek sistem performansini GARANTI ETMEZ (bkz. README
# "Karlilik" LSTM/cok-sembol/ozellik-muhendisligi vakalari). Once bu
# modelin GERCEKTEN anlamli bir sey ogrenip ogrenmedigini (R^2 > 0 mi,
# yoksa gurultuye mi uyuyor) gormeden tam sisteme entegre etmek riskli.
#
# R^2 yorumu: 0'a yakin/negatif -> model, "ortalama tahmin et" kadar
# basariliymis, gercek bir sinyal YOK demektir. Pozitif ve anlamli
# (>0.1-0.2 gibi kaba bir esik, finansal veri gurultulu oldugu icin cok
# yuksek R^2 beklenmez) -> devam etmeye deger, sonraki adim (backtest'e
# kablolamak) tartisilabilir.
#
# Calistirma (VPS'te veya lokal ortamda, backend container'i ayaktayken):
#   bash deploy/train-dynamic-exit-model.sh
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
    horizon=25,
    persist=False,
)
r = result.out_of_sample
print('satir sayisi:', result.rows_used)
print()
print('future_peak_pct (kar-al hedefi) tahmini:')
print('  MAE:', round(r.peak_mae, 4), '(ortalama tahmin hatasi, yuzde puani)')
print('  R^2:', round(r.peak_r2, 4), '(0 = ortalama kadar iyi, negatif = ondan da kotu, 1 = mukemmel)')
print()
print('future_trough_pct (zarar-durdur hedefi) tahmini:')
print('  MAE:', round(r.trough_mae, 4))
print('  R^2:', round(r.trough_r2, 4))
"

echo
echo "Tamamlandi. Model KAYDEDILMEDI (persist=False) — bu yalnizca izole bir olcum."
echo "R^2 degerleri 0'a yakin/negatifse: model gercek bir sinyal ogrenemedi, sabit ATR carpani ile devam etmek daha guvenli."
