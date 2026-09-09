#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Ozellik matrisindeki (app.ml.features.FEATURE_COLUMNS — teknik
# gostergeler, MULTI_TIMEFRAME_FEATURE_COLUMNS dahil) ikili Pearson
# korelasyonunu gercek BTC verisiyle hesaplar, |korelasyon| >= 0.90
# olan ciftleri buyukten kucuge siralayip listeler.
#
# NEDEN: kullanici isteği — "kolay olanlari ekleyip performansa
# bakalim, sonrasinda birbiriyle %90 uzeri korele olan ozellikleri
# hesaplayip eleyebiliriz" (bkz. README "Karlilik"). Yuksek korelasyonlu
# ciftler modele YENI bilgi katmiyor demektir — ayni sinyalin iki farkli
# olcumu (ornek supheli: choppiness_index vs hurst_exponent, ikisi de
# "trend mi/yatay mi" sorusuna farkli formullerle cevap veriyor).
#
# Egitim/model DEGISTIRMEZ, hicbir seyi KAYDETMEZ — salt-okunur analiz.
# multi-timeframe ozellikleri (varsayilan KAPALI, bkz.
# ml_enable_multi_timeframe_features) bu calistirma icin GECICI olarak
# acilir (yalnizca bu python surecinde, .env/DB'ye yazilmaz).
#
# Calistirma (VPS'te veya lokal ortamda, backend container'i ayaktayken):
#   bash deploy/check-feature-correlations.sh
# ============================================================

CONTAINER="${FOURKEYS_BACKEND_CONTAINER:-4keys-backend}"

docker exec "$CONTAINER" python3 -c "
from app.core.config import settings
settings.ml_enable_multi_timeframe_features = True

from app.exchanges import get_exchange
from app.ml.features import build_features, FEATURE_COLUMNS
from app.ml.multi_timeframe_features import MULTI_TIMEFRAME_FEATURE_COLUMNS, compute_multi_timeframe_features

exchange = get_exchange(settings.exchange_id)
ohlcv = exchange.fetch_ohlcv(settings.ml_primary_symbol, settings.ml_train_timeframe, settings.ml_train_lookback)
feats = build_features(ohlcv)
htf = compute_multi_timeframe_features(ohlcv)
for col in MULTI_TIMEFRAME_FEATURE_COLUMNS:
    feats[col] = htf[col].to_numpy()
feats = feats.dropna()
print(len(feats), 'satir (NaN atildiktan sonra), sembol=', settings.ml_primary_symbol, 'tf=', settings.ml_train_timeframe)

all_cols = FEATURE_COLUMNS + MULTI_TIMEFRAME_FEATURE_COLUMNS
cols = [c for c in all_cols if feats[c].std() > 1e-12]
skipped = sorted(set(all_cols) - set(cols))
if skipped:
    print('Sabit/varyanssiz oldugu icin atlanan', len(skipped), 'kolon:', skipped)

corr = feats[cols].corr().abs()
pairs = []
for i, a in enumerate(cols):
    for b in cols[i + 1:]:
        c = corr.loc[a, b]
        if c >= 0.90:
            pairs.append((c, a, b))
pairs.sort(reverse=True)

print()
print('korelasyon  ozellik A  <->  ozellik B')
print('-' * 60)
for c, a, b in pairs:
    print(round(c, 3), a, '<->', b)
print()
print('Toplam', len(pairs), 'cift >= 0.90 korelasyonlu (taranan', len(cols), 'ozellik,', len(cols) * (len(cols) - 1) // 2, 'cift).')
"

echo
echo "Tamamlandi. Otomatik eleme YAPILMAZ — hangi ozelligin tutulacagina (genelde daha yorumlanabilir/ucuz hesaplanan taraf) karar sana kalir."
