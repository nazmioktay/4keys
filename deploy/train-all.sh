#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Tum modelleri (XGBoost -> meta-label -> LSTM -> online -> regime) TEK
# CAGRIDA, deploy script'lerinde (train-xgboost-best-labeling.sh,
# train-meta.sh, train-lstm-btc-best-labeling.sh, train-online-btc.sh,
# train-regime-multi.sh) dogrulanmis AYNI parametrelerle sirayla egitir.
#
# Bir adimin basarisiz olmasi digerlerini ENGELLEMEZ; her adimin sonucu
# ayri raporlanir. LSTM en yavas adimdir, toplam sure birkac dakika
# surebilir.
#
# Cogunlukla ILK KURULUMDA (henuz hicbir model yokken, ör. yeni bir
# fourkeys_ml_artifacts volume'unden sonra) kullanilir.
# ============================================================

curl -sS -X POST http://127.0.0.1:8000/ml/train-all \
  -H "Content-Type: application/json" \
  -d '{}'
echo
echo "Tamamlandi. Her adimin \"ok\" alanini kontrol edin."
