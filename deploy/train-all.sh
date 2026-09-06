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
# ONEMLI (uretim OOM olayindan sonra degisti): bu artik canli 4keys-backend
# konteyneri ICINDE degil, AYRI, tek seferlik bir konteynerde
# (docker run --rm) calisir — ayni image, farkli komut (python -m app.cli
# train-all). Onceden HTTP ile (curl POST /ml/train-all) canli API'yi
# calistiran AYNI process'e gonderiliyordu; bu process ayni zamanda arka
# plan zamanlayicisini da barindirdigindan, bes agir adimin TOPLAM bellek
# kullanimi kutunun fiziksel RAM'ini asip kernel OOM-killer'in canli API'yi
# oldurmesine yol acti (uretimde 3 kez gozlendi). Ayri konteyner + bellek
# ustsiniri (--memory) sayesinde bu egitim ne kadar bellek yerse yesin,
# canli API process'i ARTIK ETKILENMEZ — en kotu ihtimalle bu egitim
# konteyneri kendi ici sinirinda oldurulur, canli API dokunulmamis kalir
# (bkz. README "OOM uretim olayi").
#
# Cogunlukla ILK KURULUMDA (henuz hicbir model yokken, ör. yeni bir
# fourkeys_ml_artifacts volume'unden sonra) kullanilir.
# ============================================================

NETWORK="4keys-net"
ENV_FILE="/opt/4keys/backend/.env"

NETWORK_ARGS=()
if docker network inspect "$NETWORK" >/dev/null 2>&1; then
  echo "==> 4keys-net agi bulundu, egitim konteyneri ona baglanacak (veritabani erisimi icin)."
  NETWORK_ARGS=(--network "$NETWORK")
else
  echo "==> 4keys-net agi yok, veritabansiz calisiliyor."
fi

docker rm -f 4keys-train-all 2>/dev/null || true

docker run --rm \
  --name 4keys-train-all \
  "${NETWORK_ARGS[@]}" \
  --memory=2g \
  --env-file "$ENV_FILE" \
  -v fourkeys_ml_artifacts:/app/app/ml/artifacts \
  4keys-backend \
  python -m app.cli train-all

echo
echo "Tamamlandi. Her adimin \"ok\" alanini kontrol edin."
