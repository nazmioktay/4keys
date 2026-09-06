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
# calistiran AYNI process'e gonderiliyordu.
#
# TEK BASINA AYRI KONTEYNER YETERLI DEGIL (uretimde 4. kez gozlendi):
# --memory sinirinin canli API'yi KORUMADIGI ortaya cikti — bu sinir yalnizca
# BU konteynerin KENDI ust sinirini belirler; canli API'nin tabani (~1.8GB)
# ile bu egitim konteynerinin izin verilen ustu (once 2GB) TOPLANDIGINDA
# kutunun fiziksel RAM'ini (3.7GB) asiyor, kernel'in GENEL (host-capinda)
# OOM-killer'i devreye girip yine uvicorn'u secebiliyor - konteynerler AYRI
# olsa bile. Iki ek onlem eklendi:
#   1. Bellek sinirini daha muhafazakar yapmak (--memory=1500m): canli API'nin
#      tabaniyla toplandiginda fiziksel RAM'i ASMAMASI hedeflenir - global
#      OOM-killer'a hic ulasilmamasi, boylece.
#   2. --oom-score-adj=500: kernel'e "sikisirsa ONCE bunu oldur" der (bkz.
#      deploy/recreate-backend.sh'teki canli API'nin -500'u - "bunu EN SON
#      oldur"). Boylece 1500m yine de yetersiz kalirsa, feda edilen HER ZAMAN
#      bu (yeniden calistirilabilir) egitim islemi olur, canli API degil.
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
  --memory=1500m \
  --oom-score-adj=500 \
  --env-file "$ENV_FILE" \
  -v fourkeys_ml_artifacts:/app/app/ml/artifacts \
  4keys-backend \
  python -m app.cli train-all

echo
echo "Tamamlandi. Her adimin \"ok\" alanini kontrol edin."
