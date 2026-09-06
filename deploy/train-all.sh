#!/usr/bin/env bash
set -uo pipefail
# NOT -e: egitim konteyneri OOM ile (137) veya baska bir hatayla cikabilir
# ve BU BEKLENEN bir durum - script yine de Grafana/Prometheus'u geri
# baslatma adimina (asagida trap ile) ulasmali. Cikis kodu sonda elle
# yakalanip script'in kendi cikis kodu olarak kullanilir.

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
#   1. Bellek sinirini muhafazakar yapmak: canli API'nin tabaniyla
#      toplandiginda fiziksel RAM'i ASMAMASI hedeflenir - global
#      OOM-killer'a hic ulasilmamasi, boylece.
#   2. --oom-score-adj=500: kernel'e "sikisirsa ONCE bunu oldur" der (bkz.
#      deploy/recreate-backend.sh'teki canli API'nin -500'u - "bunu EN SON
#      oldur"). Boylece sinir yine de yetersiz kalirsa, feda edilen HER ZAMAN
#      bu (yeniden calistirilabilir) egitim islemi olur, canli API degil.
# Sunucuda DOGRULANDI: uc gercek OOM olayinda da oldurulen surec bu egitim
# konteyneriydi, uvicorn DEGIL (bkz. README "OOM uretim olayi").
#
# YETERSIZ KAPASITE (sunucuda gozlendi, sonra kismen COZULDU): kutuda
# backend'in yaninda Grafana + Prometheus + TimescaleDB de calisiyor
# (~730MB toplam) - egitime neredeyse hic pay kalmiyordu. Grafana/Prometheus
# canli TRADING islevine DAHIL DEGIL (salt izleme) - egitim suresince
# GECICI olarak durdurulup (trap ile HER durumda, basari/hata/Ctrl-C fark
# etmeksizin) sonunda geri baslatiliyor. Ayrica torch'un (LSTM/PatchTST)
# artik LAZY import edilmesiyle (bkz. README "torch HER ZAMAN yukleniyordu"
# bulgusu) canli API'nin kendi tabani da ~1.42GB'tan ~590MB'a dustu - bu
# ikisi BIRLIKTE egitime kalan payi onemli olcude buyuttu, --memory buna
# gore yukseltildi (1.5GB -> 2.2GB).
# (bkz. README "OOM uretim olayi").
#
# Cogunlukla ILK KURULUMDA (henuz hicbir model yokken, ör. yeni bir
# fourkeys_ml_artifacts volume'unden sonra) kullanilir.
#
# Istege bagli 1. argüman: tek bir sembol (ör. "BTC/USDT:USDT") verilirse
# yalnizca O sembolle egitir - kutu bu kadar kucukken (3.7GB) coklu-sembol
# veri hazirlamanin bellek zirvesini test etmek/atlamak icin (bkz.
# deploy/train-all-btc-only.sh - ayni seyi TEK KOMUTLA yapan kisayol).
# Bos birakilirsa (varsayilan) tum semboller (BTC + korelasyonlu digerleri)
# kullanilir.
#
# LSTM VARSAYILAN OLARAK ATLANIR (bkz. README "5 adimin ayni process'te
# zincirlenmesi" bulgusu): sunucuda tekrar tekrar gozlendi, LSTM (a) hic
# kalite esigini gecemedi (out_of_sample_balanced_accuracy hep <0.37), (b)
# torch/PyTorch nedeniyle en pahali TEK adim - atlamanin pratik bir kaybi
# yok ama kalan adimlarin tamamlanma sansini artirir. Yine de denemek
# istersen: INCLUDE_LSTM=1 bash deploy/train-all.sh
# ============================================================

NETWORK="4keys-net"
ENV_FILE="/opt/4keys/backend/.env"
SYMBOLS_ARGS=()
if [ -n "${1:-}" ]; then
  echo "==> Yalniz $1 ile egitiliyor (coklu-sembol atlaniyor)."
  SYMBOLS_ARGS=(--symbols "$1")
fi

SKIP_ARGS=()
if [ "${INCLUDE_LSTM:-0}" != "1" ]; then
  echo "==> LSTM atlaniyor (hic kalite esigini gecemedi, en pahali adim - INCLUDE_LSTM=1 ile dahil edilebilir)."
  SKIP_ARGS=(--skip lstm)
fi

NETWORK_ARGS=()
if docker network inspect "$NETWORK" >/dev/null 2>&1; then
  echo "==> 4keys-net agi bulundu, egitim konteyneri ona baglanacak (veritabani erisimi icin)."
  NETWORK_ARGS=(--network "$NETWORK")
else
  echo "==> 4keys-net agi yok, veritabansiz calisiliyor."
fi

echo "==> Grafana/Prometheus egitim suresince gecici olarak durduruluyor (bellek acmak icin)..."
docker stop 4keys-grafana 4keys-prometheus >/dev/null 2>&1 || true

restart_monitoring() {
  echo "==> Grafana/Prometheus yeniden baslatiliyor..."
  docker start 4keys-grafana 4keys-prometheus >/dev/null 2>&1 || true
}
trap restart_monitoring EXIT

docker rm -f 4keys-train-all 2>/dev/null || true

docker run --rm \
  --name 4keys-train-all \
  "${NETWORK_ARGS[@]}" \
  --memory=2200m \
  --oom-score-adj=500 \
  --env-file "$ENV_FILE" \
  -v fourkeys_ml_artifacts:/app/app/ml/artifacts \
  4keys-backend \
  python -m app.cli train-all "${SYMBOLS_ARGS[@]}" "${SKIP_ARGS[@]}"
TRAIN_EXIT=$?

echo
if [ "$TRAIN_EXIT" -eq 0 ]; then
  echo "Tamamlandi. Her adimin \"ok\" alanini kontrol edin."
else
  echo "Egitim konteyneri hata/OOM ile sonlandi (exit=$TRAIN_EXIT) - tekrar deneyebilirsin. Canli API bundan ETKILENMEMIS olmali (bkz. bash deploy/diagnose-oom.sh ile dogrula)."
fi
exit "$TRAIN_EXIT"
