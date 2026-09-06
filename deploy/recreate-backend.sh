#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 4keys — backend container'ını, en son build edilen "4keys-backend"
# image'ıyla, doğru ağ ve port ayarlarıyla yeniden oluşturur.
#
# `docker restart`, container'ı YENİ build edilen image'la değil,
# oluşturulduğu ANDAKİ eski image'la yeniden başlatır — bu yüzden kod
# güncellemesi sonrası her zaman recreate (rm + run) gerekir.
#
# Çalıştırma (sunucuda, root olarak, /opt/4keys içinde):
#   curl -fsSL raw.githubusercontent.com/nazmioktay/4keys/main/deploy/recreate-backend.sh -o recreate-backend.sh
#   bash recreate-backend.sh
# ============================================================

APP_DIR="/opt/4keys"
NETWORK="4keys-net"
ENV_FILE="$APP_DIR/backend/.env"

docker rm -f 4keys-backend 2>/dev/null || true

# Eğitilmiş modeller (XGBoost/LSTM/online/regime/meta-label) container
# içindeki /app/app/ml/artifacts'te tutulur ve imaja (.gitignore'da olduğu
# için) GÖMÜLMEZ — bu named volume olmadan her `docker rm` + yeni image ile
# `docker run` modelleri SİLERDİ, otomatik/manuel her retrain'in sıfırdan
# başlamasına yol açardı. Volume zaten varsa `create` no-op'tur.
docker volume create fourkeys_ml_artifacts >/dev/null

# --oom-score-adj=-500: kernel'e "kutu bellek yetersizliğine düşerse EN SON
# bunu öldür" der (bkz. deploy/train-all.sh'teki eğitim konteynerinin
# +500'ü, "ÖNCE bunu öldür"). Ayrı konteynerler bile olsalar, --memory
# sınırının TEK BAŞINA canlı API'yi korumadığı üretimde görüldü (toplam
# sistem belleği fiziksel RAM'i aşınca kernel'in GENEL OOM-killer'ı devreye
# giriyor ve hangi konteyner olursa olsun seçebiliyor) — bu, sıkışma anında
# kernel'in tercihini canlı API LEHİNE, eğitim ALEYHİNE açıkça yönlendirir.
if docker network inspect "$NETWORK" >/dev/null 2>&1; then
  echo "==> 4keys-net ağı bulundu, backend ona bağlanacak (veritabanı erişimi için)."
  docker run -d \
    --name 4keys-backend \
    --network "$NETWORK" \
    --restart unless-stopped \
    --oom-score-adj=-500 \
    --publish 127.0.0.1:8000:8000 \
    --env-file "$ENV_FILE" \
    -v fourkeys_ml_artifacts:/app/app/ml/artifacts \
    4keys-backend
else
  echo "==> 4keys-net ağı yok, veritabansız çalışılıyor."
  docker run -d \
    --name 4keys-backend \
    --restart unless-stopped \
    --oom-score-adj=-500 \
    --publish 127.0.0.1:8000:8000 \
    --env-file "$ENV_FILE" \
    -v fourkeys_ml_artifacts:/app/app/ml/artifacts \
    4keys-backend
fi

echo "Tamamlandi: backend yeniden olusturuldu."
