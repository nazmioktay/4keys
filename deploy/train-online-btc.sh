#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Gercek cevrimici (online) ogrenme: river'in Adaptive Random Forest'i
# (Hoeffding agaclarindan olusan, kendi ADWIN kavram kaymasi tespitine
# sahip bir topluluk) BTC-oncelikli semboller uzerinde bar-bar
# "test-then-train" (prequential) protokoluyle degerlendirilir.
#
# XGBoost'un statik train/holdout ayriminin AKSINE, burada model
# TARIHIN BASINDAN SONUNA dogru kronolojik olarak bar bar ogrenir; her
# barda ONCE tahmin yapilir (henuz o bari gormeden), SONRA gercek
# etiketle ogrenilir. "windows" alanindaki ilk ve son pencereleri
# karsilastirarak modelin zaman icinde adapte olup olmadigini gorebiliriz.
#
# persist=True (varsayilan) oldugu icin model diske kaydedilir, ama
# canli karar motoruna HENUZ baglanmadi.
# ============================================================

curl -sS -X POST http://127.0.0.1:8000/ml/train-online \
  -H "Content-Type: application/json" \
  -d '{"window_size": 500}'
echo
echo "Tamamlandi. windows[0] (en eski donem) ile windows[-1] (en yeni donem) accuracy'sini karsilastir — artis, modelin adapte oldugunu gosterir."
