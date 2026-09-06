#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# feature_snapshots tablosundaki ESKİ, artık AMAÇSIZ kalmış satırları
# siler (bkz. README "OOM üretim olayı" - LSTM/RL şu an devre dışı
# olduğundan bu tabloyu okuyan hiçbir eğitim kodu yok, yalnızca her
# eğitim çağrısında lookback kadar satır TOPLU YAZILIYORDU - üretimde
# 291.208 satır/147MB bulundu). Kod tarafında bu yazma artık
# `settings.ml_persist_feature_snapshots` (varsayılan: False) ile
# KAPATILDI - bu script yalnızca DAHA ÖNCE birikmiş eski satırları temizler,
# disk/DB boyutunu küçültmek için.
#
# GÜVENLİ: feature_snapshots yalnızca gelecekte LSTM/RL için düşünülmüş bir
# yardımcı tablodur - ne canlı karar döngüsü ne backtest ne mevcut eğitim
# akışı bunu OKUR. Silinmesi trading/paper-trading davranışını ETKİLEMEZ.
#
# Calistirma (sunucuda, root olarak, /opt/4keys icinde):
#   bash deploy/truncate-feature-snapshots.sh
# ============================================================

echo "=== Silmeden ONCE satir sayisi ==="
docker exec 4keys-db bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT count(*) AS satir_sayisi, pg_size_pretty(pg_total_relation_size('"'"'feature_snapshots'"'"')) AS boyut FROM feature_snapshots;"' 2>&1

echo
echo "=== feature_snapshots temizleniyor (TRUNCATE) ==="
docker exec 4keys-db bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "TRUNCATE TABLE feature_snapshots;"' 2>&1

echo
echo "=== Sildikten SONRA satir sayisi (0 olmali) ==="
docker exec 4keys-db bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT count(*) AS satir_sayisi, pg_size_pretty(pg_total_relation_size('"'"'feature_snapshots'"'"')) AS boyut FROM feature_snapshots;"' 2>&1

echo
echo "Tamamlandi."
