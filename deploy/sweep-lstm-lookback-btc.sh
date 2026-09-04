#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LSTM'i (yalnizca BTC/USDT), farkli lookback (mum sayisi) degerleriyle
# art arda egitip her birinin out-of-sample sonucunu yazdirir. "Mum
# sayisini artirmak ogrenmeyi pozitif etkiler mi?" sorusuna VARSAYIMLA
# degil, ampirik olarak cevap vermek icin.
#
# UYARI: her deger icin sifirdan bir LSTM egitimi calisir (birkac dakika
# x deger sayisi kadar surebilir) ve EN SON calisan deger, canli modelin
# kaydedilmis hali olarak kalir (train_lstm_signal_model her cagrida
# model.save() yapiyor).
#
# UYARI (bellek): art arda calisan agir egitim istekleri, AYNI uzun omurlu
# uvicorn process'inde belirgin bir kumulatif bellek artisina yol acabilir
# (Python/PyTorch bellek ayiricisi her istek sonrasi belleği isletim
# sistemine tam geri vermiyor). Uc deger tek seferde OOM'a yol actiysa,
# `bash recreate-backend.sh` ile backend'i yeniden baslatip kalan
# lookback degerlerini AYRI AYRI (sweep scriptini degil, tek tek
# train-lstm-btc.sh benzeri bir cagriyi) denemek daha guvenli olur.
# ============================================================

for LOOKBACK in 10000 20000 30000; do
  echo "=== LSTM (BTC/USDT), lookback=$LOOKBACK ==="
  curl -sS -X POST http://127.0.0.1:8000/ml/train-lstm \
    -H "Content-Type: application/json" \
    -d "{\"symbols\": [\"BTC/USDT:USDT\"], \"lookback\": $LOOKBACK}"
  echo
  echo
done

echo "Tamamlandi. Yukaridaki out_of_sample_accuracy / out_of_sample_balanced_accuracy / final_train_accuracy degerlerini karsilastir."
