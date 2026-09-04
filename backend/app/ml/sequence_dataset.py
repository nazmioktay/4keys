import logging

import numpy as np
from app.exchanges.base import Exchange

from .dataset import LabelingMethod, _compute_labels, _persist_feature_snapshots
from .features import FEATURE_COLUMNS, build_features

logger = logging.getLogger(__name__)


def build_sequence_dataset(
    exchange: Exchange,
    symbols: list[str],
    timeframe: str,
    lookback: int,
    seq_len: int = 20,
    horizon: int = 5,
    threshold_pct: float = 1.0,
    labeling_method: LabelingMethod = "threshold",
    take_profit_pct: float = 2.0,
    stop_loss_pct: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LSTM eğitimi için kayan pencereli (sliding window) sekans veri seti
    kurar. `app.ml.dataset.build_training_dataset`'ten farkı: XGBoost tekil
    özellik satırlarıyla çalışırken, LSTM her tahmin için son `seq_len`
    barın özellik vektörünü sırayla (zaman bilgisini koruyarak) görür.

    Her sembol kendi içinde bağımsız işlenir ve pencereler yalnızca o
    sembolün KENDİ kronolojik/kesintisiz serisinden kurulur — semboller
    arası pencere sızıntısı olmaz.

    Döner: (X, y, time_frac)
    - X: şekil (n_pencere, seq_len, n_özellik)
    - y: şekil (n_pencere,) — her pencerenin SON barındaki etiket
    - time_frac: şekil (n_pencere,) — o sembol serisi içindeki göreli
      zaman konumu (0..1), out-of-sample holdout bölmesi için
      (bkz. `app.ml.validation.split_out_of_sample`, burada numpy
      maskesiyle eşdeğer mantık uygulanır).
    """
    # Not (bellek): önceki sürüm her pencereyi ayrı bir küçük numpy dizisi
    # olarak bir Python listesine ekliyordu (n_pencere adet nesne, her biri
    # kendi bellek tahsisi + Python nesne başlığıyla), sonra `np.stack` bunu
    # tek bir büyük diziye kopyalıyordu — liste hâlâ bellekteyken. 10.000
    # mum × ~20 sembol × seq_len=20 örtüşen pencerede bu, gerçek veri
    # boyutunun (örtüşme nedeniyle zaten ~seq_len kat şişmiş) üzerine bir
    # kat daha (parçalanma + geçici kopya) bindiriyor ve üretim sunucusunda
    # OOM'a yol açtı. Bunun yerine `sliding_window_view` ile sembol başına
    # bir VIEW (kopyasız) çıkarılır; tüm semboller tek seferde
    # `np.concatenate` ile birleştirilir (kaçınılmaz tek kopya, ama
    # sembol/pencere başına ayrı Python nesnesi yok).
    per_symbol_X: list[np.ndarray] = []
    per_symbol_y: list[np.ndarray] = []
    per_symbol_time_frac: list[np.ndarray] = []

    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, lookback)
            if len(ohlcv) < 60:
                continue
            features = build_features(ohlcv)
            _persist_feature_snapshots(symbol, timeframe, features)
            labels = _compute_labels(ohlcv, labeling_method, horizon, threshold_pct, take_profit_pct, stop_loss_pct)
            frame = features.copy()
            frame["label"] = labels
            frame = frame.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

            n = len(frame)
            if n < seq_len + 1:
                continue

            feat_values = frame[FEATURE_COLUMNS].to_numpy(dtype="float32")
            label_values = frame["label"].to_numpy()

            windows = np.lib.stride_tricks.sliding_window_view(feat_values, seq_len, axis=0)
            # sliding_window_view (n - seq_len + 1, n_özellik, seq_len) döner;
            # beklenen (n_pencere, seq_len, n_özellik) şekline taşınır.
            windows = np.moveaxis(windows, -1, 1)

            per_symbol_X.append(windows)
            per_symbol_y.append(label_values[seq_len - 1 :])
            positions = np.arange(seq_len - 1, n, dtype="float64")
            per_symbol_time_frac.append(positions / max(n - 1, 1))
        except Exception as exc:  # noqa: BLE001 - tek sembol hatası tüm eğitimi durdurmamalı
            logger.warning("sequence dataset: skipping %s: %s", symbol, exc)
            continue

    if not per_symbol_X:
        return (
            np.empty((0, seq_len, len(FEATURE_COLUMNS)), dtype="float32"),
            np.empty((0,)),
            np.empty((0,)),
        )

    X = np.concatenate(per_symbol_X, axis=0)
    y = np.concatenate(per_symbol_y, axis=0)
    time_frac = np.concatenate(per_symbol_time_frac, axis=0)
    return X, y, time_frac
