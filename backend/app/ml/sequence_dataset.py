import logging

import numpy as np
from app.exchanges.base import Exchange

from .dataset import LabelingMethod, _compute_labels
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
    all_X: list[np.ndarray] = []
    all_y: list[float] = []
    all_time_frac: list[float] = []

    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, lookback)
            if len(ohlcv) < 60:
                continue
            features = build_features(ohlcv)
            labels = _compute_labels(ohlcv, labeling_method, horizon, threshold_pct, take_profit_pct, stop_loss_pct)
            frame = features.copy()
            frame["label"] = labels
            frame = frame.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

            n = len(frame)
            if n < seq_len + 1:
                continue

            feat_values = frame[FEATURE_COLUMNS].to_numpy(dtype="float32")
            label_values = frame["label"].to_numpy()

            for i in range(seq_len - 1, n):
                all_X.append(feat_values[i - seq_len + 1 : i + 1])
                all_y.append(label_values[i])
                all_time_frac.append(i / max(n - 1, 1))
        except Exception as exc:  # noqa: BLE001 - tek sembol hatası tüm eğitimi durdurmamalı
            logger.warning("sequence dataset: skipping %s: %s", symbol, exc)
            continue

    if not all_X:
        return (
            np.empty((0, seq_len, len(FEATURE_COLUMNS)), dtype="float32"),
            np.empty((0,)),
            np.empty((0,)),
        )

    return np.stack(all_X), np.array(all_y), np.array(all_time_frac)
