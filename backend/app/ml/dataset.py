import logging
from typing import Literal

import pandas as pd

from app.exchanges.base import Exchange

from .features import FEATURE_COLUMNS, build_features
from .labeling import label_future_direction, triple_barrier_labels

logger = logging.getLogger(__name__)

LabelingMethod = Literal["threshold", "triple_barrier"]


def _compute_labels(
    ohlcv: pd.DataFrame,
    labeling_method: LabelingMethod,
    horizon: int,
    threshold_pct: float,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> pd.Series:
    if labeling_method == "triple_barrier":
        return triple_barrier_labels(ohlcv, take_profit_pct, stop_loss_pct, max_horizon=horizon)
    return label_future_direction(ohlcv["close"], horizon, threshold_pct)


def build_training_dataset(
    exchange: Exchange,
    symbols: list[str],
    timeframe: str,
    lookback: int,
    horizon: int = 5,
    threshold_pct: float = 1.0,
    labeling_method: LabelingMethod = "threshold",
    take_profit_pct: float = 2.0,
    stop_loss_pct: float = 2.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Birden çok sembolün geçmiş verisinden birleşik eğitim seti üretir.

    Her sembol bağımsız işlenir (göstergeler ve etiketler sembol içi
    hesaplanır), sonra tüm satırlar tek bir X, y çiftinde birleştirilir.

    `labeling_method`:
    - `"threshold"` (varsayılan, geriye dönük uyumlu): `horizon` mum sonraki
      getiriye göre sabit eşikli etiketleme.
    - `"triple_barrier"`: kâr hedefi / stop-loss / zaman aşımı bariyerlerinden
      hangisi önce tetiklenirse ona göre etiketleme (bkz. `labeling.py`) —
      gerçek işlem mantığını daha doğru yansıtır, `take_profit_pct` ve
      `stop_loss_pct` bu barierleri, `horizon` ise zaman aşımı penceresini
      (max_horizon) belirler.
    """
    frames: list[pd.DataFrame] = []

    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, lookback)
            if len(ohlcv) < 60:
                continue
            features = build_features(ohlcv)
            labels = _compute_labels(ohlcv, labeling_method, horizon, threshold_pct, take_profit_pct, stop_loss_pct)
            frame = features.copy()
            frame["label"] = labels
            frame = frame.dropna(subset=FEATURE_COLUMNS + ["label"])
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:  # noqa: BLE001 - tek sembol hatası tüm eğitimi durdurmamalı
            logger.warning("dataset: skipping %s: %s", symbol, exc)
            continue

    if not frames:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype="float")

    combined = pd.concat(frames, ignore_index=True)
    X = combined[FEATURE_COLUMNS]
    y = combined["label"]
    return X, y
