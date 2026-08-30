import logging

import pandas as pd

from app.exchanges.base import Exchange

from .features import FEATURE_COLUMNS, build_features
from .labeling import label_future_direction

logger = logging.getLogger(__name__)


def build_training_dataset(
    exchange: Exchange,
    symbols: list[str],
    timeframe: str,
    lookback: int,
    horizon: int = 5,
    threshold_pct: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Birden çok sembolün geçmiş verisinden birleşik eğitim seti üretir.

    Her sembol bağımsız işlenir (göstergeler ve etiketler sembol içi
    hesaplanır), sonra tüm satırlar tek bir X, y çiftinde birleştirilir.
    """
    frames: list[pd.DataFrame] = []

    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, lookback)
            if len(ohlcv) < 60:
                continue
            features = build_features(ohlcv)
            labels = label_future_direction(ohlcv["close"], horizon, threshold_pct)
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
