import logging
from typing import Literal

import pandas as pd

from app.backtest.data import timeframe_to_minutes
from app.exchanges.base import Exchange
from app.exchanges.cache import fetch_ohlcv_cached

from .data_quality import warn_if_gaps
from .features import ALL_FEATURE_COLUMNS, FEATURE_COLUMNS, build_features
from .labeling import label_future_direction, triple_barrier_labels
from .macro_features import load_macro_history, merge_macro_features
from .orderbook_features import load_orderbook_history, merge_orderbook_features
from .orderflow_features import merge_taker_flow_features

logger = logging.getLogger(__name__)

LabelingMethod = Literal["threshold", "triple_barrier"]


def _persist_feature_snapshots(symbol: str, timeframe: str, features: pd.DataFrame) -> None:
    """ML eğitimi zaten Binance'ten geniş bir geçmiş çektiği için, bu
    geçmişi `feature_snapshots`'a da yazarak LSTM/RL'nin ihtiyaç duyduğu
    uzun zaman serisini periyodik birikim yerine tek seferde "backfill"
    eder (bkz. `app.db.repository.record_feature_snapshots_bulk`).
    DB kapalı/erişilemezse veya import başarısızsa sessizce atlanır —
    eğitim akışının bir ön koşulu değildir."""
    try:
        from app.db.repository import record_feature_snapshots_bulk

        valid = features.dropna(subset=FEATURE_COLUMNS)
        if not valid.empty:
            record_feature_snapshots_bulk(symbol, timeframe, valid)
    except Exception:  # noqa: BLE001 - DB birikimi opsiyoneldir, eğitimi bozmamalı
        logger.exception("feature snapshot backfill failed for %s", symbol)


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


def _build_symbol_frames(
    exchange: Exchange,
    symbols: list[str],
    timeframe: str,
    lookback: int,
    horizon: int,
    threshold_pct: float,
    labeling_method: LabelingMethod,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> list[pd.DataFrame]:
    """Her sembol için özellik+etiket çerçevesini, kronolojik sırası
    korunmuş ve `time_frac` (0..1, o sembolün serisi içindeki göreli
    zaman konumu) kolonuyla üretir.

    `time_frac`, walk-forward / purged CV bölmelerinin (bkz.
    `app.ml.validation`) semboller arası tutarlı bir "ne kadar yakın
    zaman" ekseni üzerinde çalışabilmesi için vardır — semboller farklı
    sayıda mumla dönebildiğinden mutlak satır indeksi yerine göreli
    konum kullanılır.
    """
    frames: list[pd.DataFrame] = []
    macro_history = load_macro_history()

    for symbol in symbols:
        try:
            ohlcv = fetch_ohlcv_cached(exchange, symbol, timeframe, lookback)
            if len(ohlcv) < 60:
                continue
            warn_if_gaps(symbol, timeframe, ohlcv, timeframe_to_minutes(timeframe))
            features = build_features(ohlcv)
            _persist_feature_snapshots(symbol, timeframe, features)
            features = merge_macro_features(features, macro_history)
            features = merge_orderbook_features(features, load_orderbook_history(symbol))
            features = merge_taker_flow_features(features, ohlcv, exchange, symbol, timeframe)
            labels = _compute_labels(ohlcv, labeling_method, horizon, threshold_pct, take_profit_pct, stop_loss_pct)
            frame = features.copy()
            frame["label"] = labels
            frame["symbol"] = symbol
            frame["time_frac"] = (pd.RangeIndex(len(frame)) / max(len(frame) - 1, 1))
            frame = frame.dropna(subset=FEATURE_COLUMNS + ["label"])
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:  # noqa: BLE001 - tek sembol hatası tüm eğitimi durdurmamalı
            logger.warning("dataset: skipping %s: %s", symbol, exc)
            continue

    return frames


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
    frames = _build_symbol_frames(
        exchange, symbols, timeframe, lookback, horizon, threshold_pct, labeling_method, take_profit_pct, stop_loss_pct
    )
    if not frames:
        return pd.DataFrame(columns=ALL_FEATURE_COLUMNS), pd.Series(dtype="float")

    combined = pd.concat(frames, ignore_index=True)
    X = combined[ALL_FEATURE_COLUMNS]
    y = combined["label"]
    return X, y


def build_training_dataset_with_time(
    exchange: Exchange,
    symbols: list[str],
    timeframe: str,
    lookback: int,
    horizon: int = 5,
    threshold_pct: float = 1.0,
    labeling_method: LabelingMethod = "threshold",
    take_profit_pct: float = 2.0,
    stop_loss_pct: float = 2.0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """`build_training_dataset` ile aynıdır, ek olarak her satır için
    `time_frac`'i de döner — walk-forward/purged CV ve out-of-sample
    holdout bölmeleri (`app.ml.validation`) bunu kullanır.
    """
    frames = _build_symbol_frames(
        exchange, symbols, timeframe, lookback, horizon, threshold_pct, labeling_method, take_profit_pct, stop_loss_pct
    )
    if not frames:
        return pd.DataFrame(columns=ALL_FEATURE_COLUMNS), pd.Series(dtype="float"), pd.Series(dtype="float")

    combined = pd.concat(frames, ignore_index=True)
    X = combined[ALL_FEATURE_COLUMNS]
    y = combined["label"]
    time_frac = combined["time_frac"]
    return X, y, time_frac
