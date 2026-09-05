"""Reinforcement Learning ajanı için epizod verisi hazırlar.

XGBoost/LSTM'in tersine (her bar için ayrı bir etiket/tahmin), RL ajanı
zaman içinde SIRALI kararlar alır (pozisyon aç/tut/kapat) ve ödülü
(reward) yalnızca gerçekleşen PNL'den öğrenir — etiketleme gerekmez.
Bu modül yalnızca özellik matrisini ve kapanış fiyatlarını hazırlar;
karar alma mantığı `app.rl.environment`'tadır.
"""

import numpy as np
from app.exchanges.base import Exchange

from app.ml.features import ALL_FEATURE_COLUMNS, FEATURE_COLUMNS, build_features
from app.ml.macro_features import load_macro_history, merge_macro_features
from app.ml.multi_timeframe_features import MULTI_TIMEFRAME_FEATURE_COLUMNS, compute_multi_timeframe_features
from app.ml.openinterest_features import load_open_interest_history, merge_open_interest_features
from app.ml.orderbook_features import load_orderbook_history, merge_orderbook_features
from app.ml.orderflow_features import merge_taker_flow_features


def build_episode_data(
    exchange: Exchange,
    symbol: str,
    timeframe: str,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bir sembol için kronolojik (karışTIRILMAMIŞ) özellik matrisi ve
    kapanış fiyatı dizisini döner.

    Döner: (X: şekil (n, len(ALL_FEATURE_COLUMNS)), close: şekil (n,))

    Yalnızca teknik özelliklerin (`FEATURE_COLUMNS`) tam dolu olduğu
    satırlar tutulur (warm-up barları elenir); makro/order-book/order-flow
    özellikleri eksikse (henüz yeterli geçmiş birikmediyse ya da borsa
    desteklemiyorsa) 0.0 (nötr) ile doldurulur — XGBoost'un aksine RL
    ajanının kendi ağı NaN kabul etmez.
    """
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, lookback)
    features = build_features(ohlcv)
    features = merge_macro_features(features, load_macro_history())
    features = merge_orderbook_features(features, load_orderbook_history(symbol))
    features = merge_taker_flow_features(features, ohlcv, exchange, symbol, timeframe)
    features = merge_open_interest_features(features, load_open_interest_history(symbol))
    htf_features = compute_multi_timeframe_features(ohlcv)
    for col in MULTI_TIMEFRAME_FEATURE_COLUMNS:
        features[col] = htf_features[col].to_numpy()

    features = features.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    features = features.fillna(0.0)  # kalan NaN'lar yalnızca makro/order-book kolonlarında olabilir

    X = features[ALL_FEATURE_COLUMNS].to_numpy(dtype="float32")
    close = features["close"].to_numpy(dtype="float64")
    return X, close
