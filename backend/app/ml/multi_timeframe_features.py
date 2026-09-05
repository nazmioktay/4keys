"""1h (veya hangi zaman dilimi eğitim/canlı/backtest'te kullanılıyorsa)
birincil zaman dilimine, ÜST zaman dilimlerinin (4h, 1d) trend bağlamını
ekler — "üst zaman diliminin trendi yönünde işlem yap" prensibinin basit
bir sayısal karşılığı.

Üst-TF barlar AYRI bir borsa çağrısı GEREKTİRMEZ — kaynak OHLCV'den
(genelde 1h) `pandas.resample` ile türetilir. KRİTİK: `label="right",
closed="left"` ile her üst-TF bar, o barın TAMAMEN KAPANDIĞI zamanla
etiketlenir (ör. [08:00,12:00) aralığı "12:00" etiketini alır — 12:00'den
ÖNCE bu bar hakkında hiçbir şey bilinemez). 1h çerçeveye geri eşlerken
`merge_asof(direction="backward")` kullanılır, bu yüzden bir 1h bar
YALNIZCA o ana kadar TAMAMEN KAPANMIŞ üst-TF barları görebilir — geleceğe
bakma (look-ahead bias) YARATILMAZ. Bu, macro/orderbook/open-interest'ten
FARKLI olarak harici bir veri kaynağına bağlı değildir, bu yüzden
backtest'te de (bkz. `app.backtest.system_runner`) GERÇEK (basitleştirme
gerektirmeyen) değerlerle hesaplanabilir."""

import pandas as pd

# Kaynak zaman diliminden ("1h" varsayılır, ama herhangi bir kaynak için
# çalışır) türetilecek üst zaman dilimleri. pandas resample kuralları.
MULTI_TIMEFRAME_RULES = {"4h": "4h", "1d": "1D"}

# Üst-TF'den taşınan, trend YÖNÜ + momentum'u özetleyen küçük bir alt küme
# (bkz. `app.ml.features.build_features` — TÜM 42 gösterge değil, yalnızca
# en trend-temsilci ikisi: EMA hızlı/yavaş farkı + RSI).
_HTF_SOURCE_COLUMNS = ["ema_gap", "rsi_norm"]

MULTI_TIMEFRAME_FEATURE_COLUMNS = [f"htf_{label}_{col}" for label in MULTI_TIMEFRAME_RULES for col in _HTF_SOURCE_COLUMNS]

# Üst-TF göstergelerinin (rolling pencereler) makul şekilde ısınması için
# gereken minimum üst-TF bar sayısı — bundan azsa o zaman dilimi atlanır
# (kolonlar NaN kalır), `_MIN_CANDLES` (system_runner) ile aynı gerekçe.
_MIN_HTF_BARS = 60


def _resample_ohlcv(ohlcv: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = ohlcv.set_index("timestamp")
    resampled = (
        df.resample(rule, label="right", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    return resampled.reset_index()


def compute_multi_timeframe_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """`ohlcv` ("timestamp" kolonu gerçek datetime olmalı) ile AYNI
    index/uzunlukta, `MULTI_TIMEFRAME_FEATURE_COLUMNS` kolonlarını içeren
    bir DataFrame döner."""
    from .features import build_features  # döngüsel bağımlılığı önlemek için yerel import (bkz. orderbook_features.py aynı desen)

    result = pd.DataFrame(index=ohlcv.index)
    for col in MULTI_TIMEFRAME_FEATURE_COLUMNS:
        result[col] = float("nan")

    left = pd.DataFrame({"timestamp": pd.to_datetime(ohlcv["timestamp"])})
    left["_order"] = range(len(left))
    left_sorted = left.sort_values("timestamp")

    for label, rule in MULTI_TIMEFRAME_RULES.items():
        htf_ohlcv = _resample_ohlcv(ohlcv, rule)
        if len(htf_ohlcv) < _MIN_HTF_BARS:
            continue
        htf_features = build_features(htf_ohlcv)

        right = pd.DataFrame({"timestamp": pd.to_datetime(htf_ohlcv["timestamp"])})
        for col in _HTF_SOURCE_COLUMNS:
            right[col] = htf_features[col].to_numpy()

        merged = pd.merge_asof(left_sorted, right, on="timestamp", direction="backward")
        merged = merged.sort_values("_order").reset_index(drop=True)
        for col in _HTF_SOURCE_COLUMNS:
            result[f"htf_{label}_{col}"] = merged[col].to_numpy()

    return result
