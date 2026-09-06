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
    # PERFORMANS: burada `app.ml.features.build_features` ÇAĞRILMAZ. O
    # fonksiyon 42 gösterge (Hurst, Ichimoku, Nadaraya-Watson, dinamik
    # destek/direnç...) hesaplar ve 10.000 barlık bir seride ~2.2 saniye
    # sürer — oysa buradan yalnızca İKİ kolon (`ema_gap`, `rsi_norm`)
    # kullanılıyor. `compute_indicators` aynı EMA/RSI'yı ~0.006 saniyede
    # üretiyor (~360x hızlı). Bu, `_build_symbol_frames`/`build_sequence_dataset`
    # üzerinden eğitim başına onlarca kez çağrıldığı için eğitim süresine
    # DOĞRUDAN yansıyordu; ayrıca canlı karar döngüsünde de sembol başına
    # her turda ödeniyordu. Dönüşümler `features.py`'daki tanımlarla
    # BİREBİR aynı tutulur (bkz. aşağıdaki yorumlar).
    from app.screener.indicators import compute_indicators

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
        ind = compute_indicators(htf_ohlcv)
        # `app.ml.features.build_features` ile BİREBİR AYNI dönüşümler:
        #   rsi_norm = (rsi - 50) / 50
        #   ema_gap  = ((ema_fast - ema_slow) / close).clip(-0.1, 0.1) * 10
        htf_values = {
            "rsi_norm": (ind["rsi"] - 50) / 50,
            "ema_gap": ((ind["ema_fast"] - ind["ema_slow"]) / ind["close"]).clip(-0.1, 0.1) * 10,
        }

        right = pd.DataFrame({"timestamp": pd.to_datetime(htf_ohlcv["timestamp"])})
        for col in _HTF_SOURCE_COLUMNS:
            right[col] = htf_values[col].to_numpy()

        merged = pd.merge_asof(left_sorted, right, on="timestamp", direction="backward")
        merged = merged.sort_values("_order").reset_index(drop=True)
        for col in _HTF_SOURCE_COLUMNS:
            result[f"htf_{label}_{col}"] = merged[col].to_numpy()

    return result
