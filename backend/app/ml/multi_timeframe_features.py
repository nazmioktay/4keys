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

# GÜNCELLEME (kullanıcı isteği: "kolay olanları ekleyip performansa
# bakalım" — bkz. README "Karlılık"): üst-TF için TEK bir ek gösterge
# ("derinleştirme") — 4h için SuperTrend yönü. `_HTF_SOURCE_COLUMNS`e
# (TÜM üst-TF'lerde AYNI kolonlar) eklenmiyor, çünkü yalnızca BELİRLİ bir
# üst-TF için isteniyor. GÜNCELLEME 2 (korelasyon eleme turu): `1d` için
# `htf_1d_vwap_gap_pct` ÇIKARILDI — gerçek BTC verisiyle `htf_4h_ema_gap`
# (0.926) ve `htf_1d_rsi_norm` (0.926) ile >=0.90 korele çıktı, yeni bilgi
# katmıyordu (bkz. `deploy/check-feature-correlations.sh` çıktısı).
_EXTRA_HTF_COLUMNS = {"4h": ["htf_4h_supertrend_trend"]}

MULTI_TIMEFRAME_FEATURE_COLUMNS = (
    [f"htf_{label}_{col}" for label in MULTI_TIMEFRAME_RULES for col in _HTF_SOURCE_COLUMNS]
    + [col for cols in _EXTRA_HTF_COLUMNS.values() for col in cols]
    + ["htf_weekly_pivot_dist"]
)

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
    bir DataFrame döner.

    `settings.ml_enable_multi_timeframe_features=False` (şu anki varsayılan
    — bkz. README "temel sadeleşme", kullanıcı isteğiyle GEÇİCİ olarak
    durduruldu) iken hiçbir resample/gösterge hesaplaması YAPILMAZ, tüm
    kolonlar nötr (0.0) doner — makro/order-book'ta "veri yoksa nötr" için
    ZATEN kullanılan AYNI desen, burada da yeni bir yanlılık eklenmez."""
    from app.core.config import settings

    if not settings.ml_enable_multi_timeframe_features:
        return pd.DataFrame(0.0, index=ohlcv.index, columns=MULTI_TIMEFRAME_FEATURE_COLUMNS)

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

    # GÜNCELLEME (bkz. `_EXTRA_HTF_COLUMNS` tanımı): SuperTrend,
    # `compute_indicators` (ucuz screener versiyonu) İÇİNDE yok — ama
    # `average_true_range`/`supertrend` de (Hurst/Ichimoku/Nadaraya-Watson
    # gibi PAHALI göstergelerin aksine) TEK BAŞINA ucuz fonksiyonlar,
    # `build_features`'ın TAMAMINI çağırmadan buradan da kullanılabilirler
    # — aynı performans gerekçesi geçerliliğini korur.
    from .advanced_indicators import supertrend

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

        # `_EXTRA_HTF_COLUMNS`: yalnızca BELİRLİ üst-TF'ler için tek bir
        # ek gösterge (bkz. tanım) — `features.py`'daki `supertrend_trend`
        # ile BİREBİR AYNI dönüşüm.
        for extra_col in _EXTRA_HTF_COLUMNS.get(label, []):
            if extra_col.endswith("supertrend_trend"):
                right[extra_col] = supertrend(htf_ohlcv)["supertrend_trend"].to_numpy()

        merged = pd.merge_asof(left_sorted, right, on="timestamp", direction="backward")
        merged = merged.sort_values("_order").reset_index(drop=True)
        for col in _HTF_SOURCE_COLUMNS:
            result[f"htf_{label}_{col}"] = merged[col].to_numpy()
        for extra_col in _EXTRA_HTF_COLUMNS.get(label, []):
            result[extra_col] = merged[extra_col].to_numpy()

        # `label == "1d"` iken üst-TF OHLCV zaten günlük — haftalık pivot
        # için AYNI günlük veriyi yeniden kullanıp haftaya resample ederiz
        # (borsadan tekrar veri çekmeye gerek yok).
        if label == "1d":
            weekly = _resample_ohlcv(ohlcv, "1W")
            if len(weekly) >= 3:
                # Klasik pivot: bu haftanın pivotu ÖNCEKİ HAFTANIN
                # (H+L+C)/3'üdür — `shift(1)` ile bir önceki tamamlanmış
                # haftaya kayarız, bu haftanın barları henüz kapanmadan
                # pivot değeri BİLİNMEZ (look-ahead YOK).
                prev_pivot = ((weekly["high"] + weekly["low"] + weekly["close"]) / 3).shift(1)
                weekly_right = pd.DataFrame({"timestamp": pd.to_datetime(weekly["timestamp"]), "pivot": prev_pivot.to_numpy()})
                weekly_merged = pd.merge_asof(left_sorted, weekly_right, on="timestamp", direction="backward")
                weekly_merged = weekly_merged.sort_values("_order").reset_index(drop=True)
                close_now = ohlcv["close"].reset_index(drop=True)
                dist = ((close_now - weekly_merged["pivot"]) / close_now).clip(-0.1, 0.1) * 10
                result["htf_weekly_pivot_dist"] = dist.fillna(0.0).to_numpy()

    return result
