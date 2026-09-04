import numpy as np
import pandas as pd

from app.screener.indicators import compute_indicators

from .advanced_indicators import (
    adx,
    average_true_range,
    bollinger_bands,
    dynamic_support_resistance,
    fibonacci_retracement_position,
    heikin_ashi,
    ichimoku_cloud,
    linear_regression_channel,
    mavilim_w,
    nadaraya_watson_envelope,
    on_balance_volume,
    pmax,
    rolling_hurst_exponent,
    rolling_vwap,
    stoch_rsi_log,
    supertrend,
    wavetrend,
)
from .orderflow_features import TAKER_FLOW_FEATURE_COLUMNS

FEATURE_COLUMNS = [
    "rsi_norm",
    "macd_hist_norm",
    "ema_gap",
    "momentum",
    "volume_ratio",
    "price_position",
    "return_1",
    "return_3",
    "return_5",
    # --- Kullanıcının manuel işlemde kullandığı ek göstergeler ---
    "ha_trend",
    "ha_body_pct",
    "stoch_rsi_k",
    "stoch_rsi_d",
    "mavilim_gap",
    "pmax_trend",
    "pmax_dist_pct",
    "linreg_zscore",
    "linreg_slope_norm",
    "wt_diff_norm",
    "wt_cross",
    "nwe_position",
    "sr_dist_support_pct",
    "sr_dist_resistance_pct",
    "sr_level_count_norm",
    # --- OHLC mum yapısı (ölçeklenmiş, ham fiyat değil) ---
    "candle_body_pct",
    "candle_upper_wick_pct",
    "candle_lower_wick_pct",
    "true_range_pct",
    # --- Ek TradingView göstergeleri ---
    "atr_pct",
    "bb_percent_b",
    "bb_bandwidth_norm",
    "adx_norm",
    "di_diff_norm",
    "vwap_gap_pct",
    "obv_slope_norm",
    "supertrend_trend",
    "supertrend_dist_pct",
    "ichimoku_cloud_position",
    "ichimoku_tk_cross",
    "fib_retracement_position",
    # --- Ham hacim büyüklüğü (oran değil, hacmin kendisinin anormalliği) ---
    "volume_zscore",
    # --- Fraktal analiz: piyasanın "hafızası" (trend-devamlılığı mı,
    # ortalamaya-dönüş mü) — bkz. rolling_hurst_exponent docstring'i ---
    "hurst_exponent",
]

# Makro/piyasa bağlamı özellikleri (app.macro.service ile toplanan
# `macro_snapshots` tablosundan, zaman bazlı en-yakın-geçmiş eşleştirmeyle
# (as-of merge) eklenir — bkz. `app.ml.macro_features`). Ayrı bir liste
# olarak tutulur çünkü DB geçmişi kısaysa (henüz yeni toplanmaya
# başlandıysa) bu kolonlar geriye dönük olarak yalnızca yaklaşık
# (en-eski-bilinen-değerle doldurulmuş) değerler taşıyabilir.
MACRO_FEATURE_COLUMNS = [
    "macro_total_market_cap_norm",
    "macro_btc_dominance_norm",
    "macro_funding_rate_btc",
    "macro_vix_norm",
    "macro_gold_norm",
    "macro_sp500_norm",
    "macro_nasdaq_norm",
    "macro_nikkei_norm",
    "macro_dax_norm",
    "macro_fed_funds_rate_norm",
    "macro_ecb_deposit_rate_norm",
]

# Emir defteri (order book) özellikleri: geçmişi olmayan, bugünden
# itibaren periyodik toplanan bir kaynak (bkz. app.orderbook,
# app.ml.orderbook_features) — makrodan farklı olarak SEMBOL BAZINDA.
ORDERBOOK_FEATURE_COLUMNS = [
    "orderbook_imbalance",
    "orderbook_spread_norm",
    "orderbook_depth_norm",
]

# Mum bazlı agresif alım/satım akışı (taker buy ratio) — bkz.
# `app.ml.orderflow_features`. ORDERBOOK_FEATURE_COLUMNS'un (anlık emir
# defteri görüntüsü, geçmişi yok) aksine bu TAM GEÇMİŞE sahiptir, ama
# yine de opsiyonel/NaN-toleranslı tutulur çünkü her borsa adapter'ı
# desteklemez.

# Modelin gerçekten gördüğü tüm girdi kolonları (teknik + makro + order book + order flow).
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + MACRO_FEATURE_COLUMNS + ORDERBOOK_FEATURE_COLUMNS + TAKER_FLOW_FEATURE_COLUMNS


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """OHLCV verisinden ML modeli için özellik matrisi üretir.

    `compute_indicators`'ı çağırıp göstergeleri ölçekli/normalize edilmiş
    özelliklere dönüştürür. Dönen DataFrame, kaynak OHLCV ile aynı index'e
    sahiptir; ilk satırlar (rolling pencereler dolmadığı için) NaN içerir.
    """
    ind = compute_indicators(ohlcv)

    features = pd.DataFrame(index=ind.index)
    features["rsi_norm"] = (ind["rsi"] - 50) / 50  # -1..1
    features["macd_hist_norm"] = (ind["macd_hist"] / ind["close"]).clip(-0.05, 0.05) * 20
    features["ema_gap"] = ((ind["ema_fast"] - ind["ema_slow"]) / ind["close"]).clip(-0.1, 0.1) * 10
    features["momentum"] = (ind["momentum"] / 20).clip(-5, 5)
    features["volume_ratio"] = (ind["volume"] / ind["volume_sma"]).clip(0, 5)

    # Ham hacim büyüklüğü: log-dönüşümle ölçeklenip kendi rolling
    # ortalama/std'sine göre z-skorlanır — `volume_ratio` (kısa vadeli MA'ya
    # oran) farklı olarak, hacmin MUTLAK büyüklüğündeki anormallikleri
    # (ör. ani hacim patlaması) daha geniş bir pencerede yakalar. Ham
    # sayı (ör. "150000") yerine yine ölçekten bağımsız bir değer taşınır.
    log_volume = np.log1p(ind["volume"])
    vol_mean = log_volume.rolling(50, min_periods=10).mean()
    vol_std = log_volume.rolling(50, min_periods=10).std().replace(0, float("nan"))
    # hacim (nadiren de olsa) uzun bir pencerede tamamen sabit kalırsa
    # std=0/NaN olur — bu durumda "anormallik yok" anlamına gelen 0.0'a
    # (nötr) düşülür, tüm satırların NaN olup elenmesi yerine.
    features["volume_zscore"] = ((log_volume - vol_mean) / vol_std).clip(-5, 5).fillna(0.0)

    rolling_low = ind["low"].rolling(20).min()
    rolling_high = ind["high"].rolling(20).max()
    span = (rolling_high - rolling_low).replace(0, float("nan"))
    features["price_position"] = ((ind["close"] - rolling_low) / span).clip(0, 1)

    features["return_1"] = ind["close"].pct_change(1).clip(-0.2, 0.2) * 10
    features["return_3"] = ind["close"].pct_change(3).clip(-0.2, 0.2) * 10
    features["return_5"] = ind["close"].pct_change(5).clip(-0.2, 0.2) * 10

    # --- Heikin Ashi ---
    ha = heikin_ashi(ohlcv)
    features["ha_trend"] = np.where(ha["ha_close"] > ha["ha_open"], 1.0, -1.0)
    features["ha_body_pct"] = ((ha["ha_close"] - ha["ha_open"]) / ohlcv["close"]).clip(-0.05, 0.05) * 20

    # --- Stochastic RSI (log-getiri) ---
    sr = stoch_rsi_log(ohlcv["close"])
    features["stoch_rsi_k"] = sr["stoch_rsi_k"] * 2 - 1  # 0..1 -> -1..1
    features["stoch_rsi_d"] = sr["stoch_rsi_d"] * 2 - 1

    # --- MavilimW ---
    mav = mavilim_w(ohlcv["close"])
    features["mavilim_gap"] = ((ohlcv["close"] - mav) / ohlcv["close"]).clip(-0.1, 0.1) * 10

    # --- PMax ---
    pm = pmax(ohlcv)
    features["pmax_trend"] = pm["pmax_trend"]
    features["pmax_dist_pct"] = ((ohlcv["close"] - pm["pmax"]) / ohlcv["close"]).clip(-0.1, 0.1) * 10

    # --- Doğrusal Regresyon Kanalı ---
    lr = linear_regression_channel(ohlcv["close"], length=min(100, max(len(ohlcv) // 2, 10)))
    lr_std_safe = lr["linreg_std"].replace(0, float("nan"))
    features["linreg_zscore"] = ((ohlcv["close"] - lr["linreg_mid"]) / lr_std_safe).clip(-5, 5)
    features["linreg_slope_norm"] = (lr["linreg_slope"] / ohlcv["close"] * 100).clip(-5, 5)

    # --- WaveTrend (LazyBear) ---
    wt = wavetrend(ohlcv)
    features["wt_diff_norm"] = (wt["wt1"] - wt["wt2"]).clip(-50, 50) / 10
    features["wt_cross"] = wt["wt_cross"]

    # --- Nadaraya-Watson Envelope (causal) ---
    nwe = nadaraya_watson_envelope(ohlcv["close"])
    nwe_span = (nwe["nwe_upper"] - nwe["nwe_lower"]).replace(0, float("nan"))
    features["nwe_position"] = (((ohlcv["close"] - nwe["nwe_mid"]) / nwe_span) * 2).clip(-3, 3)

    # --- Dynamic Support/Resistance ---
    sr_levels = dynamic_support_resistance(ohlcv)
    features["sr_dist_support_pct"] = sr_levels["sr_dist_support_pct"].clip(0, 20).fillna(20)
    features["sr_dist_resistance_pct"] = sr_levels["sr_dist_resistance_pct"].clip(0, 20).fillna(20)
    features["sr_level_count_norm"] = (sr_levels["sr_level_count"] / 5).clip(0, 3)

    # --- OHLC mum yapısı (ham fiyat yerine ölçeklenmiş oranlar) ---
    candle_range = (ohlcv["high"] - ohlcv["low"]).replace(0, float("nan"))
    features["candle_body_pct"] = ((ohlcv["close"] - ohlcv["open"]) / ohlcv["close"]).clip(-0.05, 0.05) * 20
    features["candle_upper_wick_pct"] = ((ohlcv["high"] - ohlcv[["open", "close"]].max(axis=1)) / candle_range).clip(0, 1)
    features["candle_lower_wick_pct"] = ((ohlcv[["open", "close"]].min(axis=1) - ohlcv["low"]) / candle_range).clip(0, 1)

    # --- ATR (volatilite / stop-loss mesafesi) ---
    atr = average_true_range(ohlcv)
    features["atr_pct"] = (atr / ohlcv["close"]).clip(0, 0.2) * 10
    features["true_range_pct"] = (candle_range / ohlcv["close"]).clip(0, 0.2) * 10

    # --- Bollinger Bands (volatilite) ---
    bb = bollinger_bands(ohlcv["close"])
    features["bb_percent_b"] = bb["bb_percent_b"]
    features["bb_bandwidth_norm"] = bb["bb_bandwidth"].clip(0, 0.5) * 4

    # --- ADX (trend gücü) ---
    adx_df = adx(ohlcv)
    features["adx_norm"] = (adx_df["adx"] / 50).clip(0, 2)
    di_span = (adx_df["plus_di"] + adx_df["minus_di"]).replace(0, float("nan"))
    features["di_diff_norm"] = ((adx_df["plus_di"] - adx_df["minus_di"]) / di_span).clip(-1, 1)

    # --- VWAP (hacim ağırlıklı ortalama fiyat) ---
    vwap = rolling_vwap(ohlcv)
    features["vwap_gap_pct"] = ((ohlcv["close"] - vwap) / ohlcv["close"]).clip(-0.1, 0.1) * 10

    # --- OBV (hacim akışı) ---
    obv = on_balance_volume(ohlcv)
    obv_std = obv.diff().rolling(20).std().replace(0, float("nan"))
    features["obv_slope_norm"] = (obv.diff(5) / (obv_std * np.sqrt(5))).clip(-5, 5).fillna(0)

    # --- SuperTrend ---
    st = supertrend(ohlcv)
    features["supertrend_trend"] = st["supertrend_trend"]
    features["supertrend_dist_pct"] = ((ohlcv["close"] - st["supertrend"]) / ohlcv["close"]).clip(-0.1, 0.1) * 10

    # --- Ichimoku Bulutu ---
    ichi = ichimoku_cloud(ohlcv)
    cloud_top = ichi[["senkou_a", "senkou_b"]].max(axis=1)
    cloud_bottom = ichi[["senkou_a", "senkou_b"]].min(axis=1)
    cloud_span = (cloud_top - cloud_bottom).replace(0, float("nan"))
    features["ichimoku_cloud_position"] = (((ohlcv["close"] - cloud_bottom) / cloud_span) * 2 - 1).clip(-3, 3)
    features["ichimoku_tk_cross"] = np.where(ichi["tenkan"] > ichi["kijun"], 1.0, -1.0)

    # --- Fibonacci Geri Çekilme ---
    features["fib_retracement_position"] = fibonacci_retracement_position(ohlcv)

    # --- Fraktal analiz (Hurst üsteli) — bkz. rolling_hurst_exponent ---
    features["hurst_exponent"] = rolling_hurst_exponent(ohlcv["close"]).fillna(0.5)

    features["close"] = ind["close"]
    # int64 (HER ZAMAN nanosaniye epoch, çözünürlükten bağımsız) olarak
    # saklanır, datetime64 değil: DataFrame'den tek satır çekildiğinde
    # (ör. `.iloc[-1]`, `latest_feature_vector`) pandas bir Series oluşturur
    # ve datetime64 ile float64 kolonların karışımı zorla `object` dtype'a
    # düşürür — bu da XGBoost'un katı dtype kontrolünü kırar. int64,
    # float64 ile uyumlu şekilde yükseltilir. `astype("datetime64[ns]")` ile
    # ÖNCE ns'e sabitlenir — pandas 3.x'te `pd.to_datetime(..., unit="ms")`
    # varsayılan olarak `datetime64[ms]` üretir; doğrudan `.astype("int64")`
    # çağrılsaydı ms epoch elde edilir ve geri dönüşte ns sanılırsa
    # (`pd.Timestamp(value)`, unit belirtilmezse ns varsayar) 1970'e yakın
    # yanlış zaman damgaları üretilirdi.
    features["timestamp"] = ohlcv["timestamp"].astype("datetime64[ns]").astype("int64").to_numpy()
    return features


def latest_feature_vector(ohlcv: pd.DataFrame) -> pd.Series | None:
    """Canlı tahmin için en son (tam dolu) özellik satırını döner."""
    features = build_features(ohlcv).dropna()
    if features.empty:
        return None
    return features.iloc[-1]
