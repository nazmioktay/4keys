"""Kullanıcının manuel işlemde kullandığı TradingView göstergelerinin
Python/pandas karşılıkları. Her fonksiyon saf (yalnızca geçmiş veriye
bakar, geleceğe bakmaz — "repaint" etmez) ve OHLCV DataFrame'i alıp bir
pandas Series/DataFrame döner; `app.ml.features` bunlardan sayısal
öznitelik (feature) türetir.
"""

import numpy as np
import pandas as pd


def heikin_ashi(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Heikin Ashi mumlarını hesaplar (gürültüyü azaltan sentetik mumlar)."""
    ha_close = (ohlcv["open"] + ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 4
    ha_open = pd.Series(index=ohlcv.index, dtype="float64")
    ha_open.iloc[0] = (ohlcv["open"].iloc[0] + ohlcv["close"].iloc[0]) / 2
    for i in range(1, len(ohlcv)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = pd.concat([ohlcv["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([ohlcv["low"], ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({"ha_open": ha_open, "ha_high": ha_high, "ha_low": ha_low, "ha_close": ha_close})


def stoch_rsi_log(close: pd.Series, rsi_length: int = 14, stoch_length: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    """Stochastic RSI, ancak RSI ham fiyat farkı yerine LOG-getiri
    (`ln(close/close_prev)`) üzerinden hesaplanır — büyük fiyatlı
    varlıklarda (BTC gibi) ölçek-bağımsız ve istatistiksel olarak daha
    tutarlıdır.
    """
    log_return = np.log(close / close.shift(1))
    gain = log_return.clip(lower=0)
    loss = -log_return.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = (100 - (100 / (1 + rs))).fillna(50)

    rsi_min = rsi.rolling(stoch_length).min()
    rsi_max = rsi.rolling(stoch_length).max()
    span = (rsi_max - rsi_min).replace(0, float("nan"))
    stoch = ((rsi - rsi_min) / span).clip(0, 1).fillna(0.5)

    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return pd.DataFrame({"stoch_rsi_k": k, "stoch_rsi_d": d})


def _wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def mavilim_w(close: pd.Series, f1: int = 3, f2: int = 5, f3: int = 7, f4: int = 9, f5: int = 11, f6: int = 13) -> pd.Series:
    """MavilimW: kademeli WMA zinciri (her katman bir öncekinin WMA'sı)."""
    m1 = _wma(close, f1)
    m2 = _wma(m1, f2)
    m3 = _wma(m2, f3)
    m4 = _wma(m3, f4)
    m5 = _wma(m4, f5)
    m6 = _wma(m5, f6)
    return m6


def pmax(ohlcv: pd.DataFrame, atr_length: int = 10, multiplier: float = 3.0, ma_length: int = 10) -> pd.DataFrame:
    """PMax: bir hareketli ortalama üzerine sarılmış ATR bantlı trend
    takip göstergesi (SuperTrend'e benzer, MA ile yumuşatılmış)."""
    ma = ohlcv["close"].ewm(span=ma_length, adjust=False).mean()

    high, low, close = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_length, adjust=False).mean()

    upper = ma + multiplier * atr
    lower = ma - multiplier * atr

    pmax_line = pd.Series(index=ohlcv.index, dtype="float64")
    trend = pd.Series(index=ohlcv.index, dtype="float64")
    pmax_line.iloc[0] = lower.iloc[0]
    trend.iloc[0] = 1.0

    for i in range(1, len(ohlcv)):
        if ma.iloc[i] > pmax_line.iloc[i - 1]:
            pmax_line.iloc[i] = max(lower.iloc[i], pmax_line.iloc[i - 1]) if trend.iloc[i - 1] == 1 else lower.iloc[i]
            trend.iloc[i] = 1.0
        else:
            pmax_line.iloc[i] = min(upper.iloc[i], pmax_line.iloc[i - 1]) if trend.iloc[i - 1] == -1 else upper.iloc[i]
            trend.iloc[i] = -1.0

    return pd.DataFrame({"pmax": pmax_line, "pmax_trend": trend})


def linear_regression_channel(close: pd.Series, length: int = 100) -> pd.DataFrame:
    """Rolling pencerede doğrusal regresyon (OLS) fit eder; kanalın orta
    hattı, eğimi ve kalıntıların standart sapması (bant genişliği) döner."""
    x = np.arange(length)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _fit(window: np.ndarray) -> tuple[float, float]:
        y_mean = window.mean()
        slope = ((x - x_mean) * (window - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        fitted = intercept + slope * x
        resid_std = float(np.std(window - fitted))
        last_value = float(fitted[-1])
        return last_value, slope, resid_std  # type: ignore[return-value]

    mid = pd.Series(index=close.index, dtype="float64")
    slope_s = pd.Series(index=close.index, dtype="float64")
    std_s = pd.Series(index=close.index, dtype="float64")

    values = close.to_numpy()
    for i in range(length - 1, len(values)):
        window = values[i - length + 1 : i + 1]
        last_value, slope, resid_std = _fit(window)
        mid.iloc[i] = last_value
        slope_s.iloc[i] = slope
        std_s.iloc[i] = resid_std

    return pd.DataFrame({"linreg_mid": mid, "linreg_slope": slope_s, "linreg_std": std_s})


def wavetrend(ohlcv: pd.DataFrame, channel_length: int = 10, avg_length: int = 21) -> pd.DataFrame:
    """WaveTrend (LazyBear) osilatörü: wt1/wt2 ve kesişim sinyali."""
    hlc3 = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3
    esa = hlc3.ewm(span=channel_length, adjust=False).mean()
    d = (hlc3 - esa).abs().ewm(span=channel_length, adjust=False).mean()
    ci = (hlc3 - esa) / (0.015 * d.replace(0, float("nan")))
    wt1 = ci.ewm(span=avg_length, adjust=False).mean().fillna(0)
    wt2 = wt1.rolling(4).mean().fillna(wt1)

    cross = pd.Series(0, index=ohlcv.index, dtype="float64")
    diff = wt1 - wt2
    cross[(diff > 0) & (diff.shift(1) <= 0)] = 1.0
    cross[(diff < 0) & (diff.shift(1) >= 0)] = -1.0

    return pd.DataFrame({"wt1": wt1, "wt2": wt2, "wt_cross": cross})


def nadaraya_watson_envelope(close: pd.Series, bandwidth: float = 8.0, mult: float = 3.0, window: int = 200) -> pd.DataFrame:
    """Nadaraya-Watson kernel regresyonu (Gaussian çekirdek), yalnızca
    geçmiş veriye bakan (repaint etmeyen/causal) versiyon: her nokta,
    kendisinden önceki `window` bar içinde ağırlıklı ortalama olarak
    hesaplanır — LuxAlgo'nun varsayılan (geleceğe bakan) versiyonundan
    farklı olarak canlı işlemde güvenle kullanılabilir.
    """
    values = close.to_numpy()
    n = len(values)
    smoothed = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        segment = values[start : i + 1]
        distances = np.arange(len(segment) - 1, -1, -1)
        weights = np.exp(-(distances**2) / (2 * bandwidth**2))
        smoothed[i] = np.dot(segment, weights) / weights.sum()

    smoothed_s = pd.Series(smoothed, index=close.index)
    resid_std = (close - smoothed_s).rolling(window, min_periods=1).std().fillna(0)
    upper = smoothed_s + mult * resid_std
    lower = smoothed_s - mult * resid_std

    return pd.DataFrame({"nwe_mid": smoothed_s, "nwe_upper": upper, "nwe_lower": lower})


def _true_range(ohlcv: pd.DataFrame) -> pd.Series:
    high, low, close = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def average_true_range(ohlcv: pd.DataFrame, length: int = 14) -> pd.Series:
    """ATR (Average True Range) — volatilite ölçüsü, stop-loss mesafesi
    belirlemede de kullanılır."""
    return _true_range(ohlcv).ewm(alpha=1 / length, adjust=False).mean()


def bollinger_bands(close: pd.Series, length: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: hareketli ortalama +/- N standart sapma bantları."""
    mid = close.rolling(length).mean()
    std = close.rolling(length).std()
    upper = mid + mult * std
    lower = mid - mult * std
    span = (upper - lower).replace(0, float("nan"))
    percent_b = ((close - lower) / span).clip(-1, 2)
    bandwidth = (span / mid.replace(0, float("nan")))
    return pd.DataFrame({"bb_upper": upper, "bb_lower": lower, "bb_mid": mid, "bb_percent_b": percent_b, "bb_bandwidth": bandwidth})


def adx(ohlcv: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """ADX (Average Directional Index) — trend gücü (yönsüz), +DI/-DI ile
    birlikte trendin yönü de çıkarılabilir."""
    high, low = ohlcv["high"], ohlcv["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=ohlcv.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=ohlcv.index)

    tr = _true_range(ohlcv)
    atr = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr.replace(0, float("nan"))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    adx_line = dx.ewm(alpha=1 / length, adjust=False).mean()
    return pd.DataFrame({"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di})


def rolling_vwap(ohlcv: pd.DataFrame, length: int = 20) -> pd.Series:
    """Hacim ağırlıklı ortalama fiyat (VWAP), kayan pencerede — kripto
    perpetual'larda gün içi seans sıfırlaması olmadığından rolling
    pencere kullanılır (causal, geleceğe bakmaz)."""
    typical = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3
    pv = typical * ohlcv["volume"]
    return pv.rolling(length).sum() / ohlcv["volume"].rolling(length).sum().replace(0, float("nan"))


def on_balance_volume(ohlcv: pd.DataFrame) -> pd.Series:
    """OBV (On-Balance Volume): fiyat yükselirken hacmi ekler, düşerken
    çıkarır — hacim akışının fiyat yönüyle uyumlu olup olmadığını gösterir."""
    direction = np.sign(ohlcv["close"].diff().fillna(0.0))
    return (direction * ohlcv["volume"]).cumsum()


def supertrend(ohlcv: pd.DataFrame, atr_length: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """SuperTrend: ATR bantlı, klasik trend takip göstergesi."""
    atr = average_true_range(ohlcv, atr_length)
    hl2 = (ohlcv["high"] + ohlcv["low"]) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    close = ohlcv["close"].to_numpy()
    ub = upper_band.to_numpy(copy=True)
    lb = lower_band.to_numpy(copy=True)
    n = len(close)

    line = np.full(n, np.nan)
    trend = np.ones(n)
    line[0] = lb[0]
    for i in range(1, n):
        lb[i] = max(lb[i], line[i - 1]) if trend[i - 1] == 1 and close[i - 1] > line[i - 1] else lb[i]
        ub[i] = min(ub[i], line[i - 1]) if trend[i - 1] == -1 and close[i - 1] < line[i - 1] else ub[i]
        if trend[i - 1] == 1:
            trend[i] = -1.0 if close[i] < lb[i] else 1.0
        else:
            trend[i] = 1.0 if close[i] > ub[i] else -1.0
        line[i] = lb[i] if trend[i] == 1 else ub[i]

    return pd.DataFrame({"supertrend": line, "supertrend_trend": trend}, index=ohlcv.index)


def ichimoku_cloud(ohlcv: pd.DataFrame, tenkan_len: int = 9, kijun_len: int = 26, senkou_b_len: int = 52) -> pd.DataFrame:
    """Ichimoku Bulutu: Tenkan-sen, Kijun-sen ve bulut (Senkou A/B)
    sınırları — geleceğe kaydırma (plotting offset) uygulanmadan, yalnızca
    o ana kadarki veriyle (causal) hesaplanır."""
    high, low = ohlcv["high"], ohlcv["low"]
    tenkan = (high.rolling(tenkan_len).max() + low.rolling(tenkan_len).min()) / 2
    kijun = (high.rolling(kijun_len).max() + low.rolling(kijun_len).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(senkou_b_len).max() + low.rolling(senkou_b_len).min()) / 2
    return pd.DataFrame({"tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a, "senkou_b": senkou_b})


def fibonacci_retracement_position(ohlcv: pd.DataFrame, length: int = 100) -> pd.Series:
    """Son `length` bardaki en yüksek/en düşük arasında, güncel fiyatın
    Fibonacci geri çekilme seviyeleri cinsinden konumu (0 = dip, 1 = tepe;
    0.382/0.5/0.618 klasik retracement seviyeleridir)."""
    swing_high = ohlcv["high"].rolling(length).max()
    swing_low = ohlcv["low"].rolling(length).min()
    span = (swing_high - swing_low).replace(0, float("nan"))
    return ((ohlcv["close"] - swing_low) / span).clip(0, 1)


def dynamic_support_resistance(
    ohlcv: pd.DataFrame, pivot_lookback: int = 10, channel_lookback: int = 284, channel_width_pct: float = 10.0, min_pivots: int = 2
) -> pd.DataFrame:
    """LonesomeTheBlue'nun pivot-kümeleme tabanlı dinamik Destek/Direnç
    göstergesi. Bir pivot yalnızca `pivot_lookback` bar sonra onaylanır
    (doğal gecikme — geleceğe bakma yok). `channel_lookback` bar içindeki
    pivotlar, aralığın `channel_width_pct`'i genişliğindeki kanallara
    kümelenir; bir kanalda en az `min_pivots` pivot varsa Destek/Direnç
    seviyesi sayılır.

    Ham çizgi yerine, her bar için o ana kadar bilinen (yalnızca geçmişe
    dayalı) seviyelerden en yakın destek/dirence olan mesafe (%) ve aktif
    seviye sayısı döner — bunlar ML özelliği olarak doğrudan kullanılabilir.
    """
    high, low, close = ohlcv["high"].to_numpy(), ohlcv["low"].to_numpy(), ohlcv["close"].to_numpy()
    n = len(close)
    rb = pivot_lookback

    pivot_price: list[float] = []  # her pivot'un fiyatı, keşfedildiği (confirm) sırayla
    dist_support = np.full(n, np.nan)
    dist_resistance = np.full(n, np.nan)
    level_count = np.zeros(n, dtype=int)

    def _is_pivot_high(idx: int) -> bool:
        if idx - rb < 0 or idx + rb >= n:
            return False
        window = high[idx - rb : idx + rb + 1]
        return high[idx] == window.max()

    def _is_pivot_low(idx: int) -> bool:
        if idx - rb < 0 or idx + rb >= n:
            return False
        window = low[idx - rb : idx + rb + 1]
        return low[idx] == window.min()

    for i in range(n):
        confirm_idx = i - rb
        if confirm_idx >= rb:
            if _is_pivot_high(confirm_idx):
                pivot_price.append(high[confirm_idx])
            if _is_pivot_low(confirm_idx):
                pivot_price.append(low[confirm_idx])

        recent_pivots = [p for p in pivot_price[-200:]]  # performans için sınırla
        if len(recent_pivots) >= min_pivots:
            recent_window = recent_pivots[-int(channel_lookback / max(rb, 1)) :] if channel_lookback else recent_pivots
            price_range = max(recent_window) - min(recent_window) if len(recent_window) > 1 else 0.0
            cwidth = price_range * channel_width_pct / 100 if price_range > 0 else close[i] * 0.001

            sorted_pivots = sorted(recent_window)
            levels: list[float] = []
            used = [False] * len(sorted_pivots)
            for a in range(len(sorted_pivots)):
                if used[a]:
                    continue
                cluster = [sorted_pivots[a]]
                used[a] = True
                for b in range(a + 1, len(sorted_pivots)):
                    if used[b]:
                        continue
                    if sorted_pivots[b] - cluster[0] <= cwidth:
                        cluster.append(sorted_pivots[b])
                        used[b] = True
                if len(cluster) >= min_pivots:
                    levels.append(float(np.mean(cluster)))

            level_count[i] = len(levels)
            if levels:
                below = [lv for lv in levels if lv <= close[i]]
                above = [lv for lv in levels if lv >= close[i]]
                if below:
                    dist_support[i] = (close[i] - max(below)) / close[i] * 100
                if above:
                    dist_resistance[i] = (min(above) - close[i]) / close[i] * 100

    return pd.DataFrame(
        {
            "sr_dist_support_pct": dist_support,
            "sr_dist_resistance_pct": dist_resistance,
            "sr_level_count": level_count,
        },
        index=ohlcv.index,
    )


def rolling_hurst_exponent(close: pd.Series, window: int = 100) -> pd.Series:
    """Kayan pencereli Hurst üsteli (H) — bir fiyat serisinin "hafızası"nı
    (trend-devamlılığı mı, ortalamaya-dönüş mü, yoksa rastgele yürüyüş mü
    olduğunu) ölçen bir fraktal analiz göstergesi:

    - H ≈ 0.5: rastgele yürüyüş (öngörülemez, "verimli piyasa")
    - H > 0.5: trend-devamlılığı (momentum kalıcı olma eğiliminde)
    - H < 0.5: ortalamaya-dönüş (aşırı hareketler tersine dönme eğiliminde)

    Basitleştirilmiş varyans-ölçekleme yöntemiyle (fiyat farklarının
    farklı gecikmelerdeki (lag) standart sapmasının log-log eğimi, ≈2H)
    tahmin edilir — tam R/S analizi değildir ama pratikte yaygın kullanılan,
    ucuz bir yaklaşımdır.

    Not: Fraktal boyut (D) klasik olarak D = 2 - H ilişkisiyle Hurst'ten
    DOĞRUDAN türetilir — yani ayrı bir "fraktal boyut" özelliği eklemek
    bu Hurst değerinin birebir (ters yönlü) bir kopyası olur, modele YENİ
    bilgi katmaz. Bu yüzden burada yalnızca Hurst hesaplanıyor; fraktal
    boyut isteyen bir tüketici `2 - hurst` ile kendi türetebilir.
    """
    log_close = np.log(close.clip(lower=1e-9))
    lags = np.arange(2, 20)

    def _hurst(x: np.ndarray) -> float:
        taus = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
        taus = np.where(taus <= 0, 1e-8, taus)
        slope = np.polyfit(np.log(lags), np.log(taus), 1)[0]
        return float(np.clip(slope * 2.0, 0.0, 1.0))

    return log_close.rolling(window).apply(_hurst, raw=True)
