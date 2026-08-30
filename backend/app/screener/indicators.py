import pandas as pd


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame'ine tarama için gereken teknik göstergeleri ekler."""
    out = df.copy()
    out["ema_fast"] = _ema(out["close"], 20)
    out["ema_slow"] = _ema(out["close"], 50)
    out["rsi"] = _rsi(out["close"], 14)

    macd = _macd(out["close"])
    out["macd"] = macd["macd"]
    out["macd_signal"] = macd["macd_signal"]
    out["macd_hist"] = macd["macd_hist"]

    out["volume_sma"] = out["volume"].rolling(20).mean()
    out["momentum"] = out["close"].pct_change(10) * 100
    return out


def composite_score(indicators: pd.DataFrame) -> float:
    """Son mumdaki göstergelerden -100..+100 arası bir yön skoru üretir.

    Pozitif skor Long, negatif skor Short lehine sinyal demektir.
    Alt bileşenler eşit ağırlıklı: trend (EMA çaprazı), momentum (RSI'nin
    50'den sapması), MACD histogramı yönü ve fiyat momentumu.
    """
    last = indicators.iloc[-1]

    trend = 1.0 if last["ema_fast"] > last["ema_slow"] else -1.0
    trend_strength = abs(last["ema_fast"] - last["ema_slow"]) / last["close"] * 100
    trend_score = trend * min(trend_strength * 10, 100)

    rsi_score = (last["rsi"] - 50) * 2  # RSI 0..100 -> -100..100

    macd_score = 100.0 if last["macd_hist"] > 0 else -100.0
    macd_score *= min(abs(last["macd_hist"]) / last["close"] * 1000, 1.0)

    volume_ratio = last["volume"] / last["volume_sma"] if last["volume_sma"] else 1.0
    momentum_score = max(min(last["momentum"], 100), -100) * min(volume_ratio, 2.0) / 2.0

    return round((trend_score + rsi_score + macd_score + momentum_score) / 4, 2)
