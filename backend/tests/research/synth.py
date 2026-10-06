"""Sentetik piyasa: sembol başına AR(1) log-vol (volatilite kümelenmesi -> HAR becerisi), aralık vol ile orantılı."""

import numpy as np
import pandas as pd

from research.data import store


def make_market(n_symbols=14, n_days=1000, start="2021-01-01", seed=0, with_btc=True):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days, freq="D", name="open_time")
    symbols = (["BTCUSDT"] if with_btc else []) + [f"S{i:02d}USDT" for i in range(n_symbols - int(with_btc))]
    frames = {}
    for k, sym in enumerate(symbols):
        mu = np.log(0.025) + rng.normal(0, 0.2)
        phi = 0.95
        lv = np.empty(n_days)
        lv[0] = mu
        for t in range(1, n_days):
            lv[t] = mu + phi * (lv[t - 1] - mu) + rng.normal(0, 0.12)
        sigma = np.exp(lv)
        eps = rng.normal(size=n_days)
        close = 100 * np.exp(np.cumsum(sigma * eps))
        open_ = np.r_[100.0, close[:-1]]
        z1, z2 = np.abs(rng.normal(size=n_days)), np.abs(rng.normal(size=n_days))
        high = np.maximum(open_, close) * np.exp(0.8 * sigma * z1)
        low = np.minimum(open_, close) * np.exp(-0.8 * sigma * z2)
        scale = 1e9 / (k + 1)  # sembol sırası = hacim sırası (BTC en büyük)
        qv = scale * rng.uniform(0.8, 1.2, n_days)
        frames[sym] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": qv / close, "quote_volume": qv,
             "trades": 100.0, "taker_buy_base": qv / close * 0.5, "taker_buy_quote": qv * 0.5}, index=idx)
    return frames


def write_market(frames):
    for sym, df in frames.items():
        store.write_frame("um_1d", sym, df)
