"""Hedef (etiket) kütüphanesi.

Zaman hizası motorla AYNI: karar t günü kapanışında (= t+1 00:00 UTC) verilir; getiri `open[t+1] -> open[t+1+h]`,
volatilite/aralık t+1 .. t+h günlerinin barlarından. Bu yüzden etiket `date + (h + 1)` günü 00:00'da TAMAMLANIR
(`Target.label_end_days = h + 1`). Her hedef `horizon` (= h, gün) bildirir; koşucu purge = h, embargo >= h uygular.

Son h satır etiketsizdir (NaN) — doldurulmaz. `PriceData`: tarih × sembol geniş paneller (open, high, low, close)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import config

PriceData = dict[str, pd.DataFrame]  # {"open","high","low","close"} -> tarih × sembol


@dataclass(frozen=True)
class Target:
    name: str
    horizon: int
    kind: str  # "regression" | "classification" | "rank"
    params: dict = field(default_factory=dict)
    _fn: Callable[..., pd.DataFrame] = field(default=None, repr=False, compare=False)

    @property
    def label_end_days(self) -> int:
        """Etiketin tamamlandığı an: date + label_end_days gün (00:00 UTC)."""
        return self.horizon + 1

    def compute(self, prices: PriceData, membership: pd.DataFrame | None = None) -> pd.DataFrame:
        """tarih × sembol etiket paneli. `membership` (tarih × sembol bool): yalnızca evren üyeleri için etiket
        (kesitsel hedeflerde sıra evren içinde hesaplanır)."""
        y = self._fn(prices, membership)
        if membership is not None:
            y = y.where(membership.reindex(index=y.index, columns=y.columns).fillna(False))
        return y

    def describe(self) -> dict:
        return {"name": self.name, "horizon": self.horizon, "kind": self.kind, "params": self.params}


# ---------------------------------------------------------------- ortak yardımcılar
def forward_open_return(open_: pd.DataFrame, h: int) -> pd.DataFrame:
    """log(open[t+1+h] / open[t+1]) — t kapanışında karar, t+1 açılışında dolum (motorla aynı)."""
    lo = np.log(open_.where(open_ > 0))
    return lo.shift(-(1 + h)) - lo.shift(-1)


def parkinson_variance(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    return (np.log(high.where(high > 0) / low.where(low > 0)) ** 2) / (4 * np.log(2))


def trailing_vol(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """t'ye KADAR bilinen günlük log getiri std'si."""
    r = np.log(close.where(close > 0)).diff()
    return r.rolling(window, min_periods=window).std()


# ---------------------------------------------------------------- hedefler
def next_rv(h: int = 1, annualize: bool = False) -> Target:
    """İleri h gün gerçekleşen volatilite (günlük, Parkinson aralık tahmincisi): sqrt(mean_{k=1..h} PK_var[t+k]).
    Günlük veride 'gerçekleşen vol' için aralık (high/low) tahmincisi, kapanış-kapanış r²'den çok daha az gürültülüdür.
    `annualize=True`: × sqrt(365)."""

    def fn(prices: PriceData, membership):
        var = parkinson_variance(prices["high"], prices["low"])
        fwd = var.rolling(h, min_periods=h).mean().shift(-h)
        out = np.sqrt(fwd)
        return out * np.sqrt(config.ANNUALIZATION_DAYS) if annualize else out

    return Target("next_rv", h, "regression", {"h": h, "annualize": annualize}, fn)


def vol_adj_return(h: int = 1, vol_window: int = 20) -> Target:
    """İleri h gün getiri / (t'ye kadar bilinen günlük vol × sqrt(h))."""

    def fn(prices: PriceData, membership):
        fwd = forward_open_return(prices["open"], h)
        vol = trailing_vol(prices["close"], vol_window)
        return fwd / (vol * np.sqrt(h)).where(vol > 0)

    return Target("vol_adj_return", h, "regression", {"h": h, "vol_window": vol_window}, fn)


def cross_sectional_rank(h: int = 1) -> Target:
    """Her tarihte evren içinde ileri h gün getirisinin yüzdelik sırası (0..1). Evren yalnızca `membership`'ten gelir."""

    def fn(prices: PriceData, membership):
        fwd = forward_open_return(prices["open"], h)
        if membership is not None:
            fwd = fwd.where(membership.reindex(index=fwd.index, columns=fwd.columns).fillna(False))
        return fwd.rank(axis=1, pct=True)

    return Target("cross_sectional_rank", h, "rank", {"h": h}, fn)


# ---- kural stratejileri (strategy_outcome için): pozisyon ∈ {-1, 0, +1}, t kapanışına kadar bilinen veriyle ----
def _ema_trend(prices: PriceData, span: int = 50) -> pd.DataFrame:
    close = prices["close"]
    ema = close.ewm(span=span, adjust=False).mean()
    pos = (close > ema).astype(float)
    pos.iloc[: span - 1] = 0.0
    return pos


def _momentum_sign(prices: PriceData, lookback: int = 30) -> pd.DataFrame:
    lc = np.log(prices["close"].where(prices["close"] > 0))
    return np.sign(lc.diff(lookback)).fillna(0.0)


STRATEGIES: dict[str, Callable[..., pd.DataFrame]] = {"ema_trend": _ema_trend, "momentum_sign": _momentum_sign}


def strategy_outcome(strategy: str, h: int = 5, binary: bool = False, cost_bps: float | None = None, **strategy_params) -> Target:
    """Kural stratejisinin ileri h gün NET sonucu: pozisyon_t × ileri getiri − (pozisyon varsa) gidiş-dönüş maliyet.
    `binary=True`: sonuç > 0 mı (sınıflandırma). Maliyet varsayılanı: 2 × (taker ücreti + BTC/ETH kayması)."""
    if strategy not in STRATEGIES:
        raise KeyError(f"Bilinmeyen strateji '{strategy}'. Kayıtlı: {sorted(STRATEGIES)}")
    rt_cost = (cost_bps / 1e4) if cost_bps is not None else 2 * (config.FUTURES_TAKER_FEE + config.SLIPPAGE_BPS_BTC_ETH / 1e4)

    def fn(prices: PriceData, membership):
        pos = STRATEGIES[strategy](prices, **strategy_params)
        fwd = forward_open_return(prices["open"], h)
        net = pos * fwd - rt_cost * (pos != 0)
        net = net.where(fwd.notna())
        return (net > 0).astype(float).where(net.notna()) if binary else net

    kind = "classification" if binary else "regression"
    return Target("strategy_outcome", h, kind, {"strategy": strategy, "h": h, "binary": binary, "rt_cost": rt_cost, **strategy_params}, fn)


TARGET_FACTORIES: dict[str, Callable[..., Target]] = {
    "next_rv": next_rv,
    "vol_adj_return": vol_adj_return,
    "cross_sectional_rank": cross_sectional_rank,
    "strategy_outcome": strategy_outcome,
}


def get_target(name: str, **params) -> Target:
    if name not in TARGET_FACTORIES:
        raise KeyError(f"Bilinmeyen hedef '{name}'. Kayıtlı: {sorted(TARGET_FACTORIES)}")
    return TARGET_FACTORIES[name](**params)
