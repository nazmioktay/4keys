"""ohlcv_core: çok ufuklu getiri, volatilite (kapanış-kapanış + Parkinson), gün içi aralık, hacim özellikleri.
Kaynak: `um_1d` parquet (nihai pencere varsayılan KESİLİR). `available_at = open_time + 1 gün` (bar kapanışı)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import store
from .base import Source
from .registry import register_source

HORIZONS = (1, 3, 7, 14, 30, 60, 90)


def stack_features(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{özellik: tarih×sembol} -> uzun biçim (date, symbol, özellikler)."""
    parts = [df.stack(future_stack=True).rename(name) for name, df in features.items()]
    long = pd.concat(parts, axis=1)
    long.index = long.index.set_names(["date", "symbol"])
    return long.reset_index()


@register_source
class OhlcvCore(Source):
    name = "ohlcv_core"
    version = "1"
    max_staleness = pd.Timedelta(days=3)

    @property
    def history_start(self):
        syms = store.list_symbols("um_1d")
        firsts = [store.read_frame("um_1d", s).index.min() for s in syms[:5]] if syms else []
        return min(firsts) if firsts else None

    def fetch(self, start, end, symbols=None):
        symbols = symbols if symbols is not None else store.list_symbols("um_1d")
        panels = {f: store.load_panel(f, symbols, "um_1d") for f in ("open", "high", "low", "close", "quote_volume")}
        return {k: v[(v.index >= start) & (v.index <= end)] if len(v) else v for k, v in panels.items()}

    def to_panel(self, universe, dates):
        warm = pd.Timedelta(days=130)  # en uzun pencere (90g getiri + 30g) için ısınma; satırlar sonra `dates`'e kesilir
        raw = self.fetch(dates.min() - warm, dates.max(), universe)
        if not len(raw["close"]):
            return pd.DataFrame(columns=["date", "symbol", "available_at"])
        open_, high, low, close, qv = (raw[k] for k in ("open", "high", "low", "close", "quote_volume"))
        log_close = np.log(close.where(close > 0))
        ret1 = log_close.diff()
        feats: dict[str, pd.DataFrame] = {}
        for h in HORIZONS:
            feats[f"ret_{h}"] = log_close.diff(h)
        pk_var = (np.log(high.where(high > 0) / low.where(low > 0)) ** 2) / (4 * np.log(2))
        feats["rv_pk_1"] = np.sqrt(pk_var)
        feats["rv_pk_5"] = np.sqrt(pk_var.rolling(5, min_periods=5).mean())
        feats["rv_pk_22"] = np.sqrt(pk_var.rolling(22, min_periods=22).mean())
        for w in (7, 30):
            feats[f"rv_cc_{w}"] = np.sqrt((ret1**2).rolling(w, min_periods=w).mean())
        feats["range_1"] = (high - low) / open_.where(open_ > 0)
        ldv = np.log(qv.where(qv > 0))
        feats["log_dollar_vol"] = ldv
        feats["vol_z_30"] = (ldv - ldv.rolling(30, min_periods=30).mean()) / ldv.rolling(30, min_periods=30).std()
        panel = stack_features(feats)
        panel = panel[panel["date"].isin(dates)].dropna(subset=list(feats), how="all")
        panel["available_at"] = panel["date"] + pd.Timedelta(days=1)  # bar kapanışı
        return self.prefix(panel).reset_index(drop=True)
