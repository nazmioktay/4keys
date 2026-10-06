"""taker_flow: klines'taki taker alış hacmi (taker_buy_quote / quote_volume) ve 7 günlük ortalaması."""

from __future__ import annotations

import pandas as pd

from ..data import store
from .base import Source
from .ohlcv_core import stack_features
from .registry import register_source


@register_source
class TakerFlow(Source):
    name = "taker_flow"
    version = "1"

    def fetch(self, start, end, symbols=None):
        symbols = symbols if symbols is not None else store.list_symbols("um_1d")
        out = {f: store.load_panel(f, symbols, "um_1d") for f in ("taker_buy_quote", "quote_volume")}
        return {k: v[(v.index >= start) & (v.index <= end)] if len(v) else v for k, v in out.items()}

    def to_panel(self, universe, dates):
        raw = self.fetch(dates.min() - pd.Timedelta(days=10), dates.max(), universe)
        if not len(raw["quote_volume"]):
            return pd.DataFrame(columns=["date", "symbol", "available_at"])
        ratio = (raw["taker_buy_quote"] / raw["quote_volume"].where(raw["quote_volume"] > 0)).clip(0, 1)
        panel = stack_features({"ratio": ratio, "ratio_7d": ratio.rolling(7, min_periods=4).mean()})
        panel = panel[panel["date"].isin(dates)].dropna(subset=["ratio", "ratio_7d"], how="all")
        panel["available_at"] = panel["date"] + pd.Timedelta(days=1)
        return self.prefix(panel).reset_index(drop=True)
