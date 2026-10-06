"""funding: günlük funding özeti. Gün d = (d 00:00, d+1 00:00] aralığındaki ödemeler (motorla aynı sınır);
`available_at` = o aralıktaki SON ödeme zamanı (karar zamanını aşmaz)."""

from __future__ import annotations

import pandas as pd

from ..data import store
from .base import Source
from .registry import register_source


@register_source
class Funding(Source):
    name = "funding"
    version = "1"
    max_staleness = pd.Timedelta(days=2)

    def fetch(self, start, end, symbols=None):
        symbols = symbols if symbols is not None else store.list_symbols("um_funding")
        out = {}
        for s in symbols:
            df = store.load_funding(s)
            if df is not None and len(df):
                out[s] = df[(df.index > start) & (df.index <= end + pd.Timedelta(days=1))]
        return out

    def to_panel(self, universe, dates):
        raw = self.fetch(dates.min() - pd.Timedelta(days=10), dates.max(), universe)
        rows = []
        for symbol, df in raw.items():
            day = (df.index - pd.Timedelta(1, "ns")).floor("D")
            g = df["rate"].groupby(day)
            daily = pd.DataFrame({"sum": g.sum(), "mean": g.mean(), "last": g.last(), "count": g.count()})
            daily["avail"] = pd.Series(df.index, index=df.index).groupby(day).max()
            daily["sum_7d"] = daily["sum"].rolling(7, min_periods=3).mean()
            daily = daily.reset_index(names="date")
            daily["symbol"] = symbol
            rows.append(daily)
        if not rows:
            return pd.DataFrame(columns=["date", "symbol", "available_at"])
        panel = pd.concat(rows, ignore_index=True)
        panel = panel[panel["date"].isin(dates)].rename(columns={"avail": "available_at"})
        panel = panel[["date", "symbol", "sum", "mean", "last", "count", "sum_7d", "available_at"]]
        return self.prefix(panel).reset_index(drop=True)
