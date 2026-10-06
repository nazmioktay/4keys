"""macro: günlük makro/piyasa bağlamı (VIX, altın, S&P 500, Nasdaq, Nikkei, DAX) — yfinance günlük kapanışları.

`backend/app/macro` yalnızca SON değeri döndürür (geçmiş yok); geçmiş için aynı ticker'lar yfinance'ten günlük çekilir.
YAYIN GECİKMESİ: her serinin günlük barı, piyasa kapanışından SONRA bilinir. `SERIES`'teki kapanış saatleri bilinçli
muhafazakâr (yaz/kış saati içinde en geç kapanış) + `PUBLICATION_LAG` 1 saat: örn. ABD serileri d gününün 22:00-23:00 UTC'sinde,
Nikkei 07:30'da, DAX 18:00'de kullanılabilir. Satır `available_at`'ı = o satırdaki serilerin EN GEÇ olanı.
Eksik gün (hafta sonu/tatil) NaN kalır, 0'la doldurulmaz. Seviye özellikleri GENİŞLEYEN z-skordur (yalnızca geçmişle,
`app.ml.macro_features._expanding_zscore`) — önceki sürümdeki ileri bilgi hatası tekrarlanmaz."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .base import MARKET, Source
from .registry import register_source

# seri -> (yfinance ticker, muhafazakâr kapanış saati UTC)
SERIES = {
    "vix": ("^VIX", 22.0),
    "gold": ("GC=F", 22.0),
    "sp500": ("^GSPC", 21.0),
    "nasdaq": ("^IXIC", 21.0),
    "nikkei": ("^N225", 6.5),
    "dax": ("^GDAXI", 17.0),
}
PUBLICATION_LAG = pd.Timedelta(hours=1)


def _yf_fetch(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.Series]:
    import yfinance as yf

    out = {}
    for name, (ticker, _) in SERIES.items():
        hist = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"), end=(end + pd.Timedelta(days=2)).strftime("%Y-%m-%d"), interval="1d"
        )
        if hist.empty:
            continue
        s = hist["Close"].copy()
        idx = pd.DatetimeIndex(s.index)
        s.index = (idx.tz_localize(None) if idx.tz is None else idx.tz_convert(None)).normalize()
        out[name] = s.dropna()
    return out


@register_source
class Macro(Source):
    name = "macro"
    version = "1"
    scope = "market"
    publication_lag = PUBLICATION_LAG
    max_staleness = pd.Timedelta(days=5)

    def __init__(self, fetcher: Callable[[pd.Timestamp, pd.Timestamp], dict[str, pd.Series]] | None = None, **params):
        super().__init__(**params)
        self._fetcher = fetcher

    def data_signature(self) -> str:
        return "" if self._fetcher is not None else f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}"  # canlı yfinance: günlük

    @property
    def history_start(self):
        return pd.Timestamp("2000-01-01")

    def fetch(self, start, end, symbols=None):
        return (self._fetcher or _yf_fetch)(start, end)

    def to_panel(self, universe, dates):
        from app.ml.macro_features import (
            _expanding_zscore,  # yeniden kullanım: yalnızca geçmişle normalizasyon
        )

        raw = self.fetch(dates.min() - pd.Timedelta(days=60), dates.max())
        if not raw:
            return pd.DataFrame(columns=["date", "symbol", "available_at"])
        idx = sorted(set().union(*[set(s.index) for s in raw.values()]))
        wide = pd.DataFrame({k: s.reindex(idx) for k, s in raw.items()})
        feats = {}
        avail = pd.DataFrame(index=wide.index)
        for name in wide.columns:
            level = wide[name]
            logl = np.log(level.where(level > 0))
            feats[f"{name}_z"] = _expanding_zscore(logl)
            feats[f"{name}_chg_1"] = logl.diff()
            feats[f"{name}_chg_5"] = logl.diff(5)
            avail[name] = (wide.index + pd.Timedelta(hours=SERIES[name][1]) + PUBLICATION_LAG).where(level.notna())
        panel = pd.DataFrame(feats)
        panel["available_at"] = avail.max(axis=1)  # satırdaki en geç yayımlanan seri
        panel = panel.dropna(subset=["available_at"])
        panel.insert(0, "symbol", MARKET)
        panel = panel.reset_index(names="date")
        panel = panel[(panel["date"] >= dates.min() - pd.Timedelta(days=30)) & (panel["date"] <= dates.max())]
        return self.prefix(panel).reset_index(drop=True)
