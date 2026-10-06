"""sentiment_fng: günlük Fear & Greed endeksi (alternative.me, ücretsiz, 2018-02'den).

Değer, ilgili günün 00:00 UTC damgasıyla yayımlanır; yayın gecikmesi payı olarak `available_at = tarih 00:00 + 6 saat`
(yani d gününün kapanış kararı d değerini kullanabilir, d günü içindeki bir karar kullanamaz)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from .. import config
from ..guard import cut_final_test
from .base import MARKET, Source
from .registry import register_source

URL = "https://api.alternative.me/fng/?limit=0&format=json"
PUBLICATION_MARGIN = pd.Timedelta(hours=6)


def _download() -> list[dict]:
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    return r.json()["data"]


@register_source
class SentimentFng(Source):
    name = "sentiment_fng"
    version = "1"
    scope = "market"
    publication_lag = PUBLICATION_MARGIN
    max_staleness = pd.Timedelta(days=3)

    def __init__(self, fetcher: Callable[[], list[dict]] | None = None, max_age_hours: float = 12.0, **params):
        super().__init__(max_age_hours=max_age_hours, **params)
        self._fetcher = fetcher

    def data_signature(self) -> str:
        if self._fetcher is not None:
            return ""
        now = pd.Timestamp.now(tz="UTC")
        return f"{now:%Y-%m-%d}-{now.hour // 12}"  # canlı API: 12 saatlik dilim (dosya önbelleğiyle aynı ömür)

    @property
    def history_start(self):
        return pd.Timestamp("2018-02-01")

    def _cache_file(self) -> Path:
        return Path(config.CACHE_DIR) / "raw" / "fng.json"

    def fetch(self, start, end, symbols=None):
        if self._fetcher is not None:
            return self._fetcher()
        path = self._cache_file()
        if path.exists() and (time.time() - path.stat().st_mtime) < self.params["max_age_hours"] * 3600:
            return json.loads(path.read_text(encoding="utf-8"))
        data = _download()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def to_panel(self, universe, dates):
        df = pd.DataFrame(self.fetch(dates.min(), dates.max()))
        df["date"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="s", utc=True).dt.tz_convert(None).dt.floor("D")
        df["value"] = pd.to_numeric(df["value"]) / 100.0
        df = cut_final_test(df.set_index("date")).reset_index()  # kilitli pencereler (etkin bölge) KESİLİR
        df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        panel = pd.DataFrame(
            {"date": df["date"], "symbol": MARKET, "fng": df["value"], "fng_chg_7": df["value"].diff(7),
             "available_at": df["date"] + PUBLICATION_MARGIN}
        )
        panel = panel[(panel["date"] >= dates.min() - pd.Timedelta(days=30)) & (panel["date"] <= dates.max())]
        return self.prefix(panel).reset_index(drop=True)
