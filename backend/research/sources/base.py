"""Veri kaynağı eklentisi arayüzü.

Karar zamanı kuralı (KURALLAR.md §4): panelde `date` = gözlem günü; o günün kapanışında (date + 1 gün 00:00 UTC)
karar verilir. Bir özellik, `available_at <= karar zamanı` ise kullanılabilir. Her satır `available_at` taşır.
Özellik kolonları `<kaynak_adı>__<özellik>` biçiminde önek alır (çakışma olmaz).

Yeni kaynak eklemek: bu sınıftan türet, `@register_source` ile kaydet, `sources/registry.py::_MODULES`'e modülü ekle
(bkz. docs/research/EKLENTI_REHBERI.md ve sablonlar/kaynak_sablonu.py)."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd

MARKET = "__MARKET__"  # piyasa-geneli kaynakların sembol değeri
RESERVED_COLUMNS = ("date", "symbol", "available_at")


class Source(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "1"
    scope: ClassVar[str] = "symbol"  # "symbol" | "market"
    forward_only: ClassVar[bool] = False  # geçmişi yok, yalnızca bugünden itibaren toplanabilir
    publication_lag: ClassVar[pd.Timedelta] = pd.Timedelta(0)  # verinin gerçekte ne kadar gecikmeyle yayımlandığı
    max_staleness: ClassVar[pd.Timedelta] = pd.Timedelta(days=3)  # birleştirmede geriye bakma toleransı

    def __init__(self, **params: Any) -> None:
        self.params = params

    @property
    def history_start(self) -> pd.Timestamp | None:
        """Kaynağın en erken tarihi (bilinmiyorsa None)."""
        return None

    def accumulated_days(self) -> float | None:
        """forward_only kaynaklar için şimdiye dek BİRİKEN veri süresi (gün); koşucu 365 günden azını reddeder."""
        return None

    @abstractmethod
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp, symbols: list[str] | None = None) -> Any:
        """Ham veriyi döner (ağ/dosya/DB). `symbols` sembol-bazlı kaynaklar için istenen evren."""

    @abstractmethod
    def to_panel(self, universe: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
        """[date, symbol, <özellikler>, available_at] — özellikler `<name>__` önekli. `dates` istenen karar günleri
        (kaynak daha geniş aralık döndürebilir; birleştirme `available_at`'e göre yapılır)."""

    # ---- yardımcılar -------------------------------------------------------------
    def feature_columns(self, panel: pd.DataFrame) -> list[str]:
        return [c for c in panel.columns if c not in RESERVED_COLUMNS]

    def prefix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Özellik kolonlarına `<name>__` öneki ekler (rezerve kolonlar hariç)."""
        return frame.rename(columns={c: f"{self.name}__{c}" for c in frame.columns if c not in RESERVED_COLUMNS})

    def param_hash(self) -> str:
        blob = json.dumps({"name": self.name, "version": self.version, "params": self.params}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def describe(self) -> dict:
        return {
            "name": self.name, "version": self.version, "scope": self.scope, "forward_only": self.forward_only,
            "publication_lag": str(self.publication_lag), "max_staleness": str(self.max_staleness),
            "history_start": None if self.history_start is None else str(self.history_start), "params": self.params,
        }


def validate_panel(source: Source, panel: pd.DataFrame) -> None:
    """Arayüz sözleşmesi: zorunlu kolonlar, önek, available_at tipi, piyasa-geneli sembol."""
    for col in RESERVED_COLUMNS:
        if col not in panel.columns:
            raise ValueError(f"{source.name}: panelde '{col}' kolonu yok")
    if not pd.api.types.is_datetime64_any_dtype(panel["available_at"]):
        raise ValueError(f"{source.name}: available_at datetime olmalı")
    feats = [c for c in panel.columns if c not in RESERVED_COLUMNS]
    bad = [c for c in feats if not c.startswith(f"{source.name}__")]
    if bad:
        raise ValueError(f"{source.name}: özellik kolonları '{source.name}__' önekli olmalı: {bad[:3]}")
    if source.scope == "market" and not (panel["symbol"] == MARKET).all():
        raise ValueError(f"{source.name}: piyasa-geneli kaynak symbol='{MARKET}' döndürmeli")
