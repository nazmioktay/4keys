"""Kaynak kayıt defteri: config'te kaynaklar ADLA seçilir (`sources: [ohlcv_core, funding, ...]`)."""

from __future__ import annotations

import importlib

from .base import Source

SOURCE_REGISTRY: dict[str, type[Source]] = {}

# Yeni kaynak = tek dosya + bu listeye modül adı (bkz. EKLENTI_REHBERI.md)
_MODULES = ("ohlcv_core", "funding", "taker_flow", "derivatives_metrics", "macro", "sentiment_fng", "forward")
_loaded = False


def register_source(cls: type[Source]) -> type[Source]:
    if not getattr(cls, "name", None):
        raise ValueError("Kaynak sınıfında 'name' zorunlu")
    if cls.name in SOURCE_REGISTRY and SOURCE_REGISTRY[cls.name] is not cls:
        raise ValueError(f"'{cls.name}' adlı kaynak zaten kayıtlı")
    SOURCE_REGISTRY[cls.name] = cls
    return cls


def _load_all() -> None:
    global _loaded
    if _loaded:
        return
    for mod in _MODULES:
        try:
            importlib.import_module(f"research.sources.{mod}")
        except ModuleNotFoundError as exc:  # henüz yazılmamış isteğe bağlı modül
            if exc.name != f"research.sources.{mod}":
                raise
    _loaded = True


def get_source(name: str, **params) -> Source:
    _load_all()
    if name not in SOURCE_REGISTRY:
        raise KeyError(f"Bilinmeyen kaynak '{name}'. Kayıtlı kaynaklar: {sorted(SOURCE_REGISTRY)}")
    return SOURCE_REGISTRY[name](**params)


def available_sources() -> list[str]:
    _load_all()
    return sorted(SOURCE_REGISTRY)
