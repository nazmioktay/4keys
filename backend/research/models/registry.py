"""Model kayıt defteri. Yeni model = tek dosya (`@register_model("ad")`) + aşağıdaki `_MODULES` listesine modül adı
(bkz. docs/research/EKLENTI_REHBERI.md ve sablonlar/model_sablonu.py)."""

from __future__ import annotations

import importlib

from .base import BaseModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {}
_MODULES = ("linear", "lgbm", "xgb", "har_garch", "seq")
_loaded = False


def register_model(name: str):
    def deco(cls: type[BaseModel]) -> type[BaseModel]:
        if name in MODEL_REGISTRY and MODEL_REGISTRY[name] is not cls:
            raise ValueError(f"'{name}' adlı model zaten kayıtlı")
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls

    return deco


def _load_all() -> None:
    global _loaded
    if _loaded:
        return
    for mod in _MODULES:
        try:
            importlib.import_module(f"research.models.{mod}")
        except ModuleNotFoundError as exc:
            if exc.name != f"research.models.{mod}":
                raise  # eksik bağımlılık (ör. lightgbm) — gizleme
    _loaded = True


def get_model(name: str, seed: int = 0, **params) -> BaseModel:
    _load_all()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Bilinmeyen model '{name}'. Kayıtlı modeller: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](seed=seed, **params)


def available_models() -> list[str]:
    _load_all()
    return sorted(MODEL_REGISTRY)
