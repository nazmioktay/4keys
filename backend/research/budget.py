"""Deneme bütçesi (KURALLAR.md §9): soru (`question`) başına en fazla `MAX_CONFIGS_PER_QUESTION` config.

Sayım `_registry.json`'daki kayıtlardan (her kayıt bir config; ablasyonlar dahil) yapılır. Bütçe dolunca YENİ deneme
yapılmaz — mevcut sonuçlardan karar verilir. `smoke` deneyler sayılmaz."""

from __future__ import annotations

from pathlib import Path

from . import config
from .registry import _read_registry

MAX_CONFIGS_PER_QUESTION = 40


class BudgetExceededError(RuntimeError):
    pass


def configs_used(question: str, registry_file: Path | None = None) -> int:
    reg = _read_registry(Path(registry_file or config.REGISTRY_FILE))
    return sum(1 for e in reg["experiments"] if e.get("question") == question)


def assert_budget(question: str, n_new: int = 1, registry_file: Path | None = None, limit: int = MAX_CONFIGS_PER_QUESTION) -> int:
    """Yeni `n_new` config sığıyor mu? Sığmıyorsa `BudgetExceededError`. Döner: kalan bütçe (n_new sonrası)."""
    used = configs_used(question, registry_file)
    if used + n_new > limit:
        raise BudgetExceededError(
            f"'{question}' sorusu için deneme bütçesi dolu ({used}/{limit} config; {n_new} yeni istendi). "
            "Yeni deneme yapılmaz: mevcut sonuçlardan karar verin (docs/research/deneyler.md)."
        )
    return limit - used - n_new
