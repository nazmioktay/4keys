"""Deneme bütçesi (KURALLAR.md §9): soru (`question`) başına en fazla `MAX_CONFIGS_PER_QUESTION` config; smoke koşuları ayrı,
soru başına `MAX_SMOKE_PER_QUESTION` ile sınırlıdır. Mantık `research.registry`'dedir (bütçe `register_experiment` içinde KİLİT
ALTINDA da zorlanır); bu modül uyumluluk/okunabilirlik için yeniden dışa aktarır."""

from .registry import (  # noqa: F401
    MAX_CONFIGS_PER_QUESTION,
    MAX_SMOKE_PER_QUESTION,
    BudgetExceededError,
    assert_budget,
    configs_used,
    normalize_question,
    record_smoke_run,
)
