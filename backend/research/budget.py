"""Deneme bütçesi (KURALLAR.md §9): soru (`question`) başına en fazla `MAX_VARIANTS_PER_QUESTION` varyant
(deneme sayacına giren her varyant); smoke koşuları ayrı,
soru başına `MAX_SMOKE_PER_QUESTION` ile sınırlıdır. Mantık `research.registry`'dedir (bütçe `register_experiment` içinde KİLİT
ALTINDA da zorlanır); bu modül uyumluluk/okunabilirlik için yeniden dışa aktarır."""

from .registry import (  # noqa: F401
    MAX_VARIANTS_PER_QUESTION,
    MAX_SMOKE_PER_QUESTION,
    BudgetExceededError,
    assert_budget,
    variants_used,
    normalize_question,
    record_smoke_run,
)
