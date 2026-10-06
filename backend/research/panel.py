"""Deney paneli: noktasal-zamanlı evren + kaynak özellikleri + hedef -> (date, symbol) indeksli X, y.

Nihai pencere (>= 2025-10-01) yükleyicilerde kesilir; ek olarak `period.end` ihlali burada reddedilir."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .data import store
from .data.universe import universe_membership
from .guard import assert_no_final_test
from .sources import cache as source_cache
from .sources.base import Source, validate_panel
from .sources.pit import decision_frame, feature_matrix_columns, merge_sources
from .sources.registry import get_source
from .targets import Target, get_target

OHLC = ("open", "high", "low", "close")


class ForwardOnlyTooShortError(RuntimeError):
    pass


@dataclass
class Panel:
    X: pd.DataFrame  # (date, symbol) indeksli özellikler (+ __missing bayrakları)
    y: pd.Series  # (date, symbol)
    merged: pd.DataFrame  # denetim kolonlarıyla (date, symbol, decision_time, <src>__available_at, ...)
    membership: pd.DataFrame  # tarih × sembol bool (PIT evren)
    prices: dict[str, pd.DataFrame]  # tarih × sembol (adaylar), nihai pencere hariç
    sources: list[Source]
    panels: dict[str, pd.DataFrame] = field(default_factory=dict)  # kaynak -> ham panel (kapsam raporu)
    target: Target | None = None
    candidates: list[str] = field(default_factory=list)
    data_hash: str = ""
    qv_all: pd.DataFrame | None = None


def _resolve_sources(specs: list) -> list[Source]:
    out = []
    for spec in specs:
        name, params = (spec, {}) if isinstance(spec, str) else (spec["name"], {k: v for k, v in spec.items() if k != "name"})
        out.append(get_source(name, **params))
    return out


def check_forward_only(sources: list[Source], min_days: int = 365) -> None:
    """forward_only kaynaklar 12 aylık veri biriktirmeden KULLANILAMAZ (KURALLAR.md §9)."""
    for s in sources:
        if s.forward_only:
            days = s.accumulated_days() if hasattr(s, "accumulated_days") else None
            if days is None or days < min_days:
                raise ForwardOnlyTooShortError(
                    f"'{s.name}' forward_only bir kaynak: {0 if days is None else days:.0f} gün birikti, en az {min_days} gün gerekli. "
                    "Veri toplamaya devam edin (bkz. forward_collectors_enabled)."
                )


def build_panel(cfg: dict) -> Panel:
    """`cfg`: doğrulanmış deney yapılandırması (bkz. runner.load_config)."""
    start, end = pd.Timestamp(cfg["period"]["start"]), pd.Timestamp(cfg["period"]["end"])
    assert_no_final_test(pd.DatetimeIndex([end]))
    sources = _resolve_sources(cfg["sources"])
    check_forward_only(sources)
    target = get_target(cfg["target"]["name"], **{k: v for k, v in cfg["target"].items() if k != "name"})

    qv_all = store.load_panel("quote_volume")
    if qv_all.empty:
        raise ValueError("Veri yok: önce `python -m research.data.download ...` çalıştırın.")
    membership_full = universe_membership(qv_all, cfg["universe"]["n"])
    dates = pd.date_range(start, end, freq="D")
    membership = membership_full.reindex(dates).fillna(False)
    candidates = [c for c in membership.columns if membership[c].any()]
    if not candidates:
        raise ValueError("Seçilen dönemde evren boş")

    full_idx = pd.date_range(qv_all.index.min(), qv_all.index.max(), freq="D")
    prices = {f: store.load_panel(f, candidates).reindex(full_idx) for f in OHLC}
    y_wide = target.compute(prices, membership_full.reindex(full_idx).fillna(False))

    data_hash = store.snapshot_hash()
    decisions = decision_frame(dates, universe_membership=membership.loc[:, candidates])
    panels = {}
    for src in sources:
        panel = source_cache.load_or_build(src, candidates, dates, data_hash)
        validate_panel(src, panel)
        panels[src.name] = panel
    merged = merge_sources(decisions, [(s, panels[s.name]) for s in sources])

    idx = pd.MultiIndex.from_frame(merged[["date", "symbol"]])
    X = merged[feature_matrix_columns(merged)].copy()
    X.index = idx
    y_long = y_wide.stack(future_stack=True).rename("y")
    y_long.index = y_long.index.set_names(["date", "symbol"])
    y = y_long.reindex(idx)
    return Panel(X=X, y=y, merged=merged.set_index(idx), membership=membership, prices={k: v for k, v in prices.items()}, sources=sources,
                 panels=panels, target=target, candidates=candidates, data_hash=data_hash, qv_all=qv_all)
