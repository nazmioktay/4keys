"""Noktasal-zamanlı (point-in-time) birleştirme.

Her karar satırı (`date`, `symbol`) için karar zamanı = date + 1 gün (günlük kapanış). Her kaynaktan, `available_at <=
karar zamanı` olan EN SON satır `pd.merge_asof(direction="backward", tolerance=max_staleness)` ile alınır.

ÖNEMLİ: eksik değer 0 ile DOLDURULMAZ — NaN kalır ve `<kaynak>__missing` bayrağı eklenir (1 = o kaynağın tüm
özellikleri bu satırda NaN). Önceki sürümde makro özellikleri 0'la doldurmak eğitim ve canlı arasında uyumsuzluk
yaratmıştı. Denetim için `<kaynak>__available_at` kolonu taşınır (modele verilmez; `MODEL_EXCLUDE_SUFFIXES`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import MARKET, Source

DECISION_DELAY = pd.Timedelta(days=1)  # gün kapanışı = date + 1 gün 00:00 UTC
AVAILABLE_SUFFIX = "__available_at"
MISSING_SUFFIX = "__missing"
MODEL_EXCLUDE_SUFFIXES = (AVAILABLE_SUFFIX,)


def decision_frame(dates: pd.DatetimeIndex, universe_membership: pd.DataFrame | None = None, symbols: list[str] | None = None) -> pd.DataFrame:
    """Karar satırları [date, symbol, decision_time]. `universe_membership`: tarih × sembol boolean (PIT evren);
    verilmezse `symbols` × `dates` tam ızgara."""
    if universe_membership is not None:
        stacked = universe_membership.stack()
        rows = stacked[stacked].index.to_frame(index=False)
        rows.columns = ["date", "symbol"]
    else:
        rows = pd.MultiIndex.from_product([pd.DatetimeIndex(dates), symbols or []], names=["date", "symbol"]).to_frame(index=False)
    rows["date"] = pd.to_datetime(rows["date"])
    rows["decision_time"] = rows["date"] + DECISION_DELAY
    return rows.sort_values(["date", "symbol"]).reset_index(drop=True)


def merge_sources(decisions: pd.DataFrame, sources: list[tuple[Source, pd.DataFrame]]) -> pd.DataFrame:
    """`decisions` ([date, symbol, decision_time]) satırlarına her kaynağın özelliklerini PIT ekler.
    Dönen çerçevenin satır sırası/indeksi `decisions` ile aynıdır."""
    out = decisions.copy()
    left_base = decisions[["symbol", "decision_time"]].copy()
    left_base["_row"] = decisions.index
    left_base = left_base.sort_values("decision_time", kind="stable")

    for source, panel in sources:
        if panel.empty:  # hiç veri yok (ör. henüz veri toplanmamış forward_only kaynak): NaN + eksik bayrağı, 0 DEĞİL
            feats = list(source.feature_names)
            for c in feats:
                out[c] = np.nan
            out[f"{source.name}{AVAILABLE_SUFFIX}"] = pd.NaT
            out[f"{source.name}{MISSING_SUFFIX}"] = np.int8(1)
            continue
        panel = panel.assign(available_at=pd.to_datetime(panel["available_at"]))
        feats = source.feature_columns(panel)
        right = panel[["symbol", "available_at", *feats]].dropna(subset=["available_at"]).sort_values("available_at", kind="stable")
        avail_col = f"{source.name}{AVAILABLE_SUFFIX}"
        right = right.rename(columns={"available_at": avail_col})
        if source.scope == "market":
            right = right.drop(columns=["symbol"])
            merged = pd.merge_asof(
                left_base[["decision_time", "_row"]], right, left_on="decision_time", right_on=avail_col,
                direction="backward", tolerance=source.max_staleness,
            )
        else:
            merged = pd.merge_asof(
                left_base, right, left_on="decision_time", right_on=avail_col, by="symbol",
                direction="backward", tolerance=source.max_staleness,
            )
        merged = merged.set_index("_row").sort_index()
        block = merged[feats + [avail_col]].reindex(decisions.index)
        out = pd.concat([out, block], axis=1)
        out[f"{source.name}{MISSING_SUFFIX}"] = block[feats].isna().all(axis=1).astype("int8")
    return out


def audit_available_at(merged: pd.DataFrame, sources: list[Source]) -> dict[str, int]:
    """Her kaynak için `available_at > karar zamanı` ihlal sayısı (0 olmalı). NaN (eşleşmeyen) satırlar sayılmaz."""
    violations = {}
    for source in sources:
        col = f"{source.name}{AVAILABLE_SUFFIX}"
        if col not in merged.columns:
            continue
        violations[source.name] = int((merged[col] > merged["decision_time"]).sum())
    return violations


def feature_matrix_columns(merged: pd.DataFrame) -> list[str]:
    """Modele verilecek kolonlar: kaynak özellikleri + `__missing` bayrakları (denetim kolonları hariç)."""
    skip = {"date", "symbol", "decision_time"}
    return [c for c in merged.columns if c not in skip and not c.endswith(MODEL_EXCLUDE_SUFFIXES)]


__all__ = ["MARKET", "decision_frame", "merge_sources", "audit_available_at", "feature_matrix_columns", "AVAILABLE_SUFFIX", "MISSING_SUFFIX"]
