"""Kaynak kapsam raporu: tarih aralığı, sembol kapsamı, eksik oranı (kaynak başına)."""

from __future__ import annotations

import pandas as pd

from .base import Source
from .pit import MISSING_SUFFIX, decision_frame, merge_sources


def source_coverage(source: Source, panel: pd.DataFrame, universe: list[str], dates: pd.DatetimeIndex) -> dict:
    """`panel`: `source.to_panel` çıktısı. Eksik oranı, (tarih × sembol) karar ızgarasına PIT birleştirmeden sonra ölçülür
    (yani 'o kararda bu kaynaktan hiç özellik yok' oranı) ve hücre bazında NaN oranı."""
    decisions = decision_frame(dates, symbols=universe)
    merged = merge_sources(decisions, [(source, panel)])
    feats = source.feature_columns(panel)
    row_missing = float(merged[f"{source.name}{MISSING_SUFFIX}"].mean()) if len(merged) else float("nan")
    cell_nan = float(merged[feats].isna().mean().mean()) if feats and len(merged) else float("nan")
    real = panel[panel["symbol"] != "__MARKET__"] if source.scope == "symbol" else panel
    symbols = sorted(real["symbol"].unique()) if len(real) else []
    return {
        "source": source.name, "version": source.version, "scope": source.scope, "forward_only": source.forward_only,
        "publication_lag": str(source.publication_lag),
        "first_date": None if panel.empty else str(panel["date"].min())[:10],
        "last_date": None if panel.empty else str(panel["date"].max())[:10],
        "panel_rows": int(len(panel)),
        "symbols_with_data": len(symbols) if source.scope == "symbol" else None,
        "universe_size": len(universe),
        "symbol_coverage": (len(set(symbols) & set(universe)) / len(universe)) if (source.scope == "symbol" and universe) else None,
        "decision_rows_missing": row_missing,
        "cell_nan_rate": cell_nan,
        "n_features": len(feats),
    }


def coverage_markdown(entries: list[dict], title: str = "Kaynak kapsamı") -> str:
    lines = [f"# {title}", "", "| Kaynak | Sürüm | Kapsam | Aralık | Satır | Sembol kapsamı | Karar satırı eksik | Hücre NaN | Yayın gecikmesi | forward_only |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for e in entries:
        sc = "—" if e["symbol_coverage"] is None else f"{e['symbols_with_data']}/{e['universe_size']} ({e['symbol_coverage']:.0%})"
        lines.append(
            f"| {e['source']} | {e['version']} | {e['scope']} | {e['first_date']} → {e['last_date']} | {e['panel_rows']} | {sc} | "
            f"{e['decision_rows_missing']:.1%} | {e['cell_nan_rate']:.1%} | {e['publication_lag']} | {e['forward_only']} |"
        )
    return "\n".join(lines)
