"""Veri kalitesi raporu: eksik günler, sıfır hacim, log getirisi > 5σ sıçramalar. Sessiz doldurma YOK."""

from __future__ import annotations

import numpy as np
import pandas as pd


def symbol_quality(df: pd.DataFrame, freq: str = "D", jump_sigma: float = 5.0) -> dict:
    """Tek sembol için kalite özeti. `df` indeksi open_time; 'close' ve 'volume' kolonları gerekir."""
    if df is None or df.empty:
        return {"rows": 0}
    idx = df.index
    expected = pd.date_range(idx.min(), idx.max(), freq=freq)
    missing = expected.difference(idx)
    zero_volume = df.index[(df["volume"] <= 0).to_numpy()]
    log_ret = np.log(df["close"].where(df["close"] > 0)).diff().dropna()
    sigma = float(log_ret.std()) if len(log_ret) > 2 else float("nan")
    jumps = log_ret[(log_ret.abs() > jump_sigma * sigma)] if sigma == sigma and sigma > 0 else log_ret.iloc[0:0]
    duplicates = int(idx.duplicated().sum())
    return {
        "rows": int(len(df)),
        "first": idx.min(),
        "last": idx.max(),
        "expected": int(len(expected)),
        "missing_days": int(len(missing)),
        "missing_list": [d.strftime("%Y-%m-%d") for d in missing[:20]],
        "zero_volume_days": int(len(zero_volume)),
        "zero_volume_list": [d.strftime("%Y-%m-%d") for d in zero_volume[:20]],
        "jumps": [(d.strftime("%Y-%m-%d"), float(v)) for d, v in jumps.head(20).items()],
        "n_jumps": int(len(jumps)),
        "duplicates": duplicates,
    }


def quality_report(frames: dict[str, pd.DataFrame], freq: str = "D", jump_sigma: float = 5.0) -> pd.DataFrame:
    """Sembol başına özet tablo (satır = sembol)."""
    rows = {s: symbol_quality(df, freq, jump_sigma) for s, df in frames.items()}
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table.sort_index()


def to_markdown(table: pd.DataFrame, title: str = "Veri kalitesi raporu", max_rows: int = 40) -> str:
    """İnsan okunur özet; sorunlu semboller (eksik gün/sıfır hacim/sıçrama) önce."""
    lines = [f"# {title}", ""]
    if table.empty:
        return "\n".join(lines + ["(veri yok)"])
    t = table.copy()
    for col in ("missing_days", "zero_volume_days", "n_jumps", "duplicates"):
        if col not in t:
            t[col] = 0
    issues = t[(t["missing_days"] > 0) | (t["zero_volume_days"] > 0) | (t["n_jumps"] > 0) | (t["duplicates"] > 0)]
    lines += [
        f"- Sembol sayısı: {len(t)}",
        f"- İlk tarih: {t['first'].min()}  ·  Son tarih: {t['last'].max()}",
        f"- Eksik günü olan sembol: {int((t['missing_days'] > 0).sum())} (toplam {int(t['missing_days'].sum())} gün)",
        f"- Sıfır hacimli gün: {int(t['zero_volume_days'].sum())}",
        f"- |log getiri| > 5σ sıçrama: {int(t['n_jumps'].sum())}",
        "",
        "Eksik günler ve sıçramalar DOLDURULMAZ/DÜZELTİLMEZ; tüketici (motor/evren) NaN'ı olduğu gibi görür.",
        "",
        "| Sembol | İlk | Son | Satır | Eksik gün | Sıfır hacim | Sıçrama | Örnek eksik günler |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ranked = issues.sort_values(["missing_days", "n_jumps"], ascending=False).head(max_rows)
    for sym, r in ranked.iterrows():
        lines.append(
            f"| {sym} | {str(r['first'])[:10]} | {str(r['last'])[:10]} | {int(r['rows'])} | {int(r['missing_days'])} | "
            f"{int(r['zero_volume_days'])} | {int(r['n_jumps'])} | {', '.join(r.get('missing_list') or [])[:60]} |"
        )
    if len(issues) > max_rows:
        lines.append(f"| … | | | | | | | (+{len(issues) - max_rows} sembol daha) |")
    return "\n".join(lines)
