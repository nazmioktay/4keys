"""Kaynak kapsam raporu -> docs/research/kaynak_kapsami.md (tekrar üretilebilir).

Kullanım:  cd backend && python -m research.coverage_report [--n 20] [--start 2020-01-01] [--end 2025-09-30]

Her (forward_only olmayan) kaynak, ana bölgede (nihai pencere hariç) noktasal-zamanlı ilk-N evreninin aday sembolleri için
panele çevrilir ve `sources.coverage.source_coverage` ile ölçülür. forward_only kaynaklar DB'de (VPS) biriktiği için burada
ölçülmez; durumları `deploy/forward-durum.sh` ile izlenir."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config
from .data import store
from .data.universe import universe_membership
from .guard import assert_no_final_test
from .sources.coverage import coverage_markdown, source_coverage
from .sources.registry import available_sources, get_source

DEFAULT_OUT = config.DOCS_DIR / "kaynak_kapsami.md"


def _metrics_archive_summary() -> list[str]:
    """`download_metrics --coverage` listesinden (S3) arşiv kapsamı + yerelde indirilmiş semboller."""
    lines = []
    listing = Path(config.CACHE_DIR) / "metrics_coverage.csv"
    if listing.exists():
        cov = pd.read_csv(listing)
        lines.append(f"- Binance `metrics` arşivinde (S3 listesi) **{len(cov)}** sembol var; en erken gün {cov['first'].min()}.")
    downloaded = store.list_symbols("um_metrics_1d")
    lines.append(f"- Yerelde indirilmiş (`um_metrics_1d`): **{len(downloaded)}** sembol: {', '.join(downloaded) or '—'}. "
                 "Diğer semboller için `derivatives_metrics` özellikleri NaN'dır; genişletmek için "
                 "`python -m research.data.download_metrics --symbols ...`.")
    return lines


def build_report(n: int, start: str, end: str) -> str:
    dates = pd.date_range(start, end, freq="D")
    assert_no_final_test(dates)  # rapor ana bölge içindir: nihai pencereye uzanan dönem oranları şişirir ve metni yanıltır
    qv_all = store.load_panel("quote_volume")
    if qv_all.empty:
        raise SystemExit("Veri yok: önce `python -m research.data.download ...` çalıştırın.")
    membership = universe_membership(qv_all, n).reindex(dates).fillna(False)
    candidates = [c for c in membership.columns if membership[c].any()]

    entries, errors, forward = [], [], []
    for name in available_sources():
        src = get_source(name)
        if src.forward_only:
            forward.append(name)
            continue
        try:
            panel = src.to_panel(candidates, dates)
            entries.append(source_coverage(src, panel, candidates, dates))
        except Exception as exc:  # noqa: BLE001 - rapor: bir kaynağın hatası diğerlerini durdurmasın, açıkça yazılsın
            errors.append(f"- `{name}`: {type(exc).__name__}: {exc}")

    out = [
        coverage_markdown(entries, title="Kaynak kapsamı"),
        "",
        f"Üretim: {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC · `python -m research.coverage_report --n {n} --start {start} --end {end}` · "
        f"veri hash `{store.snapshot_hash()}`",
        "",
        "## Nasıl okunur",
        f"- **Dönem** {start} → {end} (ana bölge; nihai pencere >= {config.FINAL_TEST_START:%Y-%m-%d} hariç). "
        f"**Evren:** noktasal-zamanlı ilk-{n}; dönem boyunca evrene en az bir gün girmiş **{len(candidates)}** aday sembol.",
        "- **Karar satırı eksik:** (gün × aday sembol) karar ızgarasında o kaynaktan HİÇ özellik olmayan satır oranı (noktasal-zamanlı "
        "birleştirmeden sonra). Sembolün listelenmediği günler de ızgarada olduğu için oran, gerçek kullanılabilir eksiklikten yüksektir.",
        "- **Hücre NaN:** aynı ızgarada tüm özellik hücrelerinin NaN oranı (uzun pencereli özelliklerin ısınması dahil).",
        "- Piyasa-geneli kaynaklarda (`market`) sembol kapsamı anlamsızdır (—).",
        "",
        "## Türev metrikleri arşivi",
        *_metrics_archive_summary(),
        "",
        "## İleriye dönük (forward_only) kaynaklar",
        f"- {', '.join(f'`{f}`' for f in forward) or '—'}: VPS veritabanında 2026-10-06'dan beri birikiyor; burada ölçülmez. "
        "Durum: `bash deploy/forward-durum.sh`. Kullanım kuralı: KURALLAR.md §10 (yalnızca `zone: forward`, keşif 2026-11-01 → 2027-11-01).",
    ]
    if errors:
        out += ["", "## Ölçülemeyen kaynaklar", *errors]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Kaynak kapsam raporu üret")
    ap.add_argument("--n", type=int, default=20, help="noktasal-zamanlı evren büyüklüğü")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=str((config.FINAL_TEST_START - pd.Timedelta(days=1)).date()))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    text = build_report(args.n, args.start, args.end)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
