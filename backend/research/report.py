"""Tek deney için markdown + PNG raporu: NAV, drawdown, yıllık getiri çubukları, BTC al-tut kıyası."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # başsız (konteyner) çizim
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import stats  # noqa: E402
from .guard import assert_no_final_test  # noqa: E402


def _fmt(x, pct=False, nd=2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.{nd}f}%" if pct else f"{x:.{nd}f}"


def render_report(
    returns: pd.Series,
    benchmark: pd.Series | None,
    out_dir: str | Path,
    title: str,
    *,
    extra: dict | None = None,
    allow_final_test: bool = False,
    final_test_experiment_id: str | None = None,
) -> dict:
    """`returns`: günlük net getiri (indeks tarih). `benchmark`: BTC al-tut günlük getirisi (aynı tarihlerde).
    `out_dir/report.md` ve `out_dir/report.png` yazar; metrik sözlüğünü döner."""
    assert_no_final_test(returns.index, allow_final_test, final_test_experiment_id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = returns.dropna()
    bench = benchmark.reindex(r.index).dropna() if benchmark is not None else None

    m = stats.summary(r, benchmark=bench)
    bm = stats.summary(bench) if bench is not None and len(bench) else None

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), gridspec_kw={"height_ratios": [3, 1.6, 1.8]})
    nav = (1 + r).cumprod()
    axes[0].plot(nav.index, nav, label=title, lw=1.4)
    if bench is not None and len(bench):
        axes[0].plot(bench.index, (1 + bench).cumprod(), label="BTC al-tut", lw=1.0, alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_title(f"{title} — NAV (log)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(r.index, stats.drawdown_series(r) * 100, 0, alpha=0.5, label=title)
    if bench is not None and len(bench):
        axes[1].plot(bench.index, stats.drawdown_series(bench) * 100, lw=0.8, color="gray", label="BTC al-tut")
    axes[1].set_title("Drawdown (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    yr = stats.yearly_returns(r) * 100
    years = sorted(yr.index)
    x = np.arange(len(years))
    width = 0.4
    axes[2].bar(x - width / 2, [yr[y] for y in years], width, label=title)
    if bench is not None and len(bench):
        byr = stats.yearly_returns(bench) * 100
        axes[2].bar(x + width / 2, [byr.get(y, np.nan) for y in years], width, label="BTC al-tut", alpha=0.7)
    axes[2].set_xticks(x, [str(y) for y in years])
    axes[2].axhline(0, color="black", lw=0.6)
    axes[2].set_title("Takvim yılı getirisi (%)")
    axes[2].legend()
    axes[2].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    png = out_dir / "report.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)

    rows = [
        ("Yıllık getiri", _fmt(m["annual_return"], True), _fmt(bm["annual_return"], True) if bm else "—"),
        ("Yıllık vol", _fmt(m["annual_vol"], True), _fmt(bm["annual_vol"], True) if bm else "—"),
        ("Sharpe", _fmt(m["sharpe"]), _fmt(bm["sharpe"]) if bm else "—"),
        ("Sortino", _fmt(m["sortino"]), _fmt(bm["sortino"]) if bm else "—"),
        ("Calmar", _fmt(m["calmar"]), _fmt(bm["calmar"]) if bm else "—"),
        ("Maks. drawdown", _fmt(m["max_drawdown"], True), _fmt(bm["max_drawdown"], True) if bm else "—"),
        ("Maks. DD süresi (gün)", str(m["max_drawdown_days"]), str(bm["max_drawdown_days"]) if bm else "—"),
    ]
    lines = [f"# {title}", "", f"Dönem: {r.index.min():%Y-%m-%d} → {r.index.max():%Y-%m-%d} ({m['n_days']} gün)", "",
             "| Metrik | Strateji | BTC al-tut |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    if "beta_vs_btc" in m:
        lines += ["", f"BTC'ye beta: {_fmt(m['beta_vs_btc'])} · korelasyon: {_fmt(m['corr_vs_btc'])}"]
    lines += ["", "Takvim yılı getirileri: " + ", ".join(f"{y}: {_fmt(v, True, 1)}" for y, v in sorted(m["yearly_returns"].items()))]
    if extra:
        lines += ["", "## Ek bilgiler", ""] + [f"- **{k}**: {v}" for k, v in extra.items()]
    lines += ["", "![rapor](report.png)"]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"metrics": m, "benchmark_metrics": bm, "png": str(png), "md": str(out_dir / "report.md")}
