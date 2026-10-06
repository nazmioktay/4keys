"""Ablasyon: config'teki her kaynağı tek tek çıkarıp yeniden koşturur; her kaynağın IC'ye ve net Sharpe'a MARJİNAL
katkısını EŞLEŞTİRİLMİŞ durağan blok bootstrap güven aralığıyla raporlar (aynı blok indeksleri iki seriye de uygulanır).

Her ablasyon bir deneme (config) sayılır: `register=True` ise her biri `<id>__abl_<kaynak>` adıyla kaydedilir (sayaç + bütçe)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import guard, pred_metrics, signals, stats
from .oof import build_folds, run_oof
from .panel import Panel
from .registry import assert_budget, record_smoke_run, register_experiment
from .runner import _portfolio, _trim, check_panel_zone, load_config


def _primary_returns(panel: Panel, cfg: dict, X: pd.DataFrame, folds, seed: int):
    model = next(m for m in cfg["models"] if m["id"] == cfg["primary_model"])
    res = run_oof(X, panel.y, [model], panel.target, cfg["cv"], seed, folds)
    pred = res.oof[model["id"]]
    first = pred.dropna().index.get_level_values("date").min()
    weights = signals.make_weights(cfg["signal"]["adapter"], pred.unstack("symbol").reindex(columns=panel.candidates),
                                   panel.membership.reindex(columns=panel.candidates).fillna(False), **cfg["signal"]["params"])
    returns = _trim(_portfolio(panel, weights).returns, first)
    ic = pred_metrics.daily_ic(pred, panel.y)[0]
    return pred, returns, ic


def _paired_ci(a: pd.Series, b: pd.Series, stat, n_boot: int, mean_block: float, seed: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """stat(a) − stat(b) için nokta tahmini ve eşleştirilmiş blok bootstrap güven aralığı."""
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    n = len(both)
    point = float(stat(both["a"]) - stat(both["b"])) if n > 5 else float("nan")
    if n < 20:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    av, bv = both["a"].to_numpy(), both["b"].to_numpy()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = stats.stationary_bootstrap_indices(n, mean_block, rng)
        diffs[i] = stat(pd.Series(av[idx])) - stat(pd.Series(bv[idx]))
    lo, hi = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def run_ablation(
    cfg_source, panel: Panel, *, n_boot: int = 300, mean_block: float = 15.0, seed: int = 0,
    register: bool = True, smoke: bool = False, results_dir=None, log_path=None, registry_file=None,
) -> pd.DataFrame:
    """Tam model ile her kaynağı dışarıda bırakan modeli kıyaslar. Dönen tablo (satır = kaynak):
    d_ic, d_ic_lo, d_ic_hi, d_sharpe, d_sharpe_lo, d_sharpe_hi (tam − kaynaksız; pozitif = kaynak katkı veriyor).
    VARSAYILAN `register=True`: her ablasyon ayrı bir config/deneme olarak kaydedilir (`<id>__abl_<kaynak>`; sayaç + bütçe).
    Kayıtsız çalıştırmak yalnızca `smoke=True` ile mümkündür (smoke soru başına sınırlıdır; sonuç karar için kullanılamaz)."""
    cfg = load_config(cfg_source)
    check_panel_zone(panel, cfg)
    with guard.zone(cfg["zone"]):  # motor/funding yükleyicisi deneyin bölgesinde (KURALLAR.md §10)
        return _run_ablation(cfg, panel, n_boot, mean_block, seed, register, smoke, results_dir, log_path, registry_file)


def _run_ablation(cfg, panel, n_boot, mean_block, seed, register, smoke, results_dir, log_path, registry_file) -> pd.DataFrame:
    if not register and not smoke:
        raise ValueError("Kayıtsız ablasyon yalnızca smoke=True ile yapılabilir (deneme/bütçe kaçağını önlemek için)")
    if smoke:
        record_smoke_run(cfg["question"], registry_file)
    if register:
        assert_budget(cfg["question"], len(panel.sources), registry_file)  # hepsi sığmıyorsa HİÇ başlama
    dates = pd.Series(panel.X.index.get_level_values("date"), index=panel.X.index)
    folds = build_folds(dates, cfg["cv"], panel.target)
    _, ret_full, ic_full = _primary_returns(panel, cfg, panel.X, folds, cfg["seed"])
    rows = {}
    for src in panel.sources:
        keep = [c for c in panel.X.columns if not c.startswith(f"{src.name}__")]
        if not keep:
            rows[src.name] = {k: float("nan") for k in ("d_ic", "d_ic_lo", "d_ic_hi", "d_sharpe", "d_sharpe_lo", "d_sharpe_hi")}
            continue
        _, ret_wo, ic_wo = _primary_returns(panel, cfg, panel.X[keep], folds, cfg["seed"])
        d_ic = _paired_ci(ic_full, ic_wo, lambda s: float(s.mean()), n_boot, mean_block, seed)
        d_sh = _paired_ci(ret_full, ret_wo, stats.sharpe, n_boot, mean_block, seed + 1)
        rows[src.name] = dict(zip(("d_ic", "d_ic_lo", "d_ic_hi"), d_ic)) | dict(zip(("d_sharpe", "d_sharpe_lo", "d_sharpe_hi"), d_sh))
        if register:
            mdd = stats.max_drawdown(ret_wo)[0]
            register_experiment(
                f"{cfg['id']}__abl_{src.name}", {**cfg, "ablated_source": src.name}, ret_wo,
                {"sharpe": stats.sharpe(ret_wo), "max_drawdown": mdd, "ablation_of": cfg["id"], "d_ic": d_ic[0], "d_sharpe": d_sh[0]},
                n_variants=1, hypothesis=f"ablasyon: {src.name} çıkarıldı", decision="ablasyon (karar yok)", question=cfg["question"],
                results_dir=results_dir, log_path=log_path, registry_file=registry_file, data_hash=panel.data_hash,
            )
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "kaynak"
    return table


def ablation_markdown(table: pd.DataFrame) -> str:
    lines = ["| Kaynak | ΔIC | ΔIC %95 GA | ΔSharpe | ΔSharpe %95 GA |", "|---|---|---|---|---|"]
    for src, r in table.iterrows():
        lines.append(f"| {src} | {r['d_ic']:+.4f} | [{r['d_ic_lo']:+.4f}, {r['d_ic_hi']:+.4f}] | {r['d_sharpe']:+.3f} | [{r['d_sharpe_lo']:+.3f}, {r['d_sharpe_hi']:+.3f}] |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from .panel import build_panel

    ap = argparse.ArgumentParser(description="Kaynak ablasyonu")
    ap.add_argument("config")
    ap.add_argument("--smoke", action="store_true", help="Kaydetme (deneme sayılmaz; soru başına sınırlı, karar için KULLANILAMAZ)")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    table = run_ablation(cfg, build_panel(cfg), n_boot=args.n_boot, register=not args.smoke, smoke=args.smoke)
    print(ablation_markdown(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
