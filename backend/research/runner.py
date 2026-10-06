"""Deney koşucusu: tek bir YAML config -> kaynaklar × hedef × model × CV × sinyal adaptörü × kol.

Kullanım:  cd backend && python -m research.runner research/configs/ornek_vol.yaml [--smoke]

Sıra (KURALLAR.md ve plan §4): a) tarih bazlı purge+embargo'lu walk-forward, b) OOF -> oof.parquet, c) tahmin metrikleri,
d) sinyal adaptörüyle tahmin -> kol ağırlığı, e) Aşama 0 motoru + KURALLAR metrikleri, f) global deneme sayacı + Deflated
Sharpe, g) deneyler.md satırı + rapor. Sızıntı/sağlamlık testlerinden biri başarısızsa deney GEÇERSİZdir (yine kaydedilir)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import checks as chk
from . import config, engine, pred_metrics, signals, stats
from .budget import assert_budget
from .data import store
from .data.universe import universe_membership
from .oof import OofResult, build_folds, run_oof
from .panel import Panel, build_panel
from .registry import ExperimentExistsError, current_trial_count, is_registered, register_experiment
from .report import render_report

DEFAULT_CHECKS = {"shuffled_target": True, "availability": True, "extra_lag": True, "reproducibility": True}


class ConfigError(ValueError):
    pass


@dataclass
class ExperimentResult:
    id: str
    valid: bool
    smoke: bool
    metrics: dict
    checks: chk.CheckReport
    out_dir: Path
    oof_path: Path
    registered: bool
    warnings: list[str] = field(default_factory=list)


def load_config(source) -> dict:
    """YAML yolu veya sözlük -> doğrulanmış, varsayılanları doldurulmuş yapılandırma."""
    cfg = yaml.safe_load(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else json.loads(json.dumps(source))
    for key in ("id", "sources", "period", "target", "models", "signal"):
        if key not in cfg:
            raise ConfigError(f"config'te '{key}' zorunlu")
    cfg.setdefault("smoke", False)
    cfg.setdefault("question", "genel")
    cfg.setdefault("universe", {"n": 20})
    cfg.setdefault("arm", "arm")
    cfg.setdefault("seed", 0)
    cfg["checks"] = {**DEFAULT_CHECKS, **cfg.get("checks", {})}
    models = []
    for m in cfg["models"]:
        entry = {"name": m, "params": {}} if isinstance(m, str) else {"name": m["name"], "params": m.get("params", {})}
        entry["id"] = m.get("id", entry["name"]) if isinstance(m, dict) else entry["name"]
        models.append(entry)
    ids = [m["id"] for m in models]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"model kimlikleri benzersiz olmalı: {ids}")
    cfg["models"] = models
    cfg["primary_model"] = cfg.get("primary_model", ids[0])
    if cfg["primary_model"] not in ids:
        raise ConfigError(f"primary_model '{cfg['primary_model']}' model listesinde yok")
    h = int(cfg["target"].get("h", 1))
    cv = {"n_splits": 5, "embargo_days": h, "min_train_days": 365, **cfg.get("cv", {})}
    if cv["embargo_days"] < h:
        raise ConfigError(f"cv.embargo_days ({cv['embargo_days']}) hedef ufkundan (h={h}) küçük olamaz")
    cfg["cv"] = cv
    cfg["signal"].setdefault("params", {})
    if not re_id(cfg["id"]):
        raise ConfigError("id yalnızca harf/rakam/_.- içerebilir")
    return cfg


def re_id(s: str) -> bool:
    import re

    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+", str(s)))


# ------------------------------------------------------------------ portföy hesabı
def _slippage_frame(panel: Panel, index: pd.DatetimeIndex) -> pd.DataFrame:
    top20 = universe_membership(panel.qv_all, 20).reindex(index).fillna(False)
    out = pd.DataFrame(config.SLIPPAGE_BPS_OTHER, index=index, columns=panel.candidates)
    for c in panel.candidates:
        if c in top20.columns:
            out[c] = np.where(top20[c].to_numpy(), config.SLIPPAGE_BPS_TOP20, config.SLIPPAGE_BPS_OTHER)
        if c in config.TOP_TIER_SYMBOLS:
            out[c] = config.SLIPPAGE_BPS_BTC_ETH
    return out


def _funding_frame(panel: Panel, index: pd.DatetimeIndex) -> pd.DataFrame | None:
    rows = []
    for s in panel.candidates:
        f = store.load_funding(s)
        if f is not None and len(f):
            rows.append(pd.DataFrame({"symbol": s, "time": f.index, "rate": f["rate"].to_numpy()}))
    return engine.daily_funding(pd.concat(rows, ignore_index=True), index, panel.candidates) if rows else None


def _portfolio(panel: Panel, weights: pd.DataFrame, fee_mult: float = 1.0, slip_mult: float = 1.0, delay: int = 0):
    start = weights.index[weights.abs().sum(axis=1) > 0].min() if (weights.abs().sum(axis=1) > 0).any() else weights.index.min()
    open_px = panel.prices["open"].loc[start:, panel.candidates]
    w = weights.reindex(open_px.index).fillna(0.0)[panel.candidates]
    slip = _slippage_frame(panel, open_px.index) * slip_mult
    result = engine.run(
        w, open_px, fee_rate=config.FUTURES_TAKER_FEE * fee_mult, slippage_bps=slip,
        funding=_funding_frame(panel, open_px.index), delay_bars=delay,
    )
    return result


def _benchmark(panel: Panel, start: pd.Timestamp):
    if "BTCUSDT" not in panel.prices["open"].columns:
        return None
    px = panel.prices["open"].loc[start:, ["BTCUSDT"]]
    w = pd.DataFrame(1.0, index=px.index, columns=["BTCUSDT"])
    return engine.run(w, px, fee_rate=config.FUTURES_TAKER_FEE, slippage_bps=config.SLIPPAGE_BPS_BTC_ETH)


def _trim(returns: pd.Series, start: pd.Timestamp) -> pd.Series:
    return returns[returns.index >= start]


# ------------------------------------------------------------------ ana akış
def run_experiment(
    source,
    *,
    smoke: bool | None = None,
    results_dir: Path | None = None,
    log_path: Path | None = None,
    registry_file: Path | None = None,
    panel: Panel | None = None,
    extra_question_cost: int = 1,
) -> ExperimentResult:
    cfg = load_config(source)
    if smoke is not None:
        cfg["smoke"] = smoke
    smoke = bool(cfg["smoke"])
    results_dir = Path(results_dir or config.RESULTS_DIR)
    registry_file = Path(registry_file or config.REGISTRY_FILE)
    if not smoke:
        if is_registered(cfg["id"], registry_file, results_dir):  # ağır işe başlamadan: kayıtlar değiştirilmez
            raise ExperimentExistsError(f"{cfg['id']} zaten kayıtlı (kayıtlar değiştirilmez)")
        assert_budget(cfg["question"], extra_question_cost, registry_file)  # bütçe dolduysa YENİ deneme yapılmaz

    panel = panel or build_panel(cfg)
    target, seed, cv = panel.target, int(cfg["seed"]), cfg["cv"]
    dates = pd.Series(panel.X.index.get_level_values("date"), index=panel.X.index)
    folds = build_folds(dates, cv, target)
    res = run_oof(panel.X, panel.y, cfg["models"], target, cv, seed, folds)

    out_dir = (results_dir / "_smoke" / cfg["id"]) if smoke else (results_dir / cfg["id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_path = out_dir / "oof.parquet"
    pd.concat([res.oof, panel.y.rename("y")], axis=1).to_parquet(oof_path)

    # (c) tahmin metrikleri
    pred_m = {m: pred_metrics.evaluate(res.oof[m], panel.y, target.kind, target.horizon, target.name) for m in res.oof.columns}

    # sızıntı / sağlamlık testleri
    report = chk.CheckReport()
    fold_arg = fold_tuples = [(f.train_mask, f.test_mask, f.train_dates, f.test_dates) for f in res.folds]
    flags = cfg["checks"]
    if flags["availability"]:
        chk.check_availability(panel, fold_tuples, target.horizon, report)
    if flags["shuffled_target"]:
        chk.check_shuffled_target(panel, cfg["models"], cv, seed, target.horizon, fold_arg, report)
    if flags["extra_lag"]:
        chk.check_extra_lag(panel, cfg["models"], cv, seed, target.horizon, fold_arg, res.oof, report)
    if flags["reproducibility"]:
        again = run_oof(panel.X, panel.y, cfg["models"], target, cv, seed, fold_arg)
        chk.check_reproducibility(res.oof, again.oof, report)

    # (d)(e) sinyal -> kol -> motor
    adapter, aparams = cfg["signal"]["adapter"], cfg["signal"]["params"]
    variants, ret_cols, results = {}, {}, {}
    first_oof = res.oof.dropna(how="all").index.get_level_values("date").min()
    for m in res.oof.columns:
        pred_wide = res.oof[m].unstack("symbol").reindex(columns=panel.candidates)
        weights = signals.make_weights(adapter, pred_wide, panel.membership.reindex(columns=panel.candidates).fillna(False), **aparams)
        results[m] = _portfolio(panel, weights)
        ret_cols[m] = _trim(results[m].returns, first_oof)
    bench = _benchmark(panel, first_oof)
    bench_ret = _trim(bench.returns, first_oof) if bench is not None else None
    primary = cfg["primary_model"]
    for m, r in ret_cols.items():
        variants[m] = {
            "summary": stats.summary(r, bench_ret), "costs": engine.cost_summary(results[m]), "pred": pred_m[m],
            "turnover_mean": float(results[m].turnover.mean()), "skipped_positions": results[m].diagnostics["skipped_positions"],
        }

    # PBO: modeller = tek bir deneyin varyantları
    returns_matrix = pd.concat(ret_cols, axis=1).dropna()
    pbo = float("nan")
    if returns_matrix.shape[1] >= 2 and len(returns_matrix) >= 16 * 8:
        pbo = stats.pbo_cscv(returns_matrix.to_numpy(), n_blocks=16)

    # stres (ücret ×2, kayma ×3) ve +1 bar gecikme — birincil varyant
    pm_weights = signals.make_weights(adapter, res.oof[primary].unstack("symbol").reindex(columns=panel.candidates),
                                      panel.membership.reindex(columns=panel.candidates).fillna(False), **aparams)
    a = config.ACCEPTANCE
    stress = _trim(_portfolio(panel, pm_weights, a["stress_fee_mult"], a["stress_slippage_mult"]).returns, first_oof)
    delayed = _trim(_portfolio(panel, pm_weights, delay=1).returns, first_oof)
    p = variants[primary]["summary"]
    yearly = p["yearly_returns"]
    pos_years = (sum(1 for v in yearly.values() if v > 0) / len(yearly)) if yearly else float("nan")
    btc_calmar = stats.calmar(bench_ret) if bench_ret is not None else float("nan")
    n_after = current_trial_count(registry_file) + len(cfg["models"])
    dsr = stats.deflated_sharpe(ret_cols[primary], n_trials=n_after)
    acceptance = stats.check_acceptance(
        sharpe_net=p["sharpe"], deflated=dsr, pbo=pbo, max_dd=p["max_drawdown"], calmar_value=p["calmar"], btc_calmar=btc_calmar,
        positive_year_fraction=pos_years, stress_sharpe=stats.sharpe(stress), plateau_ratio=float("nan"), is_portfolio=(cfg["arm"] == "portfolio"),
    )
    metrics = {
        "id": cfg["id"], "question": cfg["question"], "primary": primary, "sharpe": p["sharpe"], "max_drawdown": p["max_drawdown"],
        "variants": variants, "pbo": pbo, "stress_sharpe": stats.sharpe(stress), "delay1_sharpe": stats.sharpe(delayed),
        "btc_calmar": btc_calmar, "acceptance": acceptance, "acceptance_not_evaluated": ["plateau (parametre taraması gerekir)"],
        "valid": report.valid, "check_results": report.results, "check_failures": report.failures, "check_warnings": report.warnings,
        "folds": len(res.folds), "n_features": int(panel.X.shape[1]), "target": target.describe(), "smoke": smoke,
    }
    metrics["deflated_sharpe_if_registered"] = dsr

    extra = {
        "geçerli": report.valid, "uyarılar": ", ".join(report.warnings) or "yok", "başarısız testler": ", ".join(report.failures) or "yok",
        "birincil model": primary, "PBO": f"{pbo:.2f}" if pbo == pbo else "—", "stres Sharpe (ücret×2, kayma×3)": f"{stats.sharpe(stress):.2f}",
        "+1 bar gecikme Sharpe": f"{stats.sharpe(delayed):.2f}",
        **{f"IC[{m}]": f"{pred_m[m].get('ic_mean', float('nan')):.3f} (t={pred_m[m].get('ic_t', float('nan')):.1f})" for m in pred_m},
    }
    render_report(ret_cols[primary], bench_ret, out_dir, f"{cfg['id']} · {primary}", extra=extra)
    (out_dir / "pred_metrics.json").write_text(json.dumps(pred_m, indent=1, default=str), encoding="utf-8")

    registered = False
    if not smoke:
        decision = "çerçeve çıktısı (karar yok)" if report.valid else f"GEÇERSİZ: {', '.join(report.failures)}"
        info = register_experiment(
            cfg["id"], cfg, pd.concat(ret_cols, axis=1) if len(ret_cols) > 1 else ret_cols[primary], {k: v for k, v in metrics.items() if k != "deflated_sharpe_if_registered"},
            n_variants=len(cfg["models"]), hypothesis=cfg.get("hypothesis", cfg["target"]["name"]), decision=decision, question=cfg["question"],
            results_dir=results_dir, log_path=log_path, registry_file=registry_file, data_hash=panel.data_hash,
        )
        registered = info["registered"]
    return ExperimentResult(cfg["id"], report.valid, smoke, metrics, report, out_dir, oof_path, registered, report.warnings)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Araştırma deney koşucusu")
    ap.add_argument("config")
    ap.add_argument("--smoke", action="store_true", help="Deneme sayacına SAYILMAZ, kayıt yazmaz")
    args = ap.parse_args(argv)
    r = run_experiment(args.config, smoke=True if args.smoke else None)
    print(f"{r.id}: {'GEÇERLİ' if r.valid else 'GEÇERSİZ: ' + ', '.join(r.checks.failures)}"
          f"{' (smoke: sayaç artmadı)' if r.smoke else ''}  -> {r.out_dir}")
    for m, v in r.metrics["variants"].items():
        s = v["summary"]
        print(f"  {m:14s} Sharpe={s['sharpe']:.2f} yıllık={s['annual_return'] * 100:.1f}% maksDD={s['max_drawdown'] * 100:.1f}%  "
              f"IC={v['pred'].get('ic_mean', float('nan')):.3f} (t={v['pred'].get('ic_t', float('nan')):.1f}) QLIKE={v['pred'].get('qlike', float('nan')):.3f}")
    if r.warnings:
        print("  UYARI:", ", ".join(r.warnings))
    return 0 if r.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
