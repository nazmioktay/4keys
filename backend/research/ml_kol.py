"""ML kolu deneyi — ÖN KAYIT: docs/research/deneyler.md "ÖN KAYIT — ml_kol" (2026-10-08).

Bu modüldeki sabitler ön kayıtla BİREBİR aynıdır; sonuç görüldükten sonra değiştirilemez (KURALLAR §5, §7).

Kullanım:  cd backend && python -m research.ml_kol            (gerçek koşu: 7 kayıtlı config, 18 varyant)
Her config `research.runner` ile KAYITLI koşar (purge+embargo walk-forward, sızıntı testleri, 250 USDT/3x motor). Bu modül
config'leri üretir, varyant serilerini OOF tahminlerinden aynı adaptör/motorla yeniden kurar (stres/gecikme dahil), plato,
ridge'e karşı eşleştirilmiş bootstrap ve karar kuralını uygular."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from . import config, guard, pred_metrics, signals, stats, trend
from .ablation import _paired_ci
from .models.lgbm import _COMMON as LGB_DEFAULTS
from .panel import build_panel
from .runner import _benchmark, _portfolio, _trim, adapter_context, load_config, run_experiment

# ---------------------------------------------------------------- ön kayıt sabitleri
QUESTION = "ml_kol"
SUMMARY_ID = "ml_kol"
HYPOTHESIS = ("Vol'e göre ölçeklenmiş 7 günlük ileri getiriyi tahmin eden model, yalnızca beklenen getirinin maliyeti aştığı sembollerde "
              "pozisyon alarak maliyetler sonrası kol eşiklerini geçen getiri üretir")
UNIVERSE_N = 30
PERIOD = ("2021-01-01", "2025-09-30")
N_SPLITS = 5
MIN_TRAIN_DAYS = 365
SEED = 0
TIERS = {"a": ["ohlcv_core"], "b": ["ohlcv_core", "funding", "taker_flow"],
         "d": ["ohlcv_core", "funding", "taker_flow", "macro", "sentiment_fng"]}
MODELS = ("ridge", "lightgbm_reg", "xgboost_reg")
H_MAIN, H_EXTRA = 7, 3
SIGNAL = dict(k=1.0, target_vol_annual=0.20, vol_window=20, vol_span=60, base_cap=0.10, gross_cap=2.0)
BAND = 0.25
MAIN = ("lightgbm_reg", "b", H_MAIN)
PLATEAU_MULTS = (0.5, 1.5)
N_PLANNED = 18
N_BOOT = 1000
MEAN_BLOCK = 15.0
SIMPLICITY = {"model": {m: i for i, m in enumerate(MODELS)}, "tier": {"a": 0, "b": 1, "d": 2}, "h": {H_MAIN: 0, H_EXTRA: 1}}


def _label(model: str, tier: str, h: int) -> str:
    return f"{model}|{tier}|h{h}"


def experiments() -> list[dict]:
    """7 config: 4 temel (her biri 3 model) + 2 plato-k (signal parametresi farklı) + 1 plato-lightgbm (4 model). Toplam 18 varyant."""
    out = []
    for tier, h in (("a", H_MAIN), ("b", H_MAIN), ("d", H_MAIN), ("b", H_EXTRA)):
        out.append({"id": f"ml_kol_{tier}_h{h}", "tier": tier, "h": h, "signal": {},
                    "models": [{"id": _label(m, tier, h), "name": m} for m in MODELS], "kind": "base"})
    for mult in PLATEAU_MULTS:
        out.append({"id": f"ml_kol_plato_k{mult}", "tier": MAIN[1], "h": MAIN[2], "signal": {"k": SIGNAL["k"] * mult},
                    "models": [{"id": f"plato|k×{mult}", "name": MAIN[0]}], "kind": "plateau"})
    lgbm = []
    for param in ("num_leaves", "learning_rate"):
        for mult in PLATEAU_MULTS:
            v = LGB_DEFAULTS[param] * mult
            v = int(round(v)) if param == "num_leaves" else v
            lgbm.append({"id": f"plato|{param}×{mult}", "name": MAIN[0], "params": {param: v}})
    out.append({"id": "ml_kol_plato_lgbm", "tier": MAIN[1], "h": MAIN[2], "signal": {}, "models": lgbm, "kind": "plateau"})
    assert sum(len(e["models"]) for e in out) == N_PLANNED
    return out


def make_config(exp: dict) -> dict:
    h = exp["h"]
    return {
        "id": exp["id"], "question": QUESTION, "hypothesis": HYPOTHESIS, "sources": TIERS[exp["tier"]], "universe": {"n": UNIVERSE_N},
        "period": {"start": PERIOD[0], "end": PERIOD[1]}, "target": {"name": "vol_adj_return", "h": h}, "models": exp["models"],
        "cv": {"n_splits": N_SPLITS, "embargo_days": h, "min_train_days": MIN_TRAIN_DAYS},
        "signal": {"adapter": "edge_threshold", "params": {"h": h, **SIGNAL, **exp["signal"]}}, "execution": {"band": BAND},
        "seed": SEED, "arm": "arm",
    }


MACRO_SERIES = ("vix", "gold", "sp500", "nasdaq", "nikkei", "dax")
MIN_FEATURE_COVERAGE = 0.5


class TierDataError(RuntimeError):
    """Bir kaynak kademesinin beklenen özellikleri yok ya da çok eksik: deney kayda başlamadan durur (sessizce b'ye dönüşmez)."""


def check_tier_features(panels: dict) -> None:
    """Kademe d panelinde 6 makro serinin ve duygu endeksinin özellikleri var ve satırların ≥ %50'sinde dolu olmalı."""
    for (tier, _h), panel in panels.items():
        if tier != "d" or "macro" not in TIERS["d"]:
            continue
        X = panel.X
        need = [f"macro__{s}_z" for s in MACRO_SERIES] + ["sentiment_fng__fng"]
        missing = [c for c in need if c not in X.columns]
        if missing:
            raise TierDataError(f"kademe d: eksik özellik kolonları {missing} (ağ/yfinance/alternative.me?) — kayıt yapılmadı")
        low = {c: round(float(X[c].notna().mean()), 3) for c in need if X[c].notna().mean() < MIN_FEATURE_COVERAGE}
        if low:
            raise TierDataError(f"kademe d: düşük doluluk {low} (< {MIN_FEATURE_COVERAGE}) — kayıt yapılmadı")


# ---------------------------------------------------------------- varyant serileri (OOF -> adaptör -> motor)
def variant_series(panel, cfg: dict, pred: pd.Series) -> dict:
    a = config.ACCEPTANCE
    adapter, params, band = cfg["signal"]["adapter"], cfg["signal"]["params"], cfg["execution"]["band"]
    ctx = adapter_context(panel, cfg["account"])
    member = panel.membership.reindex(columns=panel.candidates).fillna(False)
    w = signals.make_weights(adapter, pred.unstack("symbol").reindex(columns=panel.candidates), member, ctx, **params)
    first = pred.dropna().index.get_level_values("date").min()
    res = _portfolio(panel, w, account=cfg["account"], band=band)
    return {
        "returns": _trim(res.returns, first),
        "stress": _trim(_portfolio(panel, w, a["stress_fee_mult"], a["stress_slippage_mult"], account=cfg["account"], band=band).returns, first),
        "delay": _trim(_portfolio(panel, w, delay=1, account=cfg["account"], band=band).returns, first),
        "turnover": float(res.turnover.loc[first:].mean() * config.ANNUALIZATION_DAYS),
        "cost_to_gross": _cost_to_gross(res, first), "first": first,
    }


def _cost_to_gross(res, first) -> float:
    cost = float(res.costs.loc[first:].sum().sum())
    gross = float(res.gross_returns.loc[first:].sum())
    return cost / gross if gross > 0 else float("nan")


def _sr(s: pd.Series) -> float:
    return stats.sharpe(s)


# ---------------------------------------------------------------- ana akış
def run_study(*, panels: dict | None = None, results_dir: Path | None = None, log_path: Path | None = None,
              registry_file: Path | None = None, configs_dir: Path | None = None, trend_returns: pd.DataFrame | None = None,
              progress=print) -> dict:
    from .kesitsel import load_trend_series
    from .registry import assert_budget, current_trial_count

    from .registry import is_registered

    results_dir = Path(results_dir or config.RESULTS_DIR)
    registry_file = Path(registry_file or config.REGISTRY_FILE)
    configs_dir = Path(configs_dir or (config.BACKEND_ROOT / "research" / "configs"))
    panels = dict(panels or {})
    exps = experiments()
    if (results_dir / SUMMARY_ID / "sonuc.md").exists():
        raise RuntimeError("ml_kol zaten tamamlandı (sonuç yazılmış); kayıtlar değiştirilmez")
    done = {e["id"] for e in exps if is_registered(e["id"], registry_file, results_dir)}
    assert_budget(QUESTION, sum(len(e["models"]) for e in exps if e["id"] not in done), registry_file)  # yetmiyorsa HİÇ başlama

    # Ağ/veri gerektiren her şey HİÇBİR kayıttan önce: paneller (makro/duygu ağdan), trend serileri, kademe d ön kontrolü.
    # Böylece bir ağ hatası yarım kalmış (sayacı tüketmiş ama kararsız) bir çalışma bırakmaz.
    for exp in exps:
        key = (exp["tier"], exp["h"])
        if key not in panels:
            panels[key] = build_panel(load_config(make_config(exp)))
            progress(f"panel hazır: kademe {exp['tier']}, h={exp['h']} ({panels[key].X.shape[1]} özellik)")
    check_tier_features(panels)
    trend_df = trend_returns if trend_returns is not None else load_trend_series()

    series, rows, ics, btc = {}, {}, {}, {}
    for exp in exps:
        cfg = load_config(make_config(exp))
        panel = panels[(exp["tier"], exp["h"])]
        if exp["id"] in done:  # devam modu: kayıtlı config yeniden koşmaz (kayıtlar değiştirilmez); OOF ve test sonucu diskten
            meta = json.loads((results_dir / exp["id"] / "metrics.json").read_text(encoding="utf-8"))
            if meta.get("data_snapshot_hash") not in (None, panel.data_hash):
                raise RuntimeError(f"{exp['id']}: veri anlık görüntüsü kayıttan beri değişti ({meta['data_snapshot_hash']} != "
                                   f"{panel.data_hash}); devam edilemez (karışık sonuç)")
            oof_path, ok, failures = results_dir / exp["id"] / "oof.parquet", bool(meta["valid"]), list(meta.get("check_failures", []))
            progress(f"config zaten kayıtlı, devam: {exp['id']}")
        else:
            r = run_experiment(make_config(exp), results_dir=results_dir, log_path=log_path, registry_file=registry_file, panel=panel)
            oof_path, ok, failures = r.oof_path, r.valid, list(r.checks.failures)
        oof = pd.read_parquet(oof_path)
        with guard.zone(cfg["zone"]):
            for m in cfg["models"]:
                lab = m["id"]
                pred = oof[lab]
                series[lab] = variant_series(panel, cfg, pred)
                _, ric = pred_metrics.daily_ic(pred, oof["y"])
                ics[lab] = ric
                rows[lab] = {"variant": lab, "experiment": exp["id"], "kind": exp["kind"], "model": m["name"], "tier": exp["tier"],
                             "h": exp["h"], "valid": ok, "check_failures": failures,
                             "rank_ic": float(ric.mean()) if len(ric) else float("nan"),
                             # örtüşen h günlük etiketler: Newey-West t (gecikme h)
                             "rank_ic_t": pred_metrics.newey_west_t(ric, lag=exp["h"])}
                first = series[lab]["first"]
                b = _benchmark(panel, first)
                btc[lab] = _trim(b.returns, first) if b is not None else None
        progress(f"config tamam: {exp['id']} ({len(series)}/{N_PLANNED})")

    if len(series) != N_PLANNED:
        raise RuntimeError(f"varyant sayısı ön kayıttan farklı: {len(series)} != {N_PLANNED}")
    base = [lab for lab, rw in rows.items() if rw["kind"] == "base"]
    trials_after = current_trial_count(registry_file)
    trial_sharpes = [x for x in (trend.daily_sharpe(s["returns"]) for s in series.values()) if x == x]
    matrix = pd.concat({lab: series[lab]["returns"] for lab in base}, axis=1).dropna()
    pbo = stats.pbo_cscv(matrix.to_numpy(), n_blocks=16) if len(matrix) >= 16 * 8 else float("nan")
    main_btc = btc.get(_label(*MAIN))
    btc_calmar = stats.calmar(main_btc) if main_btc is not None else float("nan")  # rapor başlığı (ana aday penceresi)

    # plato (ana aday: lightgbm_reg, b, h7)
    main_lab = _label(*MAIN)
    base_sr = _sr(series[main_lab]["returns"])
    neigh = {lab.removeprefix("plato|"): _sr(s["returns"]) for lab, s in series.items() if rows[lab]["kind"] == "plateau"}
    ratio = (min(neigh.values()) / base_sr) if (base_sr == base_sr and base_sr > 0 and all(v == v for v in neigh.values())) else float("nan")
    plateau = {"main": main_lab, "base_sharpe": base_sr, "neighbors": neigh, "ratio": ratio}

    table = []
    for lab in base:
        rw, s = rows[lab], series[lab]
        btc_ret = btc[lab]
        btc_cal = stats.calmar(btc_ret) if btc_ret is not None else float("nan")
        summ = stats.summary(s["returns"], btc_ret)
        years = summ["yearly_returns"]
        pos_years = (sum(v > 0 for v in years.values()) / len(years)) if years else float("nan")
        dsr = stats.deflated_sharpe(s["returns"], n_trials=trials_after, trial_sharpes=trial_sharpes)
        acc = stats.check_acceptance(sharpe_net=summ["sharpe"], deflated=dsr, pbo=pbo, max_dd=summ["max_drawdown"], calmar_value=summ["calmar"],
                                     btc_calmar=btc_cal, positive_year_fraction=pos_years, stress_sharpe=_sr(s["stress"]),
                                     plateau_ratio=ratio, is_portfolio=False)
        acc["leakage_checks"] = bool(rw["valid"])
        if rw["model"] != "ridge":
            ridge = series[_label("ridge", rw["tier"], rw["h"])]["returns"]
            d, lo, hi = _paired_ci(s["returns"], ridge, _sr, N_BOOT, MEAN_BLOCK, SEED)
            acc["beats_ridge"] = bool(lo == lo and lo > 0)
            vs_ridge = {"d_sharpe": d, "lo": lo, "hi": hi}
        else:
            vs_ridge = None
        corr = {c: float(pd.concat([s["returns"], trend_df[c]], axis=1).dropna().corr().iloc[0, 1]) for c in trend_df.columns}
        failed = [k for k, v in acc.items() if not v]
        table.append({**rw, "sharpe": summ["sharpe"], "calmar": summ["calmar"], "max_dd": summ["max_drawdown"],
                      "annual_return": summ["annual_return"], "positive_years": pos_years, "dsr": dsr, "stress_sharpe": _sr(s["stress"]),
                      "delay1_sharpe": _sr(s["delay"]), "turnover": s["turnover"], "cost_to_gross": s["cost_to_gross"],
                      "corr_btc": summ.get("corr_vs_btc", float("nan")), "corr_trend": corr, "vs_ridge": vs_ridge, "acceptance": acc,
                      "status": "GEÇTİ" if not failed else "KALDI (" + ", ".join(failed) + ")"})
    table.sort(key=lambda r: (SIMPLICITY["model"][r["model"]], SIMPLICITY["tier"][r["tier"]], SIMPLICITY["h"][r["h"]]))

    # kaynak kademelerinin marjinal katkısı (bilgi): h7, her model için a→b ve b→d
    tiers_tbl = []
    for m in MODELS:
        for lo_t, hi_t in (("a", "b"), ("b", "d")):
            x, y = _label(m, hi_t, H_MAIN), _label(m, lo_t, H_MAIN)
            ds = _paired_ci(series[x]["returns"], series[y]["returns"], _sr, N_BOOT, MEAN_BLOCK, SEED)
            di = _paired_ci(ics[x], ics[y], lambda z: float(z.mean()), N_BOOT, MEAN_BLOCK, SEED)
            tiers_tbl.append({"model": m, "step": f"{lo_t}→{hi_t}", "d_sharpe": ds, "d_rank_ic": di})

    passed = [r for r in table if r["status"] == "GEÇTİ"]
    if passed:
        win = passed[0]
        decision = f"ML KOLU ADAYI: {win['variant']} (geçen en basit varyant; geçen toplam {len(passed)})"
        exp = next(e for e in exps if e["id"] == win["experiment"])
        cfg_w = make_config({**exp, "id": "ml_kol_kazanan", "models": [m for m in exp["models"] if m["id"] == win["variant"]]})
        configs_dir.mkdir(parents=True, exist_ok=True)
        (configs_dir / "ml_kol_kazanan.yaml").write_text(yaml.safe_dump(cfg_w, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        decision = "KALDI — hiçbir varyant tüm koşulları geçmedi. DUR (ızgara genişletilmez)."

    summary = {"decision": decision, "pbo": pbo, "n_variants": N_PLANNED, "trials_after": trials_after, "btc_calmar": btc_calmar,
               "plateau": plateau, "table": table, "tiers": tiers_tbl, "period": list(PERIOD)}
    out_dir = results_dir / SUMMARY_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ozet.json").write_text(json.dumps(summary, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("acceptance", "corr_trend", "vs_ridge", "check_failures")} for r in table]).to_csv(
        out_dir / "varyantlar.csv", index=False)
    md = _results_md(summary)
    (out_dir / "sonuc.md").write_text(md, encoding="utf-8")
    log = Path(log_path or config.EXPERIMENT_LOG)
    log.write_text(log.read_text(encoding="utf-8").rstrip("\n") + "\n\n" + md, encoding="utf-8")
    return summary


def _f(x, pct=False, nd=2):
    return trend._fmt(x, pct, nd)


def _results_md(s: dict) -> str:
    L = ["## SONUÇ — `ml_kol` (ön kayıt: \"ÖN KAYIT — ml_kol\")", "", f"**Karar:** {s['decision']}", "",
         f"Dönem {s['period'][0]} → {s['period'][1]} (OOF) · deneme: {s['n_variants']} varyant (toplam sayaç {s['trials_after']}) · "
         f"**PBO (12 temel varyant): {_f(s['pbo'])}** · BTC al-tut Calmar {_f(s['btc_calmar'])}", "",
         "### Temel varyantlar (sadelik sırasıyla) — KURALLAR §5 kol eşikleri + sızıntı testleri + ridge'e karşı fark",
         "| Varyant | Rank IC (t) | Sharpe | Calmar | Maks. DD | Yıllık | Poz. yıl | Turnover | Maliyet/Brüt | DSR | Stres SR | +1g SR | "
         "Ridge'e göre ΔSharpe [%95] | BTC korr. | Trend LF/LS | Durum |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["table"]:
        vr = r["vs_ridge"]
        vrs = f"{_f(vr['d_sharpe'])} [{_f(vr['lo'])}, {_f(vr['hi'])}]" if vr else "— (kıyas)"
        ct = r["corr_trend"]
        L.append(f"| {r['variant']} | {_f(r['rank_ic'], nd=3)} ({_f(r['rank_ic_t'], nd=1)}) | {_f(r['sharpe'])} | {_f(r['calmar'])} | "
                 f"{_f(r['max_dd'], True)} | {_f(r['annual_return'], True)} | {_f(r['positive_years'], True)} | {_f(r['turnover'], nd=1)} | "
                 f"{_f(r['cost_to_gross'], True)} | {_f(r['dsr'])} | {_f(r['stress_sharpe'])} | {_f(r['delay1_sharpe'])} | {vrs} | "
                 f"{_f(r['corr_btc'])} | {_f(ct.get('main_LF_wf'))} / {_f(ct.get('main_LS_wf'))} | {r['status']} |")
    p = s["plateau"]
    L += ["", "### Plato", f"- Ana aday `{p['main']}` (Sharpe {_f(p['base_sharpe'])}): oran **{_f(p['ratio'])}**; komşular: "
          + ", ".join(f"{k} {_f(v)}" for k, v in p["neighbors"].items()) + ". Plato sonucu tüm varyantlara uygulanır.",
          "", "### Kaynak kademelerinin marjinal katkısı (h=7; eşleştirilmiş blok bootstrap %95; bilgi)",
          "| Model | Adım | ΔSharpe [%95] | ΔRank IC [%95] |", "|---|---|---|---|"]
    for t in s["tiers"]:
        ds, di = t["d_sharpe"], t["d_rank_ic"]
        L.append(f"| {t['model']} | {t['step']} | {_f(ds[0])} [{_f(ds[1])}, {_f(ds[2])}] | {_f(di[0], nd=4)} [{_f(di[1], nd=4)}, {_f(di[2], nd=4)}] |")
    bad = [r["variant"] for r in s["table"] if not r["valid"]]
    L += ["", f"Sızıntı testleri: {'hepsi geçti' if not bad else 'BAŞARISIZ: ' + ', '.join(bad)}. Varyant ayrıntıları: "
          "`research/results/ml_kol/varyantlar.csv`; her config'in kendi raporu `research/results/ml_kol_*/`.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    """Gerçek veride HER ZAMAN kayıtlı koşar (KURALLAR §7)."""
    argparse.ArgumentParser(description="ml_kol deneyi (ön kayıtlı; 7 kayıtlı config, 18 varyant)").parse_args(argv)
    s = run_study()
    print(s["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
