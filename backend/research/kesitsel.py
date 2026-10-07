"""Kesitsel momentum deneyi — ÖN KAYIT: docs/research/deneyler.md "ÖN KAYIT — kesitsel" (2026-10-07).

Bu modüldeki sabitler ön kayıtla BİREBİR aynıdır; sonuç görüldükten sonra değiştirilemez (KURALLAR §5, §7).

Kullanım:  cd backend && python -m research.kesitsel            (gerçek koşu: kayıt + sayaç)
Zaman çizelgesi: skor/ağırlık gün t kapanışında (t+1 00:00 UTC) hesaplanır; motor t+1 açılışında doldurur. Pencere, kıyaslar,
stres dönemleri ve walk-forward katmanları trend_001 ile aynıdır (`research.trend`)."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, engine, sizing, stats, trend
from .report import render_report
from .trend import EVAL_END, EVAL_START, FOLD_STARTS, POSITION_START, _fmt, daily_vol, window

# ---------------------------------------------------------------- ön kayıt sabitleri
QUESTION = "kesitsel"
EXPERIMENT_ID = "kesitsel_001"
HYPOTHESIS = ("Kripto'da kesitsel momentum (Liu, Tsyvinski & Wu 2022), maliyetler sonrası trend koluyla düşük korelasyonlu pozitif "
              "getiri üretir")
LOOKBACKS = (7, 14, 28)
SKIP_DAYS = 1
QUANTILE = 5
UNIVERSES = (30, 50)
SCORES = ("duz", "vol")
PORTFOLIOS = ("a", "b", "c")
REBALANCES = ("W", "3D")
BAND = 0.25
TARGET_VOL = 0.20
VOL_SPAN = 60
NAME_CAP = 0.10
GROSS_CAP = {"a": 2.0, "b": 2.0, "c": 1.0}
BETA_WINDOW = 60
BETA_CLIP = (0.0, 2.0)
STOP_LOSS = 0.25
MAIN = ("duz", "a")
PLATEAU_MULTS = (0.5, 1.5)
MIX_VOL_WINDOW = 60
CORR_MAX = 0.5
DIAG_N = 50
HORIZONS = tuple(range(1, 61))
TREND_SERIES = ("main_LF_wf", "main_LS_wf")
TREND_RETURNS = config.RESULTS_DIR / "trend_001" / "daily_returns.parquet"
N_PLANNED = (len(UNIVERSES) * len(SCORES) * len(PORTFOLIOS) * len(REBALANCES) + len(UNIVERSES) * len(REBALANCES)
             + 2 * (len(LOOKBACKS) + 1) + 1)  # 24 + 4 + 8 + 1 = 37
BTC = "BTCUSDT"
SQRT_YEAR = math.sqrt(config.ANNUALIZATION_DAYS)


# ---------------------------------------------------------------- skor (t kapanışı; son gün hariç)
def score(close: pd.DataFrame, membership: pd.DataFrame, vol_adjusted: bool, lookbacks=LOOKBACKS, span: int = VOL_SPAN) -> pd.DataFrame:
    """L ∈ lookbacks için r_L = log(close_{t-1} / close_{t-1-L}) (vol-ayarlıda / (σ·√L)); evren üyeleri arasında kesitsel z-skor;
    z-skorların ortalaması. Bir bileşen eksikse sembol o gün skorlanmaz (NaN)."""
    mem = membership.reindex_like(close).fillna(False).astype(bool)
    logc = np.log(close)
    sigma = daily_vol(close, span) if vol_adjusted else None
    zs = []
    for L in lookbacks:
        r = logc.shift(SKIP_DAYS) - logc.shift(SKIP_DAYS + L)
        if vol_adjusted:
            r = r / (sigma * math.sqrt(L))
        r = r.replace([np.inf, -np.inf], np.nan).where(mem)
        sd = r.std(axis=1, ddof=0)
        zs.append(r.sub(r.mean(axis=1), axis=0).div(sd.where(sd > 0), axis=0))
    return sum(zs) / len(zs)


def legs(sc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(en iyi dilim, en kötü dilim) boolean; k = max(1, ⌊n/5⌋), n = o gün skorlanmış üye sayısı (n < 2 ise dilim yok).
    Eşitlikte sembol adı sırası (kararlı)."""
    n = sc.notna().sum(axis=1)
    k = (n // QUANTILE).clip(lower=1).where(n >= 2, 0)
    top = sc.rank(axis=1, ascending=False, method="first").le(k, axis=0) & sc.notna()
    bottom = sc.rank(axis=1, ascending=True, method="first").le(k, axis=0) & sc.notna()
    return top, bottom


def _inv_vol(mask: pd.DataFrame, close: pd.DataFrame, span: int) -> pd.DataFrame:
    """Dilim içinde ters vol ağırlık, satır toplamı 1 (dilim boşsa 0)."""
    inv = (1.0 / (daily_vol(close, span) * SQRT_YEAR)).replace([np.inf, -np.inf], np.nan).where(mask)
    return inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def btc_beta(long_raw: pd.DataFrame, close: pd.DataFrame, window_days: int = BETA_WINDOW) -> pd.Series:
    """Long bacak ham portföyünün (t−1 ağırlıkları × t getirisi) BTC'ye 60 günlük OLS betası; yalnızca <= t; [0, 2]."""
    simple = close.pct_change(fill_method=None)
    port = (long_raw.shift(1).fillna(0.0) * simple.fillna(0.0)).sum(axis=1)
    b = simple[BTC]
    cov = port.rolling(window_days, min_periods=window_days).cov(b)
    var = b.rolling(window_days, min_periods=window_days).var()
    return (cov / var.where(var > 0)).clip(*BETA_CLIP)


def raw_portfolio(sc: pd.DataFrame, close: pd.DataFrame, portfolio: str, span: int = VOL_SPAN) -> pd.DataFrame:
    top, bottom = legs(sc)
    long_w = _inv_vol(top, close, span)
    if portfolio == "a":
        return long_w - _inv_vol(bottom, close, span)
    if portfolio == "c":
        return long_w
    if portfolio == "b":
        beta = btc_beta(long_w, close)
        out = long_w.copy()
        out[BTC] = out[BTC] - beta.fillna(0.0)
        return out.where(beta.notna(), 0.0)  # beta yokken (ısınma) pozisyon yok
    raise ValueError(f"portföy a|b|c olmalı: {portfolio}")


def size_weights(raw: pd.DataFrame, close: pd.DataFrame, portfolio: str, caps: pd.DataFrame, span: int = VOL_SPAN) -> pd.DataFrame:
    """Hedef vol ölçeği (ham portföyün geçmiş getirisi, EWMA span) -> isim tavanı (asset_caps; (b)'de BTC hedge muaf) -> brüt tavan."""
    simple = close.pct_change(fill_method=None).fillna(0.0)
    port = (raw.shift(1).fillna(0.0) * simple).sum(axis=1)
    port_vol = port.ewm(span=span, min_periods=span).std() * SQRT_YEAR
    scale = (TARGET_VOL / port_vol.clip(lower=1e-4)).where(port_vol.notna(), 0.0)
    w = raw.mul(scale, axis=0)
    c = caps.reindex_like(w).fillna(NAME_CAP)
    if portfolio == "b" and BTC in c.columns:
        c[BTC] = np.inf
    w = w.clip(lower=-c, upper=c)
    gross = w.abs().sum(axis=1)
    shrink = (GROSS_CAP[portfolio] / gross).clip(upper=1.0).where(gross > 0, 1.0)
    return w.mul(shrink, axis=0)


# ---------------------------------------------------------------- dengeleme ve zarar limiti
def rebalance_days(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """İşlem (açılış) günleri: haftalık = pazartesi; 3 günlük = POSITION_START'tan itibaren her 3. gün."""
    idx = index[index >= POSITION_START]
    if freq == "W":
        return idx[idx.dayofweek == 0]
    if freq == "3D":
        return idx[((idx - POSITION_START).days % 3) == 0]
    raise ValueError(f"dengeleme W|3D olmalı: {freq}")


def apply_stop(target: pd.DataFrame, open_px: pd.DataFrame, close: pd.DataFrame, rebal: pd.DatetimeIndex,
               limit: float = STOP_LOSS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Short isim zarar limiti. Dengeleme işlem günü R'de short hedeflenen (target[R−1] < 0) isimlerin referansı open[R].
    t ∈ [R, sonraki R) kapanışında close_t ≥ (1+limit)·ref olursa isim tetiklenir: t ve sonrasındaki karar günlerinde (dönem sonuna
    kadar) hedef 0, t+1'den itibaren işlem maskesi True (dengeleme dışı çıkış). Döner: (hedef, işlem maskesi; işlem günü indeksli)."""
    tgt = target.copy()
    mask = pd.DataFrame(False, index=target.index, columns=target.columns)
    idx = target.index
    rebal = pd.DatetimeIndex([r for r in rebal if r in idx])
    for k, r in enumerate(rebal):
        prev = r - pd.Timedelta(days=1)
        if prev not in idx:
            continue
        short = tgt.loc[prev] < 0
        if not short.any():
            continue
        end = rebal[k + 1] - pd.Timedelta(days=1) if k + 1 < len(rebal) else idx.max()
        cols = short.index[short]
        ref = open_px.loc[r, cols]
        hit = close.loc[r:end, cols].ge((1.0 + limit) * ref, axis=1).cummax()
        # son karar günü (end) bir sonraki dengelemenin kararıdır; orada yeni hedef geçerli
        hit_dec = hit.loc[: end - pd.Timedelta(days=1)] if k + 1 < len(rebal) else hit
        if hit_dec.empty:
            continue
        tgt.loc[hit_dec.index, cols] = tgt.loc[hit_dec.index, cols].where(~hit_dec, 0.0)
        trade_days = hit_dec.index + pd.Timedelta(days=1)
        m = hit_dec.set_axis(trade_days).reindex(mask.index.intersection(trade_days))
        mask.loc[m.index, cols] = mask.loc[m.index, cols] | m.to_numpy()
    return tgt, mask


# ---------------------------------------------------------------- motor
@dataclass
class Plan:
    """Motor girdisi: karar günü indeksli hedef, işlem günü indeksli dengeleme günleri ve (varsa) zarar limiti maskesi."""
    target: pd.DataFrame
    rebal: pd.DatetimeIndex
    mask: pd.DataFrame | None = None


def run_engine(m: trend.Market, plan: Plan, *, fee_mult: float = 1.0, slip_mult: float = 1.0, delay: int = 0,
               nav: float = config.ACCOUNT_NAV) -> engine.EngineResult:
    """`delay` > 0 (+1 gün senaryosu): hedef, dengeleme günleri ve zarar limiti maskesi BİRLİKTE `delay` gün kaydırılır -> aynı karar
    `delay` gün geç işlem görür (trend_001'deki `run_stitched` hizasıyla tutarlı)."""
    idx = m.prices["open"].loc[POSITION_START:EVAL_END].index
    rebal = plan.rebal + pd.Timedelta(days=delay) if delay else plan.rebal
    mask = plan.mask.shift(delay, freq="D").reindex(plan.mask.index, fill_value=False) if (delay and plan.mask is not None) else plan.mask
    return engine.run(
        plan.target.reindex(index=idx, columns=m.prices["open"].columns).fillna(0.0), m.prices["open"].loc[idx],
        fee_rate=config.FUTURES_TAKER_FEE * fee_mult, slippage_bps=m.slippage.loc[idx] * slip_mult,
        funding=m.funding.loc[idx] if m.funding is not None else None, band=BAND, band_relative=True,
        rebalance_days=rebal, trade_mask=mask, account_size=nav, delay_bars=delay,
        min_notional=m.min_notional, amount_step=m.amount_step, lot_rounding=config.ACCOUNT_LOT_ROUNDING,
        max_gross=config.ACCOUNT_LEVERAGE,
    )


def caps_for(m: trend.Market) -> pd.DataFrame:
    return sizing.asset_caps(m.prices["close"], config.ACCOUNT_NAV, m.min_notional, m.amount_step, base_cap=NAME_CAP)


def build_plan(m: trend.Market, caps: pd.DataFrame, score_kind: str, portfolio: str, n: int, freq: str, *, stop: bool = False,
               lookbacks=LOOKBACKS, span: int = VOL_SPAN) -> Plan:
    close = m.prices["close"]
    sc = score(close, m.memberships[n], score_kind == "vol", lookbacks, span)
    tgt = size_weights(raw_portfolio(sc, close, portfolio, span), close, portfolio, caps, span).astype("float64")
    rebal = rebalance_days(m.prices["open"].index, freq)
    if stop:
        tgt, mask = apply_stop(tgt, m.prices["open"], close, rebal)
        return Plan(tgt, rebal, mask)
    return Plan(tgt, rebal)


# ---------------------------------------------------------------- walk-forward (yalnızca N ve dengeleme)
COMBOS = tuple((n, f) for n in UNIVERSES for f in REBALANCES)


def select_combo(combo_returns: dict, fold_start: pd.Timestamp):
    """`fold_start`tan ÖNCEKİ (gün < fold_start − 1) net getirilerle en yüksek Sharpe'lı (N, dengeleme); eşitlikte ızgara sırası."""
    cutoff = fold_start - pd.Timedelta(days=1)
    best, best_sr = None, -np.inf
    for combo in COMBOS:
        r = combo_returns[combo]
        sr = stats.sharpe(r[(r.index >= POSITION_START) & (r.index < cutoff)])
        if sr == sr and sr > best_sr:
            best, best_sr = combo, sr
    return best or COMBOS[0]


def stitch(plans: dict, choices: list) -> Plan:
    """Katman k: karar günü d ∈ [F_k − 1, F_{k+1} − 1) için seçilen kombinasyonun hedefi; işlem günü d+1'in dengeleme bayrağı ve
    zarar limiti maskesi de aynı kombinasyondan. İlk katmandan önceki günler ilk seçimi kullanır (yalnızca ısınma)."""
    first = plans[choices[0][1]]
    tgt = first.target.copy()
    idx = tgt.index
    is_rebal = pd.Series(idx.isin(first.rebal), index=idx)
    any_mask = any(p.mask is not None for p in plans.values())
    mask = first.mask.copy() if first.mask is not None else (pd.DataFrame(False, index=idx, columns=tgt.columns) if any_mask else None)
    for k, (start, combo) in enumerate(choices):
        p = plans[combo]
        lo = start - pd.Timedelta(days=1)
        hi = choices[k + 1][0] - pd.Timedelta(days=1) if k + 1 < len(choices) else idx.max() + pd.Timedelta(days=1)
        dec = (idx >= lo) & (idx < hi)
        tgt.loc[dec] = p.target.loc[dec]
        trade = (idx >= lo + pd.Timedelta(days=1)) & (idx < hi + pd.Timedelta(days=1))
        is_rebal.loc[trade] = idx[trade].isin(p.rebal)
        if mask is not None and p.mask is not None:
            mask.loc[trade] = p.mask.loc[trade]
    return Plan(tgt, idx[is_rebal.to_numpy()], mask)


# ---------------------------------------------------------------- teşhis, karışım, korelasyon
def horizon_curve(m: trend.Market, n: int = DIAG_N, horizons=HORIZONS) -> pd.Series:
    """Ana skor (düz), N: her t için en iyi − en kötü dilimin (eşit ağırlık) t+1 açılışından h gün sonraki açılışa ortalama log getiri
    farkı; değerlendirme penceresindeki t'lerin ortalaması (brüt). Pencere sonunu aşan ileri getiriler kullanılmaz (NaN)."""
    close, logo = m.prices["close"], np.log(m.prices["open"])
    top, bottom = legs(score(close, m.memberships[n], False))
    out = {}
    for h in horizons:
        fwd = logo.shift(-(1 + h)) - logo.shift(-1)
        spread = fwd.where(top).mean(axis=1) - fwd.where(bottom).mean(axis=1)
        out[h] = float(spread.loc[EVAL_START:EVAL_END].mean())
    return pd.Series(out, name="spread")


def equal_risk_mix(a: pd.Series, b: pd.Series, window_days: int = MIX_VOL_WINDOW) -> pd.Series:
    """Her gün ağırlık, serilerin t−1'e kadarki `window_days` günlük vol'üyle ters orantılı (toplam 1); vol yokken eşit ağırlık."""
    df = pd.concat([a, b], axis=1).dropna()
    vol = df.rolling(window_days, min_periods=window_days).std().shift(1)
    inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    w = inv.div(inv.sum(axis=1), axis=0).fillna(0.5)
    return (w * df).sum(axis=1)


def load_trend_series(path: Path = TREND_RETURNS) -> pd.DataFrame:
    return pd.read_parquet(path)[list(TREND_SERIES)]


def correlations(r: pd.Series, trend_df: pd.DataFrame) -> dict[str, float]:
    return {c: float(pd.concat([window(r), window(trend_df[c])], axis=1).dropna().corr().iloc[0, 1]) for c in trend_df.columns}


# ---------------------------------------------------------------- ana akış
def skipped_reasons(res: engine.EngineResult) -> dict[str, int]:
    """Değerlendirme penceresinde atlanan emir denemeleri, nedenine göre (KURALLAR §3: min_lot, leverage, price, ...)."""
    out: dict[str, int] = {}
    for x in res.skipped:
        if EVAL_START <= pd.Timestamp(x["date"]) <= EVAL_END:
            out[x["reason"]] = out.get(x["reason"], 0) + 1
    return out


def _status(acc: dict, evaluate_plateau: bool) -> str:
    failed = [k for k, v in acc.items() if not v and (evaluate_plateau or k != "plateau")]
    if failed:
        return "KALDI (" + ", ".join(failed) + ")"
    return "GEÇTİ" if evaluate_plateau else "eşikler geçti, plato yok"


def _label(score_kind, portfolio, n=None, freq=None, stop=False) -> str:
    s = f"{score_kind}|{portfolio}" + ("|stop" if stop else "")
    return s + (f"|N{n}|{freq}" if n is not None else "")


def run_study(*, market: trend.Market | None = None, trend_returns: pd.DataFrame | None = None, results_dir: Path | None = None,
              log_path: Path | None = None, registry_file: Path | None = None, register: bool = True, progress=print) -> dict:
    from .registry import assert_budget, current_trial_count, register_experiment
    from .runner import recommended_nav_1x

    if register:  # bütçe dolacaksa ağır hesaba HİÇ başlama (asıl zorlama kayıtta, kilit altında)
        assert_budget(QUESTION, N_PLANNED, registry_file)
    m = market or trend.load_market(universes=(10, *UNIVERSES))
    trend_df = trend_returns if trend_returns is not None else load_trend_series()
    results_dir = Path(results_dir or config.RESULTS_DIR)
    registry_file = Path(registry_file or config.REGISTRY_FILE)
    out_dir = results_dir / EXPERIMENT_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    caps = caps_for(m)
    a = config.ACCEPTANCE

    bench = trend.benchmarks(m)
    btc_w = window(bench["BTC al-tut"].returns)
    btc_calmar = stats.calmar(btc_w)

    # 1) 24 temel varyant + 4 zarar limitli ana aday
    base_ret, base_rows, plans = {}, [], {}
    families = [(s, p, False) for s in SCORES for p in PORTFOLIOS] + [(*MAIN, True)]
    for s, p, stop in families:
        for n, f in COMBOS:
            plan = build_plan(m, caps, s, p, n, f, stop=stop)
            plans[(s, p, stop, n, f)] = plan
            res = run_engine(m, plan)
            base_ret[(s, p, stop, n, f)] = res.returns
            base_rows.append({"score": s, "portfolio": p, "stop": stop, "N": n, "rebalance": f, **trend.variant_metrics(res, btc_w),
                              "lot_scale": res.diagnostics["lot_scale"], "skipped_rate": trend.skipped_rate(res)})
        progress(f"temel varyantlar: {_label(s, p, stop=stop)} tamam ({len(base_ret)})")

    # 2) walk-forward (yalnızca N, dengeleme) + stres/gecikme
    wf = {}
    for s, p, stop in families:
        fam = {(n, f): base_ret[(s, p, stop, n, f)] for n, f in COMBOS}
        choices = [(fs, select_combo(fam, fs)) for fs in FOLD_STARTS]
        plan = stitch({(n, f): plans[(s, p, stop, n, f)] for n, f in COMBOS}, choices)
        res = run_engine(m, plan)
        wf[(s, p, stop)] = {
            "choices": choices, "plan": plan, "res": res,
            "stress": run_engine(m, plan, fee_mult=a["stress_fee_mult"], slip_mult=a["stress_slippage_mult"]).returns,
            "delay": run_engine(m, plan, delay=1).returns,
        }
        progress(f"walk-forward: {_label(s, p, stop=stop)} tamam")
    main_key = (*MAIN, False)
    main_r = wf[main_key]["res"].returns

    # 3) plato (ana aday, son walk-forward seçimi)
    n_last, f_last = wf[main_key]["choices"][-1][1]
    neigh = {}
    for i, L in enumerate(LOOKBACKS):
        for mult in PLATEAU_MULTS:
            lb = list(LOOKBACKS)
            lb[i] = max(1, int(round(L * mult)))
            neigh[f"L{L}×{mult}"] = run_engine(m, build_plan(m, caps, *MAIN, n_last, f_last, lookbacks=tuple(lb))).returns
    for mult in PLATEAU_MULTS:
        neigh[f"vol_span×{mult}"] = run_engine(m, build_plan(m, caps, *MAIN, n_last, f_last, span=int(round(VOL_SPAN * mult)))).returns
    base_sr = stats.sharpe(window(base_ret[(*MAIN, False, n_last, f_last)]))
    neigh_sr = {k: stats.sharpe(window(v)) for k, v in neigh.items()}
    plateau = {"N": n_last, "rebalance": f_last, "base_sharpe": base_sr, "neighbors": neigh_sr,
               "ratio": (min(neigh_sr.values()) / base_sr) if (base_sr == base_sr and base_sr > 0) else float("nan")}
    progress("plato tamam")

    # 4) trend ilişkisi: korelasyon + eşit risk karışımı (bilgi)
    corr = correlations(main_r, trend_df)
    mix = equal_risk_mix(window(main_r), window(trend_df["main_LF_wf"]))
    mix_summary = stats.summary(mix, btc_w)
    mix_acc = stats.check_acceptance(
        sharpe_net=mix_summary["sharpe"], deflated=float("nan"), pbo=float("nan"), max_dd=mix_summary["max_drawdown"],
        calmar_value=mix_summary["calmar"], btc_calmar=btc_calmar,
        positive_year_fraction=(sum(v > 0 for v in mix_summary["yearly_returns"].values()) / len(mix_summary["yearly_returns"])
                                if mix_summary["yearly_returns"] else float("nan")),
        stress_sharpe=float("nan"), plateau_ratio=float("nan"), is_portfolio=True)
    mix_info = {"sharpe": mix_summary["sharpe"], "max_dd": mix_summary["max_drawdown"], "calmar": mix_summary["calmar"],
                "annual_return": mix_summary["annual_return"],
                "sharpe_ok": mix_acc.get("net_sharpe"), "max_dd_ok": mix_acc.get("max_drawdown")}

    # 5) deneme sayımı, PBO, DSR
    base_only = {k: v for k, v in base_ret.items() if not k[2]}
    all_trials = list(base_ret.values()) + list(neigh.values()) + [mix]
    n_variants = len(all_trials)
    if n_variants != N_PLANNED:
        raise RuntimeError(f"varyant sayısı ön kayıttan farklı: {n_variants} != {N_PLANNED}")
    trials_after = current_trial_count(registry_file) + n_variants
    trial_sharpes = [x for x in (trend.daily_sharpe(r) for r in all_trials) if x == x]
    matrix = pd.concat({_label(k[0], k[1], k[3], k[4]): window(r) for k, r in base_only.items()}, axis=1).fillna(0.0)
    pbo = stats.pbo_cscv(matrix.to_numpy(), n_blocks=16)

    # 6) walk-forward tablosu + kabul
    wf_rows = []
    for key, w in wf.items():
        met = trend.variant_metrics(w["res"], btc_w)
        dsr = stats.deflated_sharpe(window(w["res"].returns), n_trials=trials_after, trial_sharpes=trial_sharpes)
        stress_sr = stats.sharpe(window(w["stress"]))
        is_main = key == main_key
        acc = trend.acceptance(met, dsr, pbo, stress_sr, btc_calmar, plateau["ratio"] if is_main else float("nan"))
        row = {"score": key[0], "portfolio": key[1], "stop": key[2], **met, "dsr": dsr, "stress_sharpe": stress_sr,
               "delay1_sharpe": stats.sharpe(window(w["delay"])), "lot_scale": w["res"].diagnostics["lot_scale"],
               "skipped_rate": trend.skipped_rate(w["res"]), "skipped_reasons": skipped_reasons(w["res"]), "selections": [f"{fs:%Y-%m}:N{c[0]}/{c[1]}" for fs, c in w["choices"]],
               "corr_trend": correlations(w["res"].returns, trend_df), "acceptance": acc, "status": _status(acc, is_main)}
        if key[1] == "a" or key[1] == "b":
            row["legs"] = trend.leg_contrib(w["res"])
        wf_rows.append(row)
    main_row = next(r for r in wf_rows if (r["score"], r["portfolio"], r["stop"]) == main_key)

    # 7) teşhis, stres, kıyaslar, hesap
    curve = horizon_curve(m)
    peak_h = int(curve.idxmax()) if curve.notna().any() else None
    _horizon_plot(curve, out_dir / "getiri_ufuk.png")
    stress = {"ana aday (WF)": trend.stress_table(window(main_r)), "ana aday + zarar limiti (WF)": trend.stress_table(window(wf[(*MAIN, True)]["res"].returns))}
    stress |= {k: trend.stress_table(window(v.returns)) for k, v in bench.items()}
    stress["BTC al-tut"]["2020-03"] = trend.btc_march_2020_stress(m)
    bench_rows = {k: trend.variant_metrics(v, btc_w) for k, v in bench.items()}
    last_close = m.prices["close"].loc[:EVAL_END].ffill().iloc[-1]
    univ_last = m.memberships[max(UNIVERSES)].loc[:EVAL_END].iloc[-1]
    nav_1x, nav_1x_sym = recommended_nav_1x(m.min_notional, m.amount_step, last_close[univ_last[univ_last].index])

    corr_ok = all(v == v and v < CORR_MAX for v in corr.values())
    if main_row["status"] == "GEÇTİ" and corr_ok:
        decision = f"KESİTSEL KOL ADAYI: ana aday tüm eşikleri geçti ve trend korelasyonu < {CORR_MAX}"
    else:
        reasons = [] if main_row["status"] == "GEÇTİ" else [main_row["status"].removeprefix("KALDI ").strip("()")]
        if not corr_ok:
            reasons.append("trend korelasyonu >= " + str(CORR_MAX))
        decision = "KALDI — ana aday: (" + "; ".join(reasons) + "). DUR (ızgara genişletilmez)."

    summary = {
        "experiment_id": EXPERIMENT_ID, "decision": decision, "pbo": pbo, "n_variants": n_variants, "trials_after": trials_after,
        "btc_calmar": btc_calmar, "plateau": plateau, "corr_trend": corr, "mix": mix_info, "horizon": curve.to_dict(),
        "horizon_peak": peak_h, "stress": stress, "benchmarks": bench_rows, "wf": wf_rows,
        "account": {"nav": config.ACCOUNT_NAV, "leverage": config.ACCOUNT_LEVERAGE, "lot_rounding": config.ACCOUNT_LOT_ROUNDING,
                    "recommended_nav_1x": nav_1x, "recommended_nav_1x_symbol": nav_1x_sym,
                    "recommended_nav_1x_price_date": str(EVAL_END.date())},
        "unlimited_symbols": m.unlimited_symbols, "data_hash": m.data_hash, "window": [str(EVAL_START.date()), str(EVAL_END.date())],
    }
    (out_dir / "ozet.json").write_text(json.dumps(summary, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(base_rows).to_csv(out_dir / "varyantlar.csv", index=False)
    (out_dir / "varyantlar.md").write_text(_base_table_md(base_rows, btc_calmar), encoding="utf-8")
    md = _results_md(summary)
    (out_dir / "sonuc.md").write_text(md, encoding="utf-8")
    render_report(window(main_r), btc_w, out_dir / "ana_aday", f"{EXPERIMENT_ID} · ana aday (düz skor, dolar-nötr; walk-forward)",
                  extra={"karar": main_row["status"], "PBO": _fmt(pbo), "DSR": _fmt(main_row["dsr"])})

    if register:
        frame = pd.concat([window(main_r).rename("main_wf"),
                           *[window(w["res"].returns).rename(_label(k[0], k[1], stop=k[2]) + "|wf") for k, w in wf.items() if k != main_key],
                           matrix,
                           *[window(base_ret[k]).rename(_label(k[0], k[1], k[3], k[4], stop=True)) for k in base_ret if k[2]],
                           *[window(v).rename(f"plato|{k}") for k, v in neigh.items()], mix.rename("karisim_trendLF")], axis=1)
        cfg = {"question": QUESTION, "hypothesis": HYPOTHESIS, "window": summary["window"], "position_start": str(POSITION_START.date()),
               "folds": [str(f.date()) for f in FOLD_STARTS], "lookbacks": list(LOOKBACKS), "skip_days": SKIP_DAYS,
               "quantile": QUANTILE, "universes": list(UNIVERSES), "scores": list(SCORES), "portfolios": list(PORTFOLIOS),
               "rebalances": list(REBALANCES), "band_relative": BAND, "target_vol": TARGET_VOL, "vol_span": VOL_SPAN,
               "name_cap": NAME_CAP, "gross_cap": GROSS_CAP, "beta_window": BETA_WINDOW, "stop_loss": STOP_LOSS,
               "account": summary["account"], "pre_registration": "deneyler.md ÖN KAYIT — kesitsel"}
        register_experiment(
            EXPERIMENT_ID, cfg, frame.astype("float32"),
            {"sharpe": main_row["sharpe"], "max_drawdown": main_row["max_dd"], "deflated_sharpe": main_row["dsr"], "pbo": pbo,
             "corr_trend": corr, "decision": decision},
            n_variants=n_variants, hypothesis=HYPOTHESIS, decision=decision, question=QUESTION, results_dir=results_dir,
            log_path=log_path, registry_file=registry_file, data_hash=m.data_hash,
        )
        log = Path(log_path or config.EXPERIMENT_LOG)
        log.write_text(log.read_text(encoding="utf-8").rstrip("\n") + "\n\n" + md, encoding="utf-8")
    return summary


def _horizon_plot(curve: pd.Series, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve.index, curve.to_numpy() * 100, marker=".")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"Getiri–ufuk: en iyi − en kötü dilim (düz skor, ilk-{DIAG_N}, brüt)")
    ax.set_xlabel("ufuk h (gün, t+1 açılışından)")
    ax.set_ylabel("ortalama kümülatif log getiri farkı (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _base_table_md(rows: list[dict], btc_calmar: float) -> str:
    lines = [f"# {EXPERIMENT_ID} — temel varyantlar (sabit N/dengeleme; pencere {EVAL_START.date()} → {EVAL_END.date()})", "",
             f"BTC al-tut Calmar: {_fmt(btc_calmar)}. Bilgi amaçlıdır; karar walk-forward ana adaya göre verilir (bkz. sonuc.md).", "",
             "| Skor | Portföy | Zarar lim. | N | Dengeleme | Sharpe | Calmar | Maks. DD | Yıllık | Turnover | Maliyet/Brüt | Poz. yıl | BTC korr. | lot_scale | Atlanan |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['score']} | {r['portfolio']} | {'evet' if r['stop'] else '—'} | {r['N']} | {r['rebalance']} | {_fmt(r['sharpe'])} | "
                     f"{_fmt(r['calmar'])} | {_fmt(r['max_dd'], True)} | {_fmt(r['annual_return'], True)} | {_fmt(r['turnover'], nd=1)} | "
                     f"{_fmt(r['cost_to_gross'], True)} | {_fmt(r['positive_years'], True)} | {_fmt(r['corr_btc'])} | "
                     f"{_fmt(r['lot_scale'])} | {_fmt(r['skipped_rate'], True)} |")
    return "\n".join(lines) + "\n"


def _results_md(s: dict) -> str:
    acc = s["account"]
    L = [f"## SONUÇ — `{EXPERIMENT_ID}` (ön kayıt: \"ÖN KAYIT — kesitsel\")", "",
         f"**Karar:** {s['decision']}", "",
         f"Pencere {s['window'][0]} → {s['window'][1]} · hesap {acc['nav']:g} USDT, {acc['leverage']:g}x, lot: {acc['lot_rounding']} · "
         f"deneme: {s['n_variants']} varyant (toplam sayaç {s['trials_after']}) · **PBO (24 temel varyant): {_fmt(s['pbo'])}** · "
         f"BTC al-tut Calmar {_fmt(s['btc_calmar'])}", "",
         "### Walk-forward serileri (7) — KURALLAR §5 kol eşikleri (karar yalnızca ana aday: düz | a)",
         "| Skor | Portföy | Zarar lim. | Sharpe | Calmar | Maks. DD | Yıllık | Turnover | Maliyet/Brüt | Poz. yıl | DSR | BTC korr. | "
         "Trend LF korr. | Trend LS korr. | Stres SR | +1g SR | Durum |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["wf"]:
        ct = r["corr_trend"]
        L.append(f"| {r['score']} | {r['portfolio']} | {'evet' if r['stop'] else '—'} | {_fmt(r['sharpe'])} | {_fmt(r['calmar'])} | "
                 f"{_fmt(r['max_dd'], True)} | {_fmt(r['annual_return'], True)} | {_fmt(r['turnover'], nd=1)} | "
                 f"{_fmt(r['cost_to_gross'], True)} | {_fmt(r['positive_years'], True)} | {_fmt(r['dsr'])} | {_fmt(r['corr_btc'])} | "
                 f"{_fmt(ct.get('main_LF_wf'))} | {_fmt(ct.get('main_LS_wf'))} | {_fmt(r['stress_sharpe'])} | {_fmt(r['delay1_sharpe'])} | "
                 f"{r['status']} |")
    L += ["", "### Kıyaslar", "| Kıyas | Sharpe | Calmar | Maks. DD | Yıllık | Turnover | BTC korr. |", "|---|---|---|---|---|---|---|"]
    for k, r in s["benchmarks"].items():
        L.append(f"| {k} | {_fmt(r['sharpe'])} | {_fmt(r['calmar'])} | {_fmt(r['max_dd'], True)} | {_fmt(r['annual_return'], True)} | "
                 f"{_fmt(r['turnover'], nd=1)} | {_fmt(r['corr_btc'])} |")
    p = s["plateau"]
    worst = min(p["neighbors"], key=p["neighbors"].get)
    main = next(r for r in s["wf"] if (r["score"], r["portfolio"], r["stop"]) == (*MAIN, False))
    stop = next(r for r in s["wf"] if (r["score"], r["portfolio"], r["stop"]) == (*MAIN, True))
    mix = s["mix"]
    L += ["", "### Ana aday ayrıntıları",
          f"- Walk-forward seçimleri: {', '.join(main['selections'])}.",
          f"- Plato (N{p['N']}/{p['rebalance']} sabit; temel Sharpe {_fmt(p['base_sharpe'])}): oran **{_fmt(p['ratio'])}**, en kötü komşu "
          f"`{worst}` ({_fmt(p['neighbors'][worst])}). Komşular: " + ", ".join(f"{k} {_fmt(v)}" for k, v in p["neighbors"].items()) + ".",
          f"- Bacak katkısı (toplam, NAV oranı): long {_fmt(main['legs']['long'], True)}, short {_fmt(main['legs']['short'], True)} "
          f"(toplam {_fmt(main['legs']['total'], True)}). Zarar limitli: long {_fmt(stop['legs']['long'], True)}, short "
          f"{_fmt(stop['legs']['short'], True)}.",
          f"- Hesap: atlanan emir denemesi {_fmt(main['skipped_rate'], True)} (nedenler: "
          f"{', '.join(f'{k} {v}' for k, v in main['skipped_reasons'].items()) or 'yok'}; zarar limitli: "
          f"{', '.join(f'{k} {v}' for k, v in stop['skipped_reasons'].items()) or 'yok'}), lot_scale {_fmt(main['lot_scale'])}. 1x için önerilen asgari "
          f"hesap: {_fmt(acc['recommended_nav_1x'], nd=0)} USDT ({acc['recommended_nav_1x_symbol']}, fiyat tarihi "
          f"{acc['recommended_nav_1x_price_date']}).",
          f"- Trend korelasyonu (ana aday): main_LF {_fmt(s['corr_trend'].get('main_LF_wf'))}, main_LS {_fmt(s['corr_trend'].get('main_LS_wf'))} "
          f"(eşik < {CORR_MAX}).",
          f"- Eşit risk karışımı (ana aday + trend main_LF; bilgi, portföy eşikleri): Sharpe {_fmt(mix['sharpe'])}, maks. DD "
          f"{_fmt(mix['max_dd'], True)}, Calmar {_fmt(mix['calmar'])}, yıllık {_fmt(mix['annual_return'], True)}.",
          f"- Getiri–ufuk (düz skor, ilk-{DIAG_N}, brüt): kümülatif fark h={s['horizon_peak']} günde tepe yapıyor "
          f"({_fmt(s['horizon'].get(s['horizon_peak']), True) if s['horizon_peak'] else '—'}); h=7: {_fmt(s['horizon'].get(7), True)}, "
          f"h=28: {_fmt(s['horizon'].get(28), True)}, h=60: {_fmt(s['horizon'].get(60), True)}. Grafik: `getiri_ufuk.png`."]
    L += ["", "### Stres dönemleri (maks. DD / toparlanma günü)", "| Seri | " + " | ".join(trend.STRESS_PERIODS) + " |",
          "|---|" + "---|" * len(trend.STRESS_PERIODS)]
    for k, per in s["stress"].items():
        cells = []
        for name in trend.STRESS_PERIODS:
            x = per.get(name, {})
            if x.get("max_dd") is None:
                cells.append("değerlendirilemez")
            else:
                rec = x["recovery_days"]
                cells.append(f"{x['max_dd']:.1%} / {rec if rec is not None else 'toparlanmadı'}")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["", f"Tam tablo: `research/results/{EXPERIMENT_ID}/varyantlar.md`. Limiti bilinmeyen (delist) semboller kısıtsız varsayıldı: "
          f"{len(s['unlimited_symbols'])} sembol.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    """Gerçek veride HER ZAMAN kayıtlı koşar (kaydedilmeyen deneme yasak, KURALLAR §7); `register=False` yalnızca testler içindir."""
    argparse.ArgumentParser(description="kesitsel momentum deneyi (ön kayıtlı; kayıt + deneme sayacı)").parse_args(argv)
    s = run_study(register=True)
    print(s["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
