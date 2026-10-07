"""Zaman serisi momentumu (trend) deneyi — ÖN KAYIT: docs/research/deneyler.md "ÖN KAYIT — trend" (2026-10-07).

Bu modüldeki sabitler ön kayıtla BİREBİR aynıdır; sonuç görüldükten sonra değiştirilemez (KURALLAR §5, §7).

Kullanım:  cd backend && python -m research.trend            (gerçek koşu: kayıt + sayaç)
Zaman çizelgesi: sinyal/ağırlık gün t kapanışında (t+1 00:00 UTC) hesaplanır; motor t+1 açılışında doldurur."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, engine, stats
from .data import store
from .data.universe import universe_membership
from .report import render_report

# ---------------------------------------------------------------- ön kayıt sabitleri
QUESTION = "trend"
EXPERIMENT_ID = "trend_001"
HYPOTHESIS = ("Kripto zaman serisi momentumu, çok varlıklı ve vol'e göre boyutlandırılmış olarak, maliyetler sonrası BTC al-tut'tan "
              "daha yüksek Calmar üretir")
POSITION_START = pd.Timestamp("2020-10-01")
EVAL_START = pd.Timestamp("2021-04-01")
EVAL_END = pd.Timestamp("2025-09-30")
FOLD_STARTS = tuple(pd.date_range("2021-04-01", "2025-04-01", freq="6MS"))
A_LOOKBACKS = (20, 60, 120, 250)
B_LOOKBACKS = (20, 60, 120, 250)
EMA_PAIRS = ((8, 32), (16, 64), (32, 128))
DONCHIAN = ((20, 10), (55, 20))
UNIVERSES = (10, 20, 30)
BANDS = (0.0, 0.10, 0.25)
DIRECTIONS = ("LF", "LS")
TARGET_VOL = 0.25
VOL_SPAN = 60
ASSET_CAP = 0.20
GROSS_CAP = {"LF": 1.0, "LS": 1.5}
NAV_PRIMARY = 10_000.0
NAV_SMALL = 1_000.0
PLATEAU_MULTS = (0.5, 1.5)
STRESS_PERIODS = {"2020-03": ("2020-03-01", "2020-03-31"), "2021-05": ("2021-05-01", "2021-05-31"),
                  "2022-05 (LUNA)": ("2022-05-01", "2022-05-31"), "2022-11 (FTX)": ("2022-11-01", "2022-11-30")}
SQRT_YEAR = math.sqrt(config.ANNUALIZATION_DAYS)


# ---------------------------------------------------------------- sinyaller (t kapanışı; yalnızca <= t verisi)
def daily_vol(close: pd.DataFrame, span: int = VOL_SPAN) -> pd.DataFrame:
    """Günlük log getirinin EWMA std'si (span), yıllıklaştırılmamış."""
    return np.log(close).diff().ewm(span=span, min_periods=span).std()


def sig_a(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return np.sign(close / close.shift(lookback) - 1.0)


def sig_b(close: pd.DataFrame, lookback: int, span: int = VOL_SPAN) -> pd.DataFrame:
    r = np.log(close / close.shift(lookback))
    return (r / (daily_vol(close, span) * math.sqrt(lookback))).clip(-2.0, 2.0) / 2.0


def sig_ema(close: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    ema_f = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
    ema_s = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    return np.sign(ema_f - ema_s).where(ema_s.notna())


def blend(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Eşit ağırlıklı ortalama; geçmişi yetmeyen (NaN) bileşen ortalamaya girmez; hepsi NaN ise NaN."""
    stacked = np.stack([f.to_numpy(dtype=float) for f in frames])
    with warnings.catch_warnings():  # ısınmada tüm bileşenler NaN -> NaN (beklenen)
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(stacked, axis=0) if stacked.size else stacked
    return pd.DataFrame(out, index=frames[0].index, columns=frames[0].columns)


def sig_c_mix(close: pd.DataFrame, pairs=EMA_PAIRS) -> pd.DataFrame:
    return blend([sig_ema(close, f, s) for f, s in pairs])


def sig_donchian(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n_in: int, n_out: int) -> pd.DataFrame:
    """Donchian durum makinesi: kapanış önceki n_in günün en yükseğinin üstünde -> long, en düşüğünün altında -> short;
    long, önceki n_out günün en düşüğünün altında kapanınca kapanır (short simetrik). Fiyat yoksa pozisyon 0."""
    up_in = high.shift(1).rolling(n_in, min_periods=n_in).max().to_numpy()
    lo_in = low.shift(1).rolling(n_in, min_periods=n_in).min().to_numpy()
    lo_out = low.shift(1).rolling(n_out, min_periods=n_out).min().to_numpy()
    up_out = high.shift(1).rolling(n_out, min_periods=n_out).max().to_numpy()
    c = close.to_numpy(dtype=float)
    out = np.full(c.shape, np.nan)
    pos = np.zeros(c.shape[1])
    for t in range(c.shape[0]):
        ct = c[t]
        valid = np.isfinite(ct) & np.isfinite(up_in[t]) & np.isfinite(lo_in[t])
        exit_long = (pos == 1) & (ct < lo_out[t])
        exit_short = (pos == -1) & (ct > up_out[t])
        pos = np.where(exit_long | exit_short, 0.0, pos)
        flat = pos == 0
        pos = np.where(flat & (ct > up_in[t]), 1.0, pos)
        pos = np.where(flat & (ct < lo_in[t]), -1.0, pos)
        pos = np.where(valid, pos, 0.0)
        out[t] = np.where(valid, pos, np.nan)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def main_components(close: pd.DataFrame, a_lookbacks=A_LOOKBACKS, ema_pairs=EMA_PAIRS) -> list[pd.DataFrame]:
    return [sig_a(close, L) for L in a_lookbacks] + [sig_ema(close, f, s) for f, s in ema_pairs]


def all_signals(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Ön kayıttaki 12 sinyal."""
    close, high, low = prices["close"], prices["high"], prices["low"]
    out = {f"a{L}": sig_a(close, L) for L in A_LOOKBACKS}
    out |= {f"b{L}": sig_b(close, L) for L in B_LOOKBACKS}
    out["c_mix"] = sig_c_mix(close)
    out |= {f"d{i}_{o}": sig_donchian(high, low, close, i, o) for i, o in DONCHIAN}
    out["main"] = blend(main_components(close))
    return out


# ---------------------------------------------------------------- boyutlandırma
def size_weights(signal: pd.DataFrame, close: pd.DataFrame, membership: pd.DataFrame, direction: str,
                 span: int = VOL_SPAN, target_vol: float = TARGET_VOL) -> pd.DataFrame:
    """Ters vol × sinyal -> kol hedef vol ölçeği -> varlık tavanı -> brüt tavan. Yalnızca t'ye kadar veri."""
    s = signal.where(membership.reindex_like(signal).fillna(False).astype(bool))
    if direction == "LF":
        s = s.clip(lower=0.0)
    elif direction != "LS":
        raise ValueError(f"yön LF|LS olmalı: {direction}")
    sigma = daily_vol(close, span) * SQRT_YEAR
    raw = (s / sigma).where(sigma > 0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # ham portföyün geçmiş getirisi: raw[t-1] · getiri(t-1 kapanış -> t kapanış); ölçek t'de yalnızca <= t ile
    simple = close.pct_change(fill_method=None).fillna(0.0)
    port = (raw.shift(1).fillna(0.0) * simple).sum(axis=1)
    port_vol = port.ewm(span=span, min_periods=span).std() * SQRT_YEAR
    scale = (target_vol / port_vol.clip(lower=1e-4)).where(port_vol.notna(), 0.0)
    w = raw.mul(scale, axis=0).clip(-ASSET_CAP, ASSET_CAP)
    gross = w.abs().sum(axis=1)
    shrink = (GROSS_CAP[direction] / gross).clip(upper=1.0).where(gross > 0, 1.0)
    return w.mul(shrink, axis=0)


# ---------------------------------------------------------------- piyasa verisi ve motor
@dataclass
class Market:
    prices: dict[str, pd.DataFrame]  # open/high/low/close, tarih × aday sembol
    memberships: dict[int, pd.DataFrame]
    slippage: pd.DataFrame
    funding: pd.DataFrame
    min_notional: pd.Series | None
    amount_step: pd.Series | None
    unlimited_symbols: list[str]
    data_hash: str


def load_market(with_limits: bool = True, universes: tuple[int, ...] = UNIVERSES) -> Market:
    qv = store.load_panel("quote_volume")
    full_idx = pd.date_range(qv.index.min(), EVAL_END, freq="D")
    memberships = {n: universe_membership(qv, n).reindex(full_idx).fillna(False) for n in universes}
    window = memberships[max(universes)].loc[POSITION_START:EVAL_END]
    candidates = sorted(window.columns[window.any()])
    memberships = {n: m.reindex(columns=candidates).fillna(False) for n, m in memberships.items()}
    prices = {f: store.load_panel(f, candidates).reindex(full_idx) for f in ("open", "high", "low", "close")}
    top20 = universe_membership(qv, 20).reindex(index=full_idx, columns=candidates).fillna(False)
    slip = pd.DataFrame(np.where(top20.to_numpy(), config.SLIPPAGE_BPS_TOP20, config.SLIPPAGE_BPS_OTHER), index=full_idx, columns=candidates)
    for s in config.TOP_TIER_SYMBOLS:
        if s in slip.columns:
            slip[s] = config.SLIPPAGE_BPS_BTC_ETH
    rows = []
    for s in candidates:
        f = store.load_funding(s)
        if f is not None and len(f):
            rows.append(pd.DataFrame({"symbol": s, "time": f.index, "rate": f["rate"].to_numpy()}))
    funding = engine.daily_funding(pd.concat(rows, ignore_index=True), full_idx, candidates) if rows else None
    min_n = step = None
    unlimited: list[str] = []
    if with_limits:
        from .data.limits import engine_limits, get_limits

        min_n, step, unlimited = engine_limits(get_limits(candidates))
    return Market(prices, memberships, slip, funding, min_n, step, unlimited, store.snapshot_hash())


def run_engine(m: Market, target: pd.DataFrame, *, band: float = 0.0, fee_mult: float = 1.0, slip_mult: float = 1.0,
               delay: int = 0, nav: float = NAV_PRIMARY, funding: bool = True, rebalance_days=None) -> engine.EngineResult:
    idx = m.prices["open"].loc[POSITION_START:EVAL_END].index
    return engine.run(
        target.reindex(index=idx, columns=m.prices["open"].columns).fillna(0.0), m.prices["open"].loc[idx],
        fee_rate=config.FUTURES_TAKER_FEE * fee_mult, slippage_bps=m.slippage.loc[idx] * slip_mult,
        funding=m.funding.loc[idx] if (funding and m.funding is not None) else None, band=band, band_relative=True,
        account_size=nav, delay_bars=delay, min_notional=m.min_notional, amount_step=m.amount_step, rebalance_days=rebalance_days,
    )


def window(r: pd.Series) -> pd.Series:
    return r.loc[EVAL_START:EVAL_END]


# ---------------------------------------------------------------- walk-forward (yalnızca N ve bant)
def select_combo(combo_returns: dict[tuple[int, float], pd.Series], fold_start: pd.Timestamp) -> tuple[int, float]:
    """`fold_start`tan ÖNCEKİ (gün < fold_start − 1) net getirilerle en yüksek Sharpe'lı (N, bant); eşitlikte ızgara sırası."""
    cutoff = fold_start - pd.Timedelta(days=1)
    best, best_sr = None, -np.inf
    for combo in [(n, b) for n in UNIVERSES for b in BANDS]:
        r = combo_returns[combo]
        sr = stats.sharpe(r[(r.index >= POSITION_START) & (r.index < cutoff)])
        if sr == sr and sr > best_sr:
            best, best_sr = combo, sr
    return best or (UNIVERSES[0], BANDS[0])


def stitch_targets(targets: dict[tuple[int, float], pd.DataFrame], choices: list[tuple[pd.Timestamp, tuple[int, float]]]):
    """Katman k: karar günü d ∈ [F_k − 1, F_{k+1} − 1) için seçilen kombinasyonun hedefi. Bant da gün bazında taşınır.
    Döner: (hedef, gün bazında bant serisi). İlk katmandan önceki günler ilk seçimin hedefini kullanır (yalnızca ısınma)."""
    first = choices[0][1]
    out = targets[first].copy()
    bands = pd.Series(first[1], index=out.index)
    for k, (start, combo) in enumerate(choices):
        lo = start - pd.Timedelta(days=1)
        hi = choices[k + 1][0] - pd.Timedelta(days=1) if k + 1 < len(choices) else out.index.max() + pd.Timedelta(days=1)
        rows = (out.index >= lo) & (out.index < hi)
        out.loc[rows] = targets[combo].loc[rows]
        bands.loc[rows] = combo[1]
    return out, bands


def run_stitched(m: Market, target: pd.DataFrame, bands: pd.Series, **kw) -> engine.EngineResult:
    """Birleştirilmiş hedef TEK motor koşusunda; bant gün bazında katmanın seçimi (işlem günü = karar günü + 1)."""
    return run_engine(m, target, band=bands.shift(1 + kw.get("delay", 0)).bfill(), **kw)  # +1 gün senaryosunda da hizalı


# ---------------------------------------------------------------- metrikler
def variant_metrics(res: engine.EngineResult, btc_w: pd.Series | None) -> dict:
    """Değerlendirme penceresinde: Sharpe, Calmar, maks. DD, yıllık getiri, turnover (yıllık), maliyet/brüt kâr, pozitif yıl,
    BTC korelasyonu."""
    r = window(res.returns)
    s = stats.summary(r, btc_w)
    sl = slice(EVAL_START, EVAL_END)
    cost = float(res.costs.loc[sl].sum().sum())
    gross = float(res.gross_returns.loc[sl].sum())
    years = s["yearly_returns"]
    return {
        "sharpe": s["sharpe"], "calmar": s["calmar"], "max_dd": s["max_drawdown"], "annual_return": s["annual_return"],
        "turnover": float(res.turnover.loc[sl].mean() * config.ANNUALIZATION_DAYS),
        "cost_to_gross": cost / gross if gross > 0 else float("nan"),
        "positive_years": (sum(v > 0 for v in years.values()) / len(years)) if years else float("nan"),
        "corr_btc": s.get("corr_vs_btc", float("nan")),
    }


def daily_sharpe(r: pd.Series) -> float:
    r = window(r).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def stress_table(returns: pd.Series) -> dict[str, dict]:
    """Her stres dönemi: dönem içindeki en derin DD (öncesindeki tepe dahil) ve dibinden o tepeye toparlanma süresi."""
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    peak = nav.cummax()
    out = {}
    for name, (a, b) in STRESS_PERIODS.items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if len(returns) == 0 or returns.index.min() > a:
            out[name] = {"max_dd": None, "recovery_days": None, "note": "değerlendirilemez (seri dönemden sonra başlıyor)"}
            continue
        dd = (nav / peak - 1.0).loc[a:b]
        trough = dd.idxmin()
        level = float(peak.loc[trough])
        after = nav.loc[trough:]
        rec = after[after >= level]
        out[name] = {"max_dd": float(dd.min()), "recovery_days": (int((rec.index[0] - trough).days) if len(rec) else None),
                     "note": "" if len(rec) else f"{EVAL_END.date()}'e kadar toparlanmadı"}
    return out


def leg_contrib(res: engine.EngineResult) -> dict:
    """Long ve short bacağın değerlendirme penceresindeki toplam net katkısı (NAV oranı, basit toplam). Bacak, günün pozisyon
    işaretiyle; pozisyon kapatılan günde (tutulan 0) bir önceki günün işaretiyle atanır -> çıkış maliyeti de kendi bacağına
    yazılır ve long + short = toplam katkı."""
    held = res.weights_held
    side = np.sign(held).where(held != 0, np.sign(held.shift(1)).fillna(0.0))
    sl = slice(EVAL_START, EVAL_END)
    c, side = res.symbol_contrib.loc[sl], side.loc[sl]
    return {"long": float(c.where(side > 0, 0.0).sum().sum()), "short": float(c.where(side < 0, 0.0).sum().sum()),
            "total": float(c.sum().sum())}


def skipped_rate(res: engine.EngineResult) -> float:
    """Emir denemelerinin atlanan oranı: asgari emir altında kalan (atlanan) / (atlanan + gerçekten yapılan emir). Aynı pozisyon
    her gün yeniden denenip atlanırsa her deneme sayılır (payda ve pay aynı birimde: emir denemesi)."""
    sl = slice(EVAL_START, EVAL_END)
    skipped = sum(1 for x in res.skipped if EVAL_START <= pd.Timestamp(x["date"]) <= EVAL_END)
    executed = int(res.diagnostics["trades_per_day"].loc[sl].sum())
    return skipped / (skipped + executed) if (skipped + executed) else float("nan")


def acceptance(m: dict, dsr: float, pbo: float, stress_sharpe: float, btc_calmar: float, plateau: float) -> dict:
    return stats.check_acceptance(
        sharpe_net=m["sharpe"], deflated=dsr, pbo=pbo, max_dd=m["max_dd"], calmar_value=m["calmar"], btc_calmar=btc_calmar,
        positive_year_fraction=m["positive_years"], stress_sharpe=stress_sharpe, plateau_ratio=plateau, is_portfolio=False,
    )


# ---------------------------------------------------------------- kıyaslar
def benchmarks(m: Market) -> dict[str, engine.EngineResult]:
    close, idx = m.prices["close"], m.prices["open"].index
    btc = pd.DataFrame(0.0, index=idx, columns=m.prices["open"].columns)
    btc["BTCUSDT"] = 1.0
    ew = m.memberships[10].astype(float)
    ew = ew.div(ew.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    month_starts = pd.date_range(POSITION_START, EVAL_END, freq="MS")
    ema200 = close["BTCUSDT"].ewm(span=200, adjust=False, min_periods=200).mean()
    trend = btc.copy()
    trend["BTCUSDT"] = (close["BTCUSDT"] > ema200).astype(float).where(ema200.notna(), 0.0)
    return {
        "BTC al-tut": run_engine(m, btc, funding=False),
        "EW ilk-10 al-tut (aylık)": run_engine(m, ew, funding=False, rebalance_days=month_starts),
        "BTC EMA200 long/flat": run_engine(m, trend),
    }


def btc_march_2020_stress(m: Market) -> dict:
    """2020-03 yalnızca BTC al-tut için (ham açılıştan açılışa; strateji serileri bu dönemde yok)."""
    px = m.prices["open"]["BTCUSDT"].loc[:EVAL_END]
    r = (px.shift(-1) / px - 1.0).dropna()
    return stress_table(r)["2020-03"]


# ---------------------------------------------------------------- plato
def plateau_variants(m: Market, direction: str, n: int, band: float) -> dict[str, pd.Series]:
    close = m.prices["close"]
    out = {}
    for k in range(len(A_LOOKBACKS) + len(EMA_PAIRS)):
        for mult in PLATEAU_MULTS:
            a = list(A_LOOKBACKS)
            e = list(EMA_PAIRS)
            if k < len(a):
                a[k] = max(2, int(round(a[k] * mult)))
                label = f"a{A_LOOKBACKS[k]}×{mult}"
            else:
                f, s = e[k - len(a)]
                e[k - len(a)] = (max(2, int(round(f * mult))), max(3, int(round(s * mult))))
                label = f"ema{f}/{s}×{mult}"
            tgt = size_weights(blend(main_components(close, a, e)), close, m.memberships[n], direction)
            out[label] = run_engine(m, tgt, band=band).returns
    sig = blend(main_components(close))
    for mult in PLATEAU_MULTS:
        span = int(round(VOL_SPAN * mult))
        out[f"vol_span×{mult}"] = run_engine(m, size_weights(sig, close, m.memberships[n], direction, span=span), band=band).returns
    return out


# ---------------------------------------------------------------- ana akış
def _fmt(x, pct=False, nd=2):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.1%}" if pct else f"{x:.{nd}f}"


def _status(acc: dict, evaluate_plateau: bool) -> str:
    failed = [k for k, v in acc.items() if not v and (evaluate_plateau or k != "plateau")]
    if failed:
        return "KALDI (" + ", ".join(failed) + ")"
    return "GEÇTİ" if evaluate_plateau else "eşikler geçti, plato yok"


def run_study(*, market: Market | None = None, results_dir: Path | None = None, log_path: Path | None = None,
              registry_file: Path | None = None, register: bool = True, progress=print) -> dict:
    from .registry import assert_budget, current_trial_count, register_experiment

    if register:  # bütçe dolacaksa ağır hesaba HİÇ başlama (asıl zorlama kayıtta, kilit altında)
        n_signals = len(A_LOOKBACKS) + len(B_LOOKBACKS) + 1 + len(DONCHIAN) + 1  # a, b, c_mix, Donchian, main
        n_plateau = len(DIRECTIONS) * len(PLATEAU_MULTS) * (len(A_LOOKBACKS) + len(EMA_PAIRS) + 1)
        n_planned = n_signals * len(DIRECTIONS) * len(UNIVERSES) * len(BANDS) + n_plateau
        assert_budget(QUESTION, n_planned, registry_file)
    m = market or load_market()
    results_dir = Path(results_dir or config.RESULTS_DIR)
    registry_file = Path(registry_file or config.REGISTRY_FILE)
    out_dir = results_dir / EXPERIMENT_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    close = m.prices["close"]
    sigs = all_signals(m.prices)

    bench = benchmarks(m)
    btc_w = window(bench["BTC al-tut"].returns)
    btc_calmar = stats.calmar(btc_w)

    # 1) 216 temel varyant
    base_ret, base_rows, targets = {}, [], {}
    for name, sig in sigs.items():
        for d in DIRECTIONS:
            for n in UNIVERSES:
                tgt = size_weights(sig, close, m.memberships[n], d).astype("float32")
                targets[(name, d, n)] = tgt
                for b in BANDS:
                    res = run_engine(m, tgt, band=b)
                    base_ret[(name, d, n, b)] = res.returns
                    base_rows.append({"signal": name, "dir": d, "N": n, "band": b, **variant_metrics(res, btc_w)})
        progress(f"temel varyantlar: {name} tamam ({len(base_ret)}/{len(sigs) * len(DIRECTIONS) * len(UNIVERSES) * len(BANDS)})")

    # 2) walk-forward (yalnızca N, bant) + stres/gecikme
    wf = {}
    for name in sigs:
        for d in DIRECTIONS:
            combos = {(n, b): base_ret[(name, d, n, b)] for n in UNIVERSES for b in BANDS}
            choices = [(f, select_combo(combos, f)) for f in FOLD_STARTS]
            tgt, bands = stitch_targets({(n, b): targets[(name, d, n)] for n in UNIVERSES for b in BANDS}, choices)
            a = config.ACCEPTANCE
            wf[(name, d)] = {
                "choices": choices, "target": tgt, "bands": bands, "res": run_stitched(m, tgt, bands),
                # bellek: stres/gecikme koşularından yalnızca getiri tutulur
                "stress": run_stitched(m, tgt, bands, fee_mult=a["stress_fee_mult"], slip_mult=a["stress_slippage_mult"]).returns,
                "delay": run_stitched(m, tgt, bands, delay=1).returns,
            }
        progress(f"walk-forward: {name} tamam")

    # 3) plato (ana aday, son walk-forward seçimi)
    plateau, plateau_ret = {}, {}
    for d in DIRECTIONS:
        n, b = wf[("main", d)]["choices"][-1][1]
        neigh = plateau_variants(m, d, n, b)
        base_sr = stats.sharpe(window(base_ret[("main", d, n, b)]))
        srs = {k: stats.sharpe(window(v)) for k, v in neigh.items()}
        plateau[d] = {"N": n, "band": b, "base_sharpe": base_sr, "neighbors": srs,
                      "ratio": (min(srs.values()) / base_sr) if (base_sr == base_sr and base_sr > 0) else float("nan")}
        plateau_ret |= {(d, k): v for k, v in neigh.items()}
    progress("plato tamam")

    # 4) deneme sayımı, PBO, DSR
    n_variants = len(base_ret) + len(plateau_ret)
    trials_after = current_trial_count(registry_file) + n_variants
    trial_sharpes = [daily_sharpe(r) for r in list(base_ret.values()) + list(plateau_ret.values())]
    trial_sharpes = [x for x in trial_sharpes if x == x]
    matrix = pd.concat({f"{k[0]}|{k[1]}|N{k[2]}|b{k[3]}": window(r) for k, r in base_ret.items()}, axis=1).fillna(0.0)
    pbo = stats.pbo_cscv(matrix.to_numpy(), n_blocks=16)

    # 5) walk-forward tablosu + kabul
    wf_rows = []
    for (name, d), w in wf.items():
        met = variant_metrics(w["res"], btc_w)
        dsr = stats.deflated_sharpe(window(w["res"].returns), n_trials=trials_after, trial_sharpes=trial_sharpes)
        stress_sr = stats.sharpe(window(w["stress"]))
        is_main = name == "main"
        acc = acceptance(met, dsr, pbo, stress_sr, btc_calmar, plateau[d]["ratio"] if is_main else float("nan"))
        wf_rows.append({"signal": name, "dir": d, **met, "dsr": dsr, "stress_sharpe": stress_sr,
                        "delay1_sharpe": stats.sharpe(window(w["delay"])),
                        "selections": [f"{f:%Y-%m}:N{c[0]}/b{c[1]:.2f}" for f, c in w["choices"]],
                        "acceptance": acc, "status": _status(acc, is_main),
                        **({"legs": leg_contrib(w["res"])} if d == "LS" else {})})

    # 6) ana aday ekleri: NAV 1.000, stres dönemleri, turnover–getiri
    small = {}
    for d in DIRECTIONS:
        w = wf[("main", d)]
        res_small = run_stitched(m, w["target"], w["bands"], nav=NAV_SMALL)
        small[d] = {"skipped_rate_1000": skipped_rate(res_small), "skipped_rate_10000": skipped_rate(w["res"]),
                    "annual_return_1000": stats.annual_return(window(res_small.returns)),
                    "annual_return_10000": stats.annual_return(window(w["res"].returns))}
    stress = {f"main {d} (WF)": stress_table(window(wf[("main", d)]["res"].returns)) for d in DIRECTIONS}
    stress |= {k: stress_table(window(v.returns)) for k, v in bench.items()}
    stress["BTC al-tut"]["2020-03"] = btc_march_2020_stress(m)
    bench_rows = {k: variant_metrics(v, btc_w) for k, v in bench.items()}
    _turnover_plot(base_rows, out_dir / "turnover_getiri.png")

    main_status = {d: next(r for r in wf_rows if r["signal"] == "main" and r["dir"] == d) for d in DIRECTIONS}
    passed = [d for d, r in main_status.items() if r["status"] == "GEÇTİ"]
    if passed:
        decision = f"TREND KOLU: ana aday {', '.join(passed)} tüm eşikleri geçti"
    else:
        decision = "KALDI — " + "; ".join(f"main {d}: {r['status'].removeprefix('KALDI ')}" for d, r in main_status.items()) + ". DUR (ızgara genişletilmez)."

    # 7) yaz
    summary = {
        "experiment_id": EXPERIMENT_ID, "decision": decision, "pbo": pbo, "n_variants": n_variants, "trials_after": trials_after,
        "btc_calmar": btc_calmar, "plateau": plateau, "nav_small": small, "stress": stress, "benchmarks": bench_rows,
        "wf": wf_rows, "unlimited_symbols": m.unlimited_symbols, "data_hash": m.data_hash,
        "window": [str(EVAL_START.date()), str(EVAL_END.date())],
    }
    (out_dir / "ozet.json").write_text(json.dumps(summary, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(base_rows).to_csv(out_dir / "varyantlar_216.csv", index=False)
    (out_dir / "varyantlar_216.md").write_text(_base_table_md(base_rows, btc_calmar), encoding="utf-8")
    md = _results_md(summary)
    (out_dir / "sonuc.md").write_text(md, encoding="utf-8")
    for d in DIRECTIONS:
        render_report(window(wf[("main", d)]["res"].returns), btc_w, out_dir / f"main_{d}", f"{EXPERIMENT_ID} · main {d} (walk-forward)",
                      extra={"karar": main_status[d]["status"], "PBO": _fmt(pbo), "DSR": _fmt(main_status[d]["dsr"])})

    if register:
        frame = pd.concat([window(wf[("main", "LF")]["res"].returns).rename("main_LF_wf"),
                           window(wf[("main", "LS")]["res"].returns).rename("main_LS_wf"),
                           *[window(w["res"].returns).rename(f"{k[0]}_{k[1]}_wf") for k, w in wf.items() if k[0] != "main"],
                           matrix], axis=1)
        cfg = {"question": QUESTION, "hypothesis": HYPOTHESIS, "window": summary["window"], "position_start": str(POSITION_START.date()),
               "folds": [str(f.date()) for f in FOLD_STARTS], "a": list(A_LOOKBACKS), "b": list(B_LOOKBACKS),
               "ema_pairs": [list(p) for p in EMA_PAIRS], "donchian": [list(p) for p in DONCHIAN], "universes": list(UNIVERSES),
               "bands_relative": list(BANDS), "target_vol": TARGET_VOL, "vol_span": VOL_SPAN, "asset_cap": ASSET_CAP,
               "gross_cap": GROSS_CAP, "nav": NAV_PRIMARY, "pre_registration": "deneyler.md ÖN KAYIT — trend"}
        mm = main_status["LF"]
        register_experiment(
            EXPERIMENT_ID, cfg, frame.astype("float32"),
            {"sharpe": mm["sharpe"], "max_drawdown": mm["max_dd"], "deflated_sharpe": mm["dsr"], "pbo": pbo, "decision": decision,
             "main_LS": {k: main_status["LS"][k] for k in ("sharpe", "max_dd", "dsr", "status")}},
            n_variants=n_variants, hypothesis=HYPOTHESIS, decision=decision, question=QUESTION, results_dir=results_dir,
            log_path=log_path, registry_file=registry_file, data_hash=m.data_hash,
        )
        log = Path(log_path or config.EXPERIMENT_LOG)
        log.write_text(log.read_text(encoding="utf-8").rstrip("\n") + "\n\n" + md, encoding="utf-8")
    return summary


def _turnover_plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, d in zip(axes, DIRECTIONS):
        for n, marker in zip(UNIVERSES, "osD"):
            pts = sorted([r for r in rows if r["signal"] == "main" and r["dir"] == d and r["N"] == n], key=lambda r: r["band"])
            ax.plot([r["turnover"] for r in pts], [r["annual_return"] for r in pts], marker=marker, label=f"ilk-{n}")
            for r in pts:
                ax.annotate(f"b{r['band']:.2f}", (r["turnover"], r["annual_return"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"main {d}: turnover - net yıllık getiri")
        ax.set_xlabel("yıllık turnover (×NAV)")
        ax.set_ylabel("net yıllık getiri")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _base_table_md(rows: list[dict], btc_calmar: float) -> str:
    lines = [f"# {EXPERIMENT_ID} — 216 temel varyant (sabit N/bant; pencere {EVAL_START.date()} → {EVAL_END.date()})", "",
             f"BTC al-tut Calmar: {_fmt(btc_calmar)}. Bilgi amaçlıdır; karar walk-forward ana adaya göre verilir (bkz. sonuc.md).", "",
             "| Sinyal | Yön | N | Bant | Sharpe | Calmar | Maks. DD | Yıllık | Turnover | Maliyet/Brüt | Poz. yıl | BTC korr. |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['signal']} | {r['dir']} | {r['N']} | {r['band']:.2f} | {_fmt(r['sharpe'])} | {_fmt(r['calmar'])} | "
                     f"{_fmt(r['max_dd'], True)} | {_fmt(r['annual_return'], True)} | {_fmt(r['turnover'], nd=1)} | "
                     f"{_fmt(r['cost_to_gross'], True)} | {_fmt(r['positive_years'], True)} | {_fmt(r['corr_btc'])} |")
    return "\n".join(lines) + "\n"


def _results_md(s: dict) -> str:
    L = [f"## SONUÇ — `{EXPERIMENT_ID}` (ön kayıt: \"ÖN KAYIT — trend\")", "",
         f"**Karar:** {s['decision']}", "",
         f"Pencere {s['window'][0]} → {s['window'][1]} · NAV {NAV_PRIMARY:,.0f} USDT · deneme: {s['n_variants']} varyant "
         f"(toplam sayaç {s['trials_after']}) · **PBO (216 varyant): {_fmt(s['pbo'])}** · BTC al-tut Calmar {_fmt(s['btc_calmar'])}", "",
         "### Walk-forward seçilmiş varyantlar (24) — KURALLAR §5 kol eşikleri",
         "| Sinyal | Yön | Sharpe | Calmar | Maks. DD | Yıllık | Turnover | Maliyet/Brüt | Poz. yıl | DSR | BTC korr. | Stres SR | +1g SR | Durum |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["wf"]:
        L.append(f"| {r['signal']} | {r['dir']} | {_fmt(r['sharpe'])} | {_fmt(r['calmar'])} | {_fmt(r['max_dd'], True)} | "
                 f"{_fmt(r['annual_return'], True)} | {_fmt(r['turnover'], nd=1)} | {_fmt(r['cost_to_gross'], True)} | "
                 f"{_fmt(r['positive_years'], True)} | {_fmt(r['dsr'])} | {_fmt(r['corr_btc'])} | {_fmt(r['stress_sharpe'])} | "
                 f"{_fmt(r['delay1_sharpe'])} | {r['status']} |")
    L += ["", "### Kıyaslar", "| Kıyas | Sharpe | Calmar | Maks. DD | Yıllık | Turnover | BTC korr. |", "|---|---|---|---|---|---|---|"]
    for k, r in s["benchmarks"].items():
        L.append(f"| {k} | {_fmt(r['sharpe'])} | {_fmt(r['calmar'])} | {_fmt(r['max_dd'], True)} | {_fmt(r['annual_return'], True)} | "
                 f"{_fmt(r['turnover'], nd=1)} | {_fmt(r['corr_btc'])} |")
    L += ["", "### Ana aday ayrıntıları"]
    for d in DIRECTIONS:
        p = s["plateau"][d]
        r = next(x for x in s["wf"] if x["signal"] == "main" and x["dir"] == d)
        worst = min(p["neighbors"], key=p["neighbors"].get)
        ns = s["nav_small"][d]
        L.append(f"- **main {d}**: walk-forward seçimleri {', '.join(r['selections'])}. Plato (N{p['N']}/b{p['band']:.2f} sabit; "
                 f"temel Sharpe {_fmt(p['base_sharpe'])}): oran **{_fmt(p['ratio'])}**, en kötü komşu `{worst}` "
                 f"({_fmt(p['neighbors'][worst])}). NAV 1.000: atlanan emir denemesi {_fmt(ns['skipped_rate_1000'], True)} "
                 f"(10.000'de {_fmt(ns['skipped_rate_10000'], True)}), yıllık getiri "
                 f"{_fmt(ns['annual_return_1000'], True)} vs {_fmt(ns['annual_return_10000'], True)}.")
        if "legs" in r:
            L.append(f"  Bacak katkısı (toplam, NAV oranı; çıkış maliyeti kendi bacağında): long {_fmt(r['legs']['long'], True)}, "
                     f"short {_fmt(r['legs']['short'], True)} (toplam {_fmt(r['legs']['total'], True)}).")
    L += ["", "### Stres dönemleri (maks. DD / toparlanma günü)", "| Seri | " + " | ".join(STRESS_PERIODS) + " |",
          "|---|" + "---|" * len(STRESS_PERIODS)]
    for k, per in s["stress"].items():
        cells = []
        for name in STRESS_PERIODS:
            x = per.get(name, {})
            if x.get("max_dd") is None:
                cells.append("değerlendirilemez")
            else:
                rec = x["recovery_days"]
                cells.append(f"{x['max_dd']:.1%} / {rec if rec is not None else 'toparlanmadı'}")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["", f"Tam tablo: `research/results/{EXPERIMENT_ID}/varyantlar_216.md`; turnover-getiri: `turnover_getiri.png`. "
          f"Limiti bilinmeyen (delist) semboller kısıtsız varsayıldı: {len(s['unlimited_symbols'])} sembol.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    """Gerçek veride HER ZAMAN kayıtlı koşar (kaydedilmeyen deneme yasak, KURALLAR §7); `register=False` yalnızca testler içindir."""
    argparse.ArgumentParser(description="trend deneyi (ön kayıtlı; kayıt + deneme sayacı)").parse_args(argv)
    s = run_study(register=True)
    print(s["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
