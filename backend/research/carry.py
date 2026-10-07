"""Carry deneyi (funding + vadeli basis, delta-nötr) — ÖN KAYIT: docs/research/deneyler.md "ÖN KAYIT — carry" (2026-10-07).

Bu modüldeki sabitler ön kayıtla BİREBİR aynıdır; sonuç görüldükten sonra değiştirilemez (KURALLAR §5, §7).

Kullanım:  cd backend && python -m research.carry            (gerçek koşu: kayıt + sayaç)

Simülatör (günlük, açılıştan açılışa): karar gün t kapanışında, işlem t+1 açılışında. Pozisyon = q_s coin spot long + q_f coin
perp/quarterly short (hedefte q_s = q_f). Spot cüzdanı coin'i, vadeli cüzdan marjı (E_p) tutar; boştaki nakit ayrı (getiri %0).
Ağırlık motoru (`research.engine`) iki bacak / ayrı cüzdan / tasfiye modellemediği için ayrı, küçük bir simülatör kullanılır."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, engine, stats, trend
from .data import store
from .data.universe import universe_membership
from .kesitsel import load_trend_series
from .report import render_report
from .trend import EVAL_END, EVAL_START, _fmt, window

# ---------------------------------------------------------------- ön kayıt sabitleri
QUESTION = "carry"
EXPERIMENT_ID = "carry_001"
HYPOTHESIS = ("Perpetual funding ve vadeli basis, delta-nötr toplandığında maliyetler sonrası risksiz getirinin belirgin üstünde ve "
              "trend koluyla düşük korelasyonlu getiri sağlar")
SIM_START = pd.Timestamp("2020-01-01")
NAV0 = config.ACCOUNT_NAV  # 250 USDT
RISK_FREE = 0.04
MIN_ANNUAL_RETURN = 2 * RISK_FREE
SPOT_FEE = config.SPOT_FEE
FUT_FEE = config.FUTURES_TAKER_FEE
LIQ_MMR = 0.01  # tasfiye: E_p − q·(H − P_açılış) ≤ %1·q·H
MARGIN_BAND = (0.5, 1.5)  # vadeli cüzdan özsermayesi / hedef marj bu aralık dışına çıkınca yeniden boyutlama
PM_BUFFER = 0.10  # Portföy Marjı: bağlanan sermaye 1,10·N (kaldıraç 10'a eşdeğer marj yapısı, tasfiye yok)
LEG_TAU_MIN = 1  # bacak riski birincil τ (dakika); 5 dk duyarlılık
LEG_TAU_SENS = 5
SIGMA_1H_HOURS = 7 * 24
PRICE_GAP_DAYS = 5  # MTM için fiyat boşluğu ileri taşınır; daha uzun boşluk = fiyat bitti (delist) -> kapat
A_ASSETS = ("BTCUSDT", "ETHUSDT")
LEVERAGES = (1, 2, 3)
B_TOP = 30
B_WINDOWS = (3, 7)
B_KS = (3, 5, 8)
B_LEVERAGES = (1, 3)
SAFETY_MARGIN = 0.10
EXIT_YIELD = 0.03
HOLD_DAYS = 14
VOLUME_CAP = 0.001
VOLUME_WINDOW = 30
C_MIN_DTE = 30
C_MODES = ("vade", "gec")
C_LEVERAGES = (1, 3)
PLATEAU_MULTS = (0.5, 1.5)
B_MAIN = {"K": 5, "W": 7, "lev": 1}
C_MAIN = {"asset": "BTCUSDT", "mode": "vade", "lev": 1}
FUNDING_CAP_EVENT = 0.003
DELIST_LOOKBACK = 30
SHOCK = 0.30
N_BASE = len(A_ASSETS) * len(LEVERAGES) + len(A_ASSETS) + len(B_KS) * len(B_WINDOWS) * len(B_LEVERAGES) \
    + len(A_ASSETS) * len(C_MODES) * len(C_LEVERAGES)  # 8 + 12 + 8 = 28
N_PLANNED = N_BASE + 2 * 3 + 2  # + plato 8 = 36


# ---------------------------------------------------------------- veri
@dataclass
class Data:
    idx: pd.DatetimeIndex
    spot: dict[str, pd.DataFrame]  # open/close/quote_volume: tarih × sembol (perp adıyla)
    perp: dict[str, pd.DataFrame]  # open/high/close/quote_volume
    funding: pd.DataFrame  # günlük oran toplamı ((d, d+1] ödemeleri; pozitif = long öder)
    funding_max: pd.DataFrame  # günlük en büyük |olay oranı|
    slip_bps: pd.DataFrame
    leg_sigma_1h: pd.DataFrame  # bacak riski için 1 saatlik σ (karar günü itibarıyla)
    membership: pd.DataFrame  # noktasal-zamanlı ilk 30 perpetual
    fut: dict[str, pd.DataFrame] = field(default_factory=dict)  # quarterly: open/high/close (idx'e göre)
    fut_meta: dict[str, tuple[str, pd.Timestamp]] = field(default_factory=dict)  # kontrat -> (taban, vade günü)
    step: pd.Series | None = None
    min_notional: pd.Series | None = None
    unlimited: list[str] = field(default_factory=list)
    data_hash: str = "synthetic"
    funding_cap_count: pd.DataFrame | None = None  # günlük |oran| ≥ %0,3 olay sayısı


def _sigma_1h(symbol: str, idx: pd.DatetimeIndex) -> pd.Series | None:
    """Karar günü d için (d+1 00:00'a kadarki) son 168 saatin 1 saatlik log getiri std'si."""
    k = store.load_klines(symbol, "um_1h")
    if k is None or len(k) < SIGMA_1H_HOURS:
        return None
    r = np.log(k["close"]).diff()
    s = r.rolling(SIGMA_1H_HOURS, min_periods=SIGMA_1H_HOURS // 2).std()
    return s.groupby(s.index.floor("D")).last().reindex(idx)


def load_data(with_limits: bool = True) -> Data:
    qv = store.load_panel("quote_volume")
    idx = pd.date_range(SIM_START, EVAL_END, freq="D")
    mem_all = universe_membership(qv, B_TOP).reindex(idx).fillna(False)
    spot_syms = set(store.list_symbols("spot_1d"))
    cands = sorted((set(mem_all.columns[mem_all.any()]) | set(A_ASSETS)) & spot_syms)
    membership = mem_all.reindex(columns=cands).fillna(False).astype(bool)
    perp = {f: store.load_panel(f, cands).reindex(index=idx, columns=cands) for f in ("open", "high", "close", "quote_volume")}
    spot = {f: store.load_panel(f, cands, kind="spot_1d").reindex(index=idx, columns=cands) for f in ("open", "close", "quote_volume")}
    rows, maxes, counts = [], {}, {}
    for s in cands:
        f = store.load_funding(s)
        if f is not None and len(f):
            rows.append(pd.DataFrame({"symbol": s, "time": f.index, "rate": f["rate"].to_numpy()}))
            day = (pd.to_datetime(f.index) - pd.Timedelta(1, "ns")).floor("D")
            maxes[s] = f["rate"].abs().groupby(day).max()
            counts[s] = (f["rate"].abs() >= FUNDING_CAP_EVENT).astype(float).groupby(day).sum()
    funding = engine.daily_funding(pd.concat(rows, ignore_index=True), idx, cands)
    funding_max = pd.DataFrame(maxes).reindex(index=idx, columns=cands).fillna(0.0)
    top20 = universe_membership(qv, 20).reindex(index=idx, columns=cands).fillna(False)
    slip = pd.DataFrame(np.where(top20.to_numpy(), config.SLIPPAGE_BPS_TOP20, config.SLIPPAGE_BPS_OTHER), index=idx, columns=cands)
    for s in config.TOP_TIER_SYMBOLS:
        if s in slip.columns:
            slip[s] = config.SLIPPAGE_BPS_BTC_ETH
    daily_sig = np.log(perp["close"]).diff().rolling(7, min_periods=5).std() / math.sqrt(24)
    sig = {}
    for s in cands:
        h = _sigma_1h(s, idx)
        sig[s] = h.combine_first(daily_sig[s]) if h is not None else daily_sig[s]
    leg_sigma = pd.DataFrame(sig).reindex(index=idx, columns=cands)
    fut, meta = {}, {}
    for c in store.list_symbols("delivery_um_1d"):
        base, _, ymd = c.partition("_")
        if base not in A_ASSETS:
            continue
        k = store.load_klines(c, "delivery_um_1d")
        if k is None or not len(k):
            continue
        fut[c] = k[["open", "high", "close"]].reindex(idx)
        meta[c] = (base, pd.Timestamp(pd.to_datetime(ymd, format="%y%m%d")))
    step = min_n = None
    unlimited: list[str] = []
    if with_limits:
        from .data.limits import engine_limits, get_limits

        min_n, step, unlimited = engine_limits(get_limits(cands))
    cap_count = pd.DataFrame(counts).reindex(index=idx, columns=cands).fillna(0.0)
    return Data(idx, spot, perp, funding, funding_max, slip, leg_sigma, membership, fut, meta, step, min_n, unlimited,
                store.snapshot_hash(), cap_count)


# ---------------------------------------------------------------- maliyet yardımcıları
def leg_risk(sigma_1h, tau_min: float):
    """E|ΔP|/P, τ dakikalık bacak arası fark: σ_1s·√(τ/60)·√(2/π)."""
    return sigma_1h * math.sqrt(tau_min / 60.0) * math.sqrt(2.0 / math.pi)


def roundtrip_cost(d: Data, tau_min: float = LEG_TAU_MIN) -> pd.DataFrame:
    """Notional'a oranla gidiş-dönüş maliyet: 2 × (spot ücreti + vadeli ücreti + iki bacak kayması + bacak riski)."""
    slip = d.slip_bps / 1e4
    return 2.0 * (SPOT_FEE + FUT_FEE + 2 * slip + leg_risk(d.leg_sigma_1h, tau_min).fillna(0.0))


# ---------------------------------------------------------------- strateji kararları (karar günü j; yalnızca <= j verisi)
@dataclass
class Params:
    margin: float = SAFETY_MARGIN
    exit: float = EXIT_YIELD
    hold: float = HOLD_DAYS


class StratA:
    def __init__(self, asset: str):
        self.asset = asset

    def decide(self, d: Data, j: int, held: set, nav: float) -> dict[str, float]:
        return {("spot", self.asset, "perp", self.asset): 1.0}

    def cap(self, d: Data, j: int, inst) -> float:
        return np.inf


class StratB:
    def __init__(self, d: Data, K: int, W: int, p: Params = Params(), tau: float = LEG_TAU_MIN):
        self.K, self.p = K, p
        self.y = d.funding.rolling(W, min_periods=W).mean() * 365.0
        self.thr = roundtrip_cost(d, tau) * 365.0 / p.hold + p.margin
        vol = pd.concat([d.perp["quote_volume"], d.spot["quote_volume"]]).groupby(level=0).min()  # perp ve spotun küçüğü
        self.vcap = VOLUME_CAP * vol.rolling(VOLUME_WINDOW, min_periods=VOLUME_WINDOW // 2).mean()
        self.ok = d.membership & d.spot["close"].notna() & d.perp["close"].notna()

    def decide(self, d: Data, j: int, held: set, nav: float) -> dict:
        y, thr, ok = self.y.iloc[j], self.thr.iloc[j], self.ok.iloc[j]
        keep = {h for h in held if np.isfinite(y.get(h[1], np.nan)) and y[h[1]] >= self.p.exit}
        free = self.K - len(keep)
        out = {h: 1.0 / self.K for h in keep}
        if free > 0:
            held_syms = {h[1] for h in held}
            elig = y[ok & (y >= thr)].drop(labels=list(held_syms), errors="ignore").sort_values(ascending=False, kind="stable")
            for s in elig.index[:free]:
                out[("spot", s, "perp", s)] = 1.0 / self.K
        return out

    def cap(self, d: Data, j: int, inst) -> float:
        v = self.vcap.iloc[j].get(inst[1], np.nan)
        return float(v) if np.isfinite(v) else 0.0


class StratC:
    def __init__(self, d: Data, asset: str, mode: str, p: Params = Params(), tau: float = LEG_TAU_MIN):
        self.asset, self.mode, self.p = asset, mode, p
        self.delay = 0  # simülatör ayarlar: işlem günü = karar günü + 1 + gecikme
        self.rt = roundtrip_cost(d, tau)[asset]
        self.contracts = sorted((c for c, (b, _) in d.fut_meta.items() if b == asset), key=lambda c: d.fut_meta[c][1])

    def basis(self, d: Data, j: int, c: str) -> tuple[float, int]:
        day = d.idx[j]
        dte = (d.fut_meta[c][1] - day).days
        f, s = d.fut[c]["close"].iloc[j], d.spot["close"][self.asset].iloc[j]
        if not (np.isfinite(f) and np.isfinite(s) and s > 0 and dte > 0):
            return float("nan"), dte
        return (f / s - 1.0) * 365.0 / dte, dte

    def decide(self, d: Data, j: int, held: set, nav: float) -> dict:
        k = j + 1 + self.delay
        trade_day = d.idx[k] if k < len(d.idx) else None
        if any(d.fut_meta[c][1] == trade_day for c in self.contracts):  # vade günü: yeni kontrat ertesi karar gününde
            return {h: 1.0 for h in held if d.fut_meta[h[3]][1] != trade_day}
        for h in held:
            if self.mode == "vade":
                return {h: 1.0}
            ann, _ = self.basis(d, j, h[3])
            if np.isfinite(ann) and ann >= self.p.exit:
                return {h: 1.0}
        held_c = {h[3] for h in held}
        for c in self.contracts:
            ann, dte = self.basis(d, j, c)
            if dte < C_MIN_DTE or not np.isfinite(ann) or c in held_c:
                continue
            thr = self.rt.iloc[j] * 365.0 / dte + self.p.margin
            return {("spot", self.asset, "fut", c): 1.0} if ann >= thr else {}
        return {}

    def cap(self, d: Data, j: int, inst) -> float:
        return np.inf


# ---------------------------------------------------------------- simülatör
@dataclass
class Pos:
    inst: tuple
    q_s: float
    q_f: float
    ep: float
    opened: pd.Timestamp
    pnl: float
    ls: float  # son SENKRON açılış fiyatları (iki bacak da aynı gün fiyatlı); MTM ve işlem bu fiyatlarla
    lh: float
    last_sync: int


@dataclass
class SimResult:
    returns: pd.Series
    nav: pd.Series
    bound: pd.Series
    components: pd.DataFrame  # funding, basis, costs, leg_risk, liquidation (USDT; maliyetler pozitif tutulur)
    trades: int
    liquidations: int
    episodes: list[dict]
    funding_cap_events: int
    skipped: dict = field(default_factory=dict)
    lot_scale: float = float("nan")


def simulate(d: Data, strat, *, lev: float, pm: bool = False, fee_mult: float = 1.0, slip_mult: float = 1.0,
             tau: float = LEG_TAU_MIN, cash_rate: float = 0.0, delay: int = 0, nav0: float = NAV0,
             start: pd.Timestamp | None = None) -> SimResult:
    """`start`: hesap bu günün açılışında `nav0` ile başlar (karar verisi tam panelden, yalnızca <= karar günü). Fiyat boşluğu:
    iki bacağın birinin açılış fiyatı yoksa o gün MTM ve işlem yapılmaz (hedge senkron kalır; funding işlemeye devam eder);
    `PRICE_GAP_DAYS`'ten uzun senkron boşluk = fiyat bitti (delist): iki bacak son mevcut KAPANIŞTAN kapatılır."""
    idx, n = d.idx, len(d.idx)
    i0 = 0 if start is None else int(idx.searchsorted(pd.Timestamp(start)))
    eff_lev = (1.0 / PM_BUFFER) if pm else float(lev)
    if hasattr(strat, "delay"):
        strat.delay = delay
    cache: dict = {}

    def raw(kind, sym, fld):
        key = (kind, sym, fld)
        if key not in cache:
            src = d.spot[fld][sym] if kind == "spot" else (d.perp[fld][sym] if kind == "perp" else d.fut[sym][fld])
            cache[key] = src.to_numpy(dtype=float)
        return cache[key]

    def last_close(kind, sym, i):
        c = raw(kind, sym, "close")[:i]
        ok = np.flatnonzero(np.isfinite(c))
        return c[ok[-1]] if len(ok) else np.nan

    slip = d.slip_bps.to_numpy() / 1e4
    sig = d.leg_sigma_1h.to_numpy()
    fund = d.funding.to_numpy()
    capcnt = (d.funding_cap_count if d.funding_cap_count is not None
              else (d.funding_max >= FUNDING_CAP_EVENT).astype(float)).to_numpy()
    col = {s: k for k, s in enumerate(d.slip_bps.columns)}

    def lot(sym, qty, price):
        step = float(d.step.get(sym, np.nan)) if d.step is not None else np.nan
        mn = float(d.min_notional.get(sym, np.nan)) if d.min_notional is not None else np.nan
        if not np.isfinite(price) or price <= 0 or qty <= 0:
            return 0.0
        min_q = (mn / price) if np.isfinite(mn) else 0.0
        if np.isfinite(step) and step > 0:
            min_q = max(step, math.ceil(min_q / step - 1e-9) * step)
            q = round(qty / step) * step
        else:
            q = qty
        if min_q > 0 and qty < 0.5 * min_q:
            return 0.0
        return max(q, min_q)

    def step_of(sym):
        v = float(d.step.get(sym, np.nan)) if d.step is not None else np.nan
        return v if np.isfinite(v) and v > 0 else 0.0

    def lot_sym(inst):
        return inst[3] if inst[2] == "perp" else inst[1]  # quarterly: aynı varlığın perp limiti

    cash, positions = float(nav0), {}
    out_ret, out_nav, out_bound, comps = [], [], [], []
    trades = liqs = 0
    cap_events = 0.0
    skipped: dict[str, int] = {}
    lot_target = lot_final = 0.0
    episodes: list[dict] = []

    def opens(inst, i):
        return raw("spot", inst[1], "open")[i], raw(inst[2], inst[3], "open")[i]

    def trade_cost(inst, i, dqs, dqf, s, h):
        c = col[inst[1]]
        sl = slip[i, c] * slip_mult
        lr = leg_risk(sig[i - 1, c] if i > 0 and np.isfinite(sig[i - 1, c]) else 0.0, tau) * slip_mult
        fees = abs(dqs) * s * (SPOT_FEE * fee_mult + sl) + abs(dqf) * h * (FUT_FEE * fee_mult + sl)
        return fees, max(abs(dqs) * s, abs(dqf) * h) * lr

    def mark(p: Pos, s, h, comp):
        """İki bacağı (s, h) fiyatına işaretle; fark basis kalemine."""
        sp, fp = p.q_s * (s - p.ls), -p.q_f * (h - p.lh)
        p.ep += fp
        p.pnl += sp + fp
        comp["basis"] += sp + fp
        p.ls, p.lh = s, h

    def close(p: Pos, i, reason, comp):
        nonlocal cash, trades
        fees, legr = trade_cost(p.inst, i, p.q_s, p.q_f, p.ls, p.lh)
        cash += p.q_s * p.ls + p.ep - fees - legr
        comp["costs"] += fees
        comp["leg_risk"] += legr
        p.pnl -= fees + legr
        trades += 1
        episodes.append({"symbol": p.inst[1], "hedge": p.inst[3], "opened": str(p.opened.date()), "closed": str(idx[i].date()),
                         "reason": reason, "pnl": p.pnl})

    nav_prev = float(nav0)
    for i in range(i0, n - 1):
        comp = {"funding": 0.0, "basis": 0.0, "costs": 0.0, "leg_risk": 0.0, "liquidation": 0.0}
        # 1) vade ve fiyatı biten (delist) pozisyonlar
        synced = {}
        for key in list(positions):
            p = positions[key]
            s, h = opens(p.inst, i)
            if p.inst[2] == "fut" and d.fut_meta[p.inst[3]][1] <= idx[i]:
                s_use = s if np.isfinite(s) else p.ls
                mark(p, s_use, s_use, comp)  # uzlaşma ≈ spot açılışı
                close(p, i, "vade", comp)
                del positions[key]
                continue
            ok = bool(np.isfinite(s) and np.isfinite(h))
            if not ok and i - p.last_sync > PRICE_GAP_DAYS:
                sc, hc = last_close("spot", p.inst[1], i), last_close(p.inst[2], p.inst[3], i)
                mark(p, sc if np.isfinite(sc) else p.ls, hc if np.isfinite(hc) else p.lh, comp)
                close(p, i, "fiyat bitti (delist)", comp)
                del positions[key]
                continue
            synced[key] = ok
        nav_open = cash + sum(p.q_s * p.ls + p.ep for p in positions.values())
        # 2) karar (j = i − 1 − delay); fiyatı olmayan pozisyon o gün işlem görmez
        j = i - 1 - delay
        desired = strat.decide(d, j, set(positions), nav_open) if j >= 0 else {}
        for key in [k for k in positions if k not in desired and synced.get(k)]:
            close(positions.pop(key), i, "sinyal", comp)
        # 3) cüzdan dengeleme (bant dışında): önce iki bacak birlikte yeniden boyutlanır (aynı q); lot yuvarlaması marjı hedefe
        #    getiremezse boştaki nakit vadeli cüzdana ücretsiz transferle aktarılır (fazlası nakde döner)
        for key, p in list(positions.items()):
            if not synced.get(key):
                continue
            s, h = p.ls, p.lh
            target_margin = p.q_f * h / eff_lev
            ratio = p.ep / target_margin if target_margin > 0 else 0.0
            if p.q_f > 0 and MARGIN_BAND[0] <= ratio <= MARGIN_BAND[1] and abs(p.q_s - p.q_f) < 1e-12:
                continue
            e = p.q_s * s + p.ep
            q_new = lot(lot_sym(p.inst), e / (s + h / eff_lev), h)
            if q_new <= 0:
                close(positions.pop(key), i, "yetersiz özsermaye", comp)
                continue
            if not (q_new == p.q_s == p.q_f):
                fees, legr = trade_cost(p.inst, i, q_new - p.q_s, q_new - p.q_f, s, h)
                p.ep = e - q_new * s - fees - legr
                p.q_s = p.q_f = q_new
                p.pnl -= fees + legr
                comp["costs"] += fees
                comp["leg_risk"] += legr
                trades += 1
            target_margin = p.q_f * h / eff_lev
            if p.ep < MARGIN_BAND[0] * target_margin:
                move = min(target_margin - p.ep, cash)
            elif p.ep > MARGIN_BAND[1] * target_margin:
                move = target_margin - p.ep
            else:
                move = 0.0
            p.ep += move  # transfer: NAV değişmez
            cash -= move
        # 4) yeni pozisyonlar
        for key, w in desired.items():
            if key in positions:
                continue
            s, h = opens(key, i)
            if not (np.isfinite(s) and np.isfinite(h) and s > 0 and h > 0):
                skipped["price"] = skipped.get("price", 0) + 1
                continue
            capital = min(w * nav_open, cash)
            cap = strat.cap(d, j, key)
            notional = min(capital / (1.0 + 1.0 / eff_lev), cap)
            sym = lot_sym(key)
            q = lot(sym, notional / h, h)
            if q <= 0:
                reason = "volume_cap" if cap < capital / (1.0 + 1.0 / eff_lev) else "min_lot"
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            stp = step_of(sym)
            for _ in range(10_000):  # nakde sığana kadar küçült (lotlu: bir adım; lotsuz: orantılı)
                if q <= 0:
                    break
                fees, legr = trade_cost(key, i, q, q, s, h)
                need = q * s + q * h / eff_lev + fees + legr
                if need <= cash + 1e-9:
                    break
                q = lot(sym, q - stp, h) if stp > 0 else q * (cash / need) * 0.999
            if q <= 0:
                skipped["cash"] = skipped.get("cash", 0) + 1
                continue
            fees, legr = trade_cost(key, i, q, q, s, h)
            ep = q * h / eff_lev
            cash -= q * s + ep + fees + legr
            positions[key] = Pos(key, q, q, ep, idx[i], -(fees + legr), s, h, i)
            lot_target += notional
            lot_final += q * h
            comp["costs"] += fees
            comp["leg_risk"] += legr
            trades += 1
        bound = sum(p.q_s * p.ls + p.ep for p in positions.values())
        # 5) gün içi: funding, tasfiye, MTM açılış i -> açılış i+1 (iki bacak da fiyatlıysa)
        for p in positions.values():
            s1, h1 = opens(p.inst, i + 1)
            sync1 = bool(np.isfinite(s1) and np.isfinite(h1))
            fnd = 0.0
            if p.inst[2] == "perp" and p.q_f > 0:
                c = col[p.inst[1]]
                fnd = p.q_f * p.lh * fund[i, c]  # short alır (oran > 0)
                cap_events += capcnt[i, c]
            high = raw(p.inst[2], p.inst[3], "high")[i]
            if not pm and p.q_f > 0 and np.isfinite(high) and p.ep - p.q_f * (high - p.lh) <= LIQ_MMR * p.q_f * high:
                # tasfiye: vadeli cüzdan 0'a iner (negatif bakiye borsanın sigorta fonunda kalır), short kapanır, funding kaybolur.
                # Kayıp = hedge'li kalsaydı oluşacak vadeli PnL'e göre fark ("liquidation"); hedge'li kısım basis'e.
                liqs += 1
                hedged = -p.q_f * ((h1 if sync1 else p.lh) - p.lh)
                comp["liquidation"] += p.ep + hedged  # kayıp (pozitif): gerçekleşen −E_p ile hedge'li PnL farkı
                comp["basis"] += hedged
                p.pnl += -p.ep
                p.ep = 0.0
                p.q_f = 0.0
                if sync1:
                    sp = p.q_s * (s1 - p.ls)
                    p.pnl += sp
                    comp["basis"] += sp
                    p.ls, p.lh, p.last_sync = s1, h1, i + 1
                continue
            p.ep += fnd
            p.pnl += fnd
            comp["funding"] += fnd
            if sync1:
                mark(p, s1, h1, comp)
                p.last_sync = i + 1
        cash *= 1.0 + cash_rate / 365.0
        nav_next = cash + sum(p.q_s * p.ls + p.ep for p in positions.values())
        out_ret.append(nav_next / nav_prev - 1.0 if nav_prev > 0 else 0.0)
        out_nav.append(nav_next)
        out_bound.append(bound)
        comps.append(comp)
        nav_prev = nav_next
    days = idx[i0:n - 1]
    return SimResult(pd.Series(out_ret, index=days), pd.Series(out_nav, index=days), pd.Series(out_bound, index=days),
                     pd.DataFrame(comps, index=days), trades, liqs, episodes, int(round(cap_events)), skipped,
                     (lot_final / lot_target) if lot_target > 0 else float("nan"))


# ---------------------------------------------------------------- varyantlar
def variants(d: Data) -> dict[str, tuple]:
    """Etiket -> (strateji, simülatör kwargs). 28 temel varyant, sadelik sırasıyla."""
    out = {}
    for lev in LEVERAGES:
        for a in A_ASSETS:
            out[f"A|{a[:3]}|{lev}x"] = (StratA(a), {"lev": lev})
    for a in A_ASSETS:
        out[f"A|{a[:3]}|PM"] = (StratA(a), {"lev": 1, "pm": True})
    for lev in C_LEVERAGES:
        for mode in C_MODES:
            for a in A_ASSETS:
                out[f"C|{a[:3]}|{mode}|{lev}x"] = (("C", a, mode), {"lev": lev})
    for lev in B_LEVERAGES:
        for K in B_KS:
            for W in sorted(B_WINDOWS, reverse=True):
                out[f"B|K{K}|W{W}|{lev}x"] = (("B", K, W), {"lev": lev})
    return out


def build(d: Data, spec, p: Params = Params()):
    """Giriş/çıkış eşikleri HER ZAMAN birincil maliyetle (τ = 1 dk, ücret/kayma ×1) hesaplanır; stres ve τ = 5 dk koşuları yalnızca
    yürütme maliyetini değiştirir (aynı kararlar, daha pahalı işlem)."""
    if isinstance(spec, StratA):
        return spec
    if spec[0] == "B":
        return StratB(d, spec[1], spec[2], p)
    return StratC(d, spec[1], spec[2], p)


def run_variant(d: Data, spec, kw: dict, p: Params = Params(), **extra) -> SimResult:
    extra.setdefault("start", EVAL_START)
    return simulate(d, build(d, spec, p), **kw, **extra)


def plateau_runs(d: Data) -> dict[str, tuple[str, SimResult]]:
    out = {}
    for name, base in (("margin", SAFETY_MARGIN), ("exit", EXIT_YIELD), ("hold", HOLD_DAYS)):
        for mult in PLATEAU_MULTS:
            p = Params(**{name: base * mult})
            out[f"B|{name}×{mult}"] = ("B", run_variant(d, ("B", B_MAIN["K"], B_MAIN["W"]), {"lev": B_MAIN["lev"]}, p))
    for mult in PLATEAU_MULTS:
        p = Params(margin=SAFETY_MARGIN * mult)
        out[f"C|margin×{mult}"] = ("C", run_variant(d, ("C", C_MAIN["asset"], C_MAIN["mode"]), {"lev": C_MAIN["lev"]}, p))
    return out


# ---------------------------------------------------------------- metrikler
def longest_underwater(r: pd.Series) -> int:
    nav = (1 + r.fillna(0)).cumprod()
    under = nav < nav.cummax()
    best = cur = 0
    for u in under.to_numpy():
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return best


def sim_metrics(res: SimResult, btc_w: pd.Series, trend_df: pd.DataFrame) -> dict:
    r = window(res.returns)
    s = stats.summary(r, btc_w)
    sl = slice(EVAL_START, EVAL_END)
    nav_prev = res.nav.shift(1).fillna(NAV0).loc[sl]
    bound = res.bound.loc[sl]
    pnl = res.nav.diff().fillna(res.nav.iloc[0] - NAV0).loc[sl]
    nav0_w = float(nav_prev.iloc[0]) if len(nav_prev) else NAV0
    years = s["yearly_returns"]
    comps = res.components.loc[sl].sum() / nav0_w
    corr = {c: float(pd.concat([r, window(trend_df[c])], axis=1).dropna().corr().iloc[0, 1]) for c in trend_df.columns}
    return {
        "annual_return": s["annual_return"], "vol": s["annual_vol"], "sharpe": s["sharpe"], "max_dd": s["max_drawdown"],
        "calmar": s["calmar"], "underwater_days": longest_underwater(r),
        "capital_efficiency": float((bound / nav_prev).mean()),
        # bağlanan sermayeye göre (aritmetik): Σ günlük PnL / Σ günlük bağlanan sermaye × 365 (pozisyonsuz günlerin PnL'i dahil)
        "bound_annual_return": float(pnl.sum() / bound.sum() * config.ANNUALIZATION_DAYS) if bound.sum() > 0 else float("nan"),
        "trades": res.trades, "liquidations": res.liquidations, "funding_cap_events": res.funding_cap_events,
        "skipped": dict(res.skipped), "lot_scale": res.lot_scale, "pnl_liquidation": -float(comps["liquidation"]),
        "pnl_funding": float(comps["funding"]), "pnl_basis": float(comps["basis"]), "pnl_costs": -float(comps["costs"]),
        "pnl_leg_risk": -float(comps["leg_risk"]),
        "positive_years": (sum(v > 0 for v in years.values()) / len(years)) if years else float("nan"),
        "yearly_full": {int(k): float(v) for k, v in stats.yearly_returns(res.returns).items()},
        "corr_btc": s.get("corr_vs_btc", float("nan")), "corr_trend": corr,
    }


def margin_report(d: Data) -> dict:
    """Her varlık ve kaldıraç: pencere içi en kötü günlük yükseliş (high/açılış − 1) ve +%30 şok, hedef marjda ve dengeleme alt
    bandında (marjın %50'si) tamponu aşıyor mu (tasfiye eşiği ≈ marj oranı − %1)."""
    out = {}
    for a in A_ASSETS:
        rise = (d.perp["high"][a] / d.perp["open"][a] - 1.0).loc[EVAL_START:EVAL_END]
        worst = float(rise.max())
        for lev in LEVERAGES:
            m = 1.0 / lev
            out[f"{a}|{lev}x"] = {"worst_daily_rise": worst, "worst_date": str(rise.idxmax().date()),
                                  "survives_worst_at_target": worst < m - LIQ_MMR, "survives_worst_at_band": worst < MARGIN_BAND[0] * m - LIQ_MMR,
                                  "survives_30pct_at_target": SHOCK < m - LIQ_MMR, "survives_30pct_at_band": SHOCK < MARGIN_BAND[0] * m - LIQ_MMR}
    for a in A_ASSETS:  # PM: fiyat şoku hedge'le dengelenir; %10 tampon yalnızca basis şoku (perp−spot farkı) içindir
        b = (d.perp["close"][a] / d.spot["close"][a] - 1.0).loc[EVAL_START:EVAL_END]
        worst_b = float(b.diff().abs().max())
        out[f"{a}|PM (günlük basis değişimi vs %10 tampon)"] = {"worst_daily_rise": worst_b, "worst_date": str(b.diff().abs().idxmax().date()),
                          "survives_worst_at_target": worst_b < PM_BUFFER, "survives_worst_at_band": worst_b < MARGIN_BAND[0] * PM_BUFFER,
                          "survives_30pct_at_target": True, "survives_30pct_at_band": True, "note": "PM: en kötü günlük basis değişimi"}
    return out


def delist_report(d: Data, sims: dict[str, SimResult]) -> list[dict]:
    """B varyantlarında fiyatı biten (delist) sembolde, bitişten önceki 30 gün içinde açık olan carry pozisyonları."""
    rows = []
    for label, res in sims.items():
        for e in res.episodes:
            if e["reason"] == "fiyat bitti (delist)":
                rows.append({"variant": label, **e})
    last_valid = d.perp["close"].apply(lambda c: c.last_valid_index())
    ended = last_valid[last_valid < EVAL_END - pd.Timedelta(days=PRICE_GAP_DAYS)]
    for label, res in sims.items():
        for e in res.episodes:
            s = e["symbol"]
            if s in ended.index and e["reason"] != "fiyat bitti (delist)":
                end = ended[s]
                if pd.Timestamp(e["closed"]) >= end - pd.Timedelta(days=DELIST_LOOKBACK):
                    rows.append({"variant": label, **e, "note": f"fiyat {end.date()}'de bitti"})
    return rows


# ---------------------------------------------------------------- ana akış
def _check(m: dict, dsr: float, pbo: float, stress_sr: float, btc_calmar: float, plateau: float) -> dict:
    acc = stats.check_acceptance(
        sharpe_net=m["sharpe"], deflated=dsr, pbo=pbo, max_dd=m["max_dd"], calmar_value=m["calmar"], btc_calmar=btc_calmar,
        positive_year_fraction=m["positive_years"], stress_sharpe=stress_sr, plateau_ratio=plateau, is_portfolio=False)
    acc["min_return"] = bool(m["annual_return"] == m["annual_return"] and m["annual_return"] >= MIN_ANNUAL_RETURN)
    return acc


def run_study(*, data: Data | None = None, trend_returns: pd.DataFrame | None = None, results_dir: Path | None = None,
              log_path: Path | None = None, registry_file: Path | None = None, register: bool = True, progress=print) -> dict:
    from .registry import assert_budget, current_trial_count, register_experiment

    if register:
        assert_budget(QUESTION, N_PLANNED, registry_file)
    d = data or load_data()
    trend_df = trend_returns if trend_returns is not None else load_trend_series()
    results_dir = Path(results_dir or config.RESULTS_DIR)
    registry_file = Path(registry_file or config.REGISTRY_FILE)
    out_dir = results_dir / EXPERIMENT_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    a = config.ACCEPTANCE

    btc_open = d.perp["open"][["BTCUSDT"]].loc[trend.POSITION_START:EVAL_END]
    btc = engine.run(pd.DataFrame(1.0, index=btc_open.index, columns=["BTCUSDT"]), btc_open,
                     fee_rate=config.FUTURES_TAKER_FEE, slippage_bps=config.SLIPPAGE_BPS_BTC_ETH)
    btc_w = window(btc.returns)
    btc_calmar = stats.calmar(btc_w)
    rf_series = pd.Series((1 + RISK_FREE) ** (1 / 365) - 1, index=btc_w.index)

    # 1) 28 temel varyant (+ stres, gecikme, τ=5, faizli boş nakit)
    specs = variants(d)
    base, rows = {}, {}
    for label, (spec, kw) in specs.items():
        res = run_variant(d, spec, kw)
        base[label] = res
        metrics = sim_metrics(res, btc_w, trend_df)
        if label.startswith("A|"):  # A ayrıca 2020-01-01'den koşar: yalnızca yıllık tablo (bilgi)
            metrics["yearly_full"] = {int(k): float(v) for k, v in stats.yearly_returns(run_variant(d, spec, kw, start=None).returns).items()}
        rows[label] = {
            "metrics": metrics,
            "stress": window(run_variant(d, spec, kw, fee_mult=a["stress_fee_mult"], slip_mult=a["stress_slippage_mult"]).returns),
            "delay": window(run_variant(d, spec, kw, delay=1).returns),
            "tau5": window(run_variant(d, spec, kw, tau=LEG_TAU_SENS).returns),
            "cash_rf": window(run_variant(d, spec, kw, cash_rate=RISK_FREE).returns),
        }
        progress(f"varyant: {label} tamam ({len(base)}/{len(specs)})")

    # 2) plato
    plat = plateau_runs(d)
    b_main = f"B|K{B_MAIN['K']}|W{B_MAIN['W']}|{B_MAIN['lev']}x"
    c_main = f"C|{C_MAIN['asset'][:3]}|{C_MAIN['mode']}|{C_MAIN['lev']}x"
    plateau = {}
    for strat, main in (("B", b_main), ("C", c_main)):
        base_sr = stats.sharpe(window(base[main].returns))
        neigh = {k: stats.sharpe(window(v.returns)) for k, (s, v) in plat.items() if s == strat}
        plateau[strat] = {"main": main, "base_sharpe": base_sr, "neighbors": neigh,
                          "ratio": (min(neigh.values()) / base_sr)
                          if (base_sr == base_sr and base_sr > 0 and all(v == v for v in neigh.values())) else float("nan")}
    plateau["A"] = {"main": None, "ratio": float("inf"), "note": "ayarlanabilir parametre yok; uygulanamaz (geçti sayılır)"}
    progress("plato tamam")

    # 3) deneme sayımı, PBO, DSR
    all_trials = [r.returns for r in base.values()] + [v.returns for _, v in plat.values()]
    if len(all_trials) != N_PLANNED:
        raise RuntimeError(f"varyant sayısı ön kayıttan farklı: {len(all_trials)} != {N_PLANNED}")
    trials_after = current_trial_count(registry_file) + N_PLANNED
    trial_sharpes = [x for x in (trend.daily_sharpe(r) for r in all_trials) if x == x]
    matrix = pd.concat({k: window(v.returns) for k, v in base.items()}, axis=1).fillna(0.0)
    pbo = stats.pbo_cscv(matrix.to_numpy(), n_blocks=16)

    # 4) kabul
    table = []
    for label, row in rows.items():
        m = row["metrics"]
        dsr = stats.deflated_sharpe(window(base[label].returns), n_trials=trials_after, trial_sharpes=trial_sharpes)
        stress_sr = stats.sharpe(row["stress"])
        acc = _check(m, dsr, pbo, stress_sr, btc_calmar, plateau[label[0]]["ratio"])
        failed = [k for k, v in acc.items() if not v]
        table.append({"variant": label, **m, "dsr": dsr, "stress_sharpe": stress_sr, "delay1_sharpe": stats.sharpe(row["delay"]),
                      "tau5_sharpe": stats.sharpe(row["tau5"]), "cash_rf_annual": stats.annual_return(row["cash_rf"]),
                      "acceptance": acc, "status": "GEÇTİ" if not failed else "KALDI (" + ", ".join(failed) + ")"})
    passed = [r for r in table if r["status"] == "GEÇTİ"]  # tablo sadelik sırasında
    if passed:
        decision = f"CARRY KOLU ADAYI: {passed[0]['variant']} (geçen en basit varyant; geçen toplam {len(passed)})"
    else:
        decision = "KALDI — hiçbir varyant tüm koşulları geçmedi. DUR (ızgara genişletilmez)."

    # 5) ekler
    main_labels = ["A|BTC|1x", c_main, b_main]
    stress = {k: trend.stress_table(window(base[k].returns)) for k in main_labels}
    stress["BTC al-tut"] = trend.stress_table(btc_w)
    b_sims = {k: v for k, v in base.items() if k.startswith("B|")}
    from .runner import recommended_nav_1x

    last = d.idx[-1]
    univ = sorted(set(d.membership.columns[d.membership.loc[last]]) | set(A_ASSETS))
    nav_1x, nav_1x_sym = recommended_nav_1x(d.min_notional, d.step, d.perp["close"].loc[:last].ffill().iloc[-1][univ])
    summary = {
        "experiment_id": EXPERIMENT_ID, "decision": decision, "pbo": pbo, "n_variants": N_PLANNED, "trials_after": trials_after,
        "btc_calmar": btc_calmar, "btc": stats.summary(btc_w), "risk_free": {"annual": RISK_FREE, "annual_return": stats.annual_return(rf_series)},
        "plateau": plateau, "table": table, "stress": stress, "margin": margin_report(d), "delist": delist_report(d, b_sims),
        "unlimited_symbols": d.unlimited, "data_hash": d.data_hash, "window": [str(EVAL_START.date()), str(EVAL_END.date())],
        "account": {"nav": NAV0, "recommended_nav_1x": nav_1x, "recommended_nav_1x_symbol": nav_1x_sym,
                    "recommended_nav_1x_price_date": str(last.date())},
    }
    (out_dir / "ozet.json").write_text(json.dumps(summary, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("acceptance", "yearly_full", "corr_trend")} for r in table]).to_csv(
        out_dir / "varyantlar.csv", index=False)
    md = _results_md(summary)
    (out_dir / "sonuc.md").write_text(md, encoding="utf-8")
    best = passed[0]["variant"] if passed else "A|BTC|1x"
    render_report(window(base[best].returns), btc_w, out_dir / "aday", f"{EXPERIMENT_ID} · {best}",
                  extra={"karar": next(r["status"] for r in table if r["variant"] == best), "PBO": _fmt(pbo)})

    if register:
        frame = pd.concat({**{k: window(v.returns) for k, v in base.items()}, **{f"plato|{k}": window(v.returns) for k, (_, v) in plat.items()}},
                          axis=1)
        cfg = {"question": QUESTION, "hypothesis": HYPOTHESIS, "window": summary["window"], "sim_start_a_yearly_only": str(SIM_START.date()),
               "eval_sim_start": str(EVAL_START.date()),
               "nav": NAV0, "risk_free": RISK_FREE, "min_annual_return": MIN_ANNUAL_RETURN, "leverages": list(LEVERAGES),
               "pm_buffer": PM_BUFFER, "liq_mmr": LIQ_MMR, "margin_band": list(MARGIN_BAND), "leg_tau_min": LEG_TAU_MIN,
               "b": {"top": B_TOP, "windows": list(B_WINDOWS), "ks": list(B_KS), "leverages": list(B_LEVERAGES), "safety_margin": SAFETY_MARGIN,
                     "exit": EXIT_YIELD, "hold_days": HOLD_DAYS, "volume_cap": VOLUME_CAP},
               "c": {"min_dte": C_MIN_DTE, "modes": list(C_MODES), "leverages": list(C_LEVERAGES)},
               "pre_registration": "deneyler.md ÖN KAYIT — carry"}
        first = table[0]
        register_experiment(
            EXPERIMENT_ID, cfg, frame.astype("float32"),
            {"sharpe": first["sharpe"], "max_drawdown": first["max_dd"], "deflated_sharpe": first["dsr"], "pbo": pbo, "decision": decision,
             "first_variant": first["variant"]},
            n_variants=N_PLANNED, hypothesis=HYPOTHESIS, decision=decision, question=QUESTION, results_dir=results_dir,
            log_path=log_path, registry_file=registry_file, data_hash=d.data_hash,
        )
        log = Path(log_path or config.EXPERIMENT_LOG)
        log.write_text(log.read_text(encoding="utf-8").rstrip("\n") + "\n\n" + md, encoding="utf-8")
    return summary


def _results_md(s: dict) -> str:
    L = [f"## SONUÇ — `{EXPERIMENT_ID}` (ön kayıt: \"ÖN KAYIT — carry\")", "",
         f"**Karar:** {s['decision']}", "",
         f"Pencere {s['window'][0]} → {s['window'][1]} · hesap {s['account']['nav']:g} USDT · deneme: {s['n_variants']} varyant "
         f"(toplam sayaç {s['trials_after']}) · **PBO (28 temel varyant): {_fmt(s['pbo'])}** · BTC al-tut Calmar {_fmt(s['btc_calmar'])} · "
         f"risksiz %{RISK_FREE * 100:.0f} (getiri eşiği %{MIN_ANNUAL_RETURN * 100:.0f})", "",
         "### Varyantlar (sadelik sırasıyla) — KURALLAR §5 kol eşikleri + yıllık getiri ≥ %8",
         "| Varyant | Yıllık (NAV) | Yıllık (bağlı serm., aritm.) | Vol | Sharpe | Maks. DD | Calmar | Su altı (g) | Serm. verim. | İşlem | "
         "Tasfiye | Funding | Basis | Maliyet | Bacak r. | Tasfiye kaybı | Atlanan | lot_scale | DSR | Stres SR | τ5 SR | +1g SR | BTC korr. | "
         "Trend LF/LS | Durum |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["table"]:
        ct = r["corr_trend"]
        L.append(f"| {r['variant']} | {_fmt(r['annual_return'], True)} | {_fmt(r['bound_annual_return'], True)} | {_fmt(r['vol'], True)} | "
                 f"{_fmt(r['sharpe'])} | {_fmt(r['max_dd'], True)} | {_fmt(r['calmar'])} | {r['underwater_days']} | "
                 f"{_fmt(r['capital_efficiency'], True)} | {r['trades']} | {r['liquidations']} | {_fmt(r['pnl_funding'], True)} | "
                 f"{_fmt(r['pnl_basis'], True)} | {_fmt(r['pnl_costs'], True)} | {_fmt(r['pnl_leg_risk'], True)} | "
                 f"{_fmt(r['pnl_liquidation'], True)} | {', '.join(f'{k} {v}' for k, v in r['skipped'].items()) or '—'} | "
                 f"{_fmt(r['lot_scale'])} | {_fmt(r['dsr'])} | "
                 f"{_fmt(r['stress_sharpe'])} | {_fmt(r['tau5_sharpe'])} | {_fmt(r['delay1_sharpe'])} | {_fmt(r['corr_btc'])} | "
                 f"{_fmt(ct.get('main_LF_wf'))} / {_fmt(ct.get('main_LS_wf'))} | {r['status']} |")
    L += ["", "PnL kalemleri pencere toplamıdır, pencere başı NAV'ına (250 USDT) oranla. Basis = iki bacağın fiyat PnL'i; tasfiye kaybı = "
          "hedge'li kalsaydı oluşacak vadeli PnL'e göre fark (ayrı kalem). Stres (ücret ×2, kayma ve bacak riski ×3) ve τ = 5 dk koşularında "
          "giriş/çıkış eşikleri birincil maliyetle kalır; yalnızca yürütme pahalanır. Bağlanan sermayeye göre getiri aritmetiktir "
          "(Σ PnL / Σ bağlı sermaye × 365).",
          "1x için önerilen asgari hesap (KURALLAR §3 tanımı: her sembolün en küçük lotu %20 tavana sığar; carry'de gereken hesap "
          "yaklaşık lot × (1 + 1/kaldıraç) × K): "
          f"{_fmt(s['account']['recommended_nav_1x'], nd=0)} USDT ({s['account']['recommended_nav_1x_symbol']}, "
          f"fiyat tarihi {s['account']['recommended_nav_1x_price_date']}).",
          "", "### Yıllık getiri (NAV). A: 2020-01-01'den ayrı koşu (2020 ve 2021-Q1 bilgi); B ve C: 2021-04-01'den (2021 kısmi yıl)",
          "| Varyant | " + " | ".join(str(y) for y in range(2020, 2026)) + " | Boş nakit %4 ile yıllık |", "|---|" + "---|" * 7]
    for r in s["table"]:
        L.append(f"| {r['variant']} | " + " | ".join(_fmt(r["yearly_full"].get(y), True) for y in range(2020, 2026)) +
                 f" | {_fmt(r['cash_rf_annual'], True)} |")
    L += ["", "### Plato"]
    for k in ("B", "C"):
        p = s["plateau"][k]
        L.append(f"- {k} ana aday `{p['main']}` (Sharpe {_fmt(p['base_sharpe'])}): oran **{_fmt(p['ratio'])}**; komşular: "
                 + ", ".join(f"{n} {_fmt(v)}" for n, v in p["neighbors"].items()) + ".")
    L.append(f"- A: {s['plateau']['A']['note']}.")
    L += ["", "### Marj tamponu (tasfiye eşiği ≈ marj oranı − %1)", "| Varlık/kaldıraç | En kötü günlük yükseliş | Hedef marjda | Bant altında (%50) | +%30 hedefte | +%30 bant altında |",
          "|---|---|---|---|---|---|"]
    yn = {True: "yeter", False: "YETMEZ"}
    for k, v in s["margin"].items():
        L.append(f"| {k} | {_fmt(v['worst_daily_rise'], True)} ({v['worst_date']}) | {yn[v['survives_worst_at_target']]} | "
                 f"{yn[v['survives_worst_at_band']]} | {yn[v['survives_30pct_at_target']]} | {yn[v['survives_30pct_at_band']]} |")
    L += ["", "### Stres dönemleri (maks. DD / toparlanma günü)", "| Seri | " + " | ".join(trend.STRESS_PERIODS) + " |",
          "|---|" + "---|" * len(trend.STRESS_PERIODS)]
    for k, per in s["stress"].items():
        cells = []
        for name in trend.STRESS_PERIODS:
            x = per.get(name, {})
            cells.append("değerlendirilemez" if x.get("max_dd") is None else
                         f"{x['max_dd']:.1%} / {x['recovery_days'] if x['recovery_days'] is not None else 'toparlanmadı'}")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    dl = s["delist"]
    L += ["", "### Delist ve kuyruk riski", f"- B varyantlarında delist edilen sembolde (bitişten önceki {DELIST_LOOKBACK} gün) açık carry "
          f"pozisyonu: {len(dl)} kayıt" + (": " + "; ".join(f"{x['variant']} {x['symbol']} {x['opened']}→{x['closed']} PnL {x['pnl']:.2f} USDT"
                                                          for x in dl[:12]) if dl else "") + ".",
          "- Funding tavanı yakınlığı (tutulan pozisyonda |oran| ≥ %0,3/olay): tablo sütunu yok; `ozet.json` → `funding_cap_events`.",
          "- ADL simüle edilemez (veri yok).",
          f"- BTC al-tut (pencere): Sharpe {_fmt(s['btc']['sharpe'])}, yıllık {_fmt(s['btc']['annual_return'], True)}; risksiz %4 kıyası (D) yıllık "
          f"{_fmt(s['risk_free']['annual_return'], True)}.",
          f"- Limiti bilinmeyen (delist) semboller kısıtsız varsayıldı: {len(s['unlimited_symbols'])} sembol.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    """Gerçek veride HER ZAMAN kayıtlı koşar (kaydedilmeyen deneme yasak, KURALLAR §7); `register=False` yalnızca testler içindir."""
    argparse.ArgumentParser(description="carry deneyi (ön kayıtlı; kayıt + deneme sayacı)").parse_args(argv)
    s = run_study(register=True)
    print(s["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
