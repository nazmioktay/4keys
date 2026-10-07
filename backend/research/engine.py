"""Vektörel portföy backtest motoru (günlük).

Zaman çizelgesi (KURALLAR.md §4): sinyal günlük kapanışta (00:00 UTC) üretilir; dolum sonraki
açılışta. `target[d]` = d günü kapanışında (= d+1 gününün 00:00'ı) karar verilen hedef ağırlık;
`delay_bars=0` iken d+1 açılışında dolar ve d+1 günü boyunca tutulur. Bir günün getirisi
açılıştan-açılışa (`open[d+1]/open[d]-1`) ölçülür; son gün (d+1 açılışı yok) atılır.

Maliyet = |Δağırlık| × (ücret + kayma) (her dolumda NAV'a oranla). Funding: notional × oran,
`(gün başı, gün sonu]` aralığındaki ödemeler (long öder, short alır) — `system_runner` ile aynı
sınır semantiği (çıkış sınırı dahil, giriş sınırı hariç).

Girdi/çıktı ağırlıkları NAV'a oranlı, işaretlidir (+long, −short).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    FUTURES_TAKER_FEE,
    SLIPPAGE_BPS_BTC_ETH,
    SLIPPAGE_BPS_OTHER,
    SLIPPAGE_BPS_TOP20,
    TOP_TIER_SYMBOLS,
)
from .guard import assert_no_final_test


@dataclass
class EngineResult:
    nav: pd.Series  # gün sonu NAV endeksi (başlangıç 1,0)
    returns: pd.Series  # günlük net getiri
    turnover: pd.Series  # Σ|Δağırlık|
    costs: pd.DataFrame  # kolonlar: commission, slippage, funding (NAV oranı, pozitif = maliyet)
    gross_returns: pd.Series
    symbol_contrib: pd.DataFrame  # sembol bazında net getiri katkısı (toplamı = returns)
    weights_held: pd.DataFrame  # gün boyunca tutulan (maliyet sonrası) ağırlıklar
    skipped: list[dict] = field(default_factory=list)  # asgari emir altında açılmayanlar
    diagnostics: dict = field(default_factory=dict)

    def symbol_pnl(self) -> pd.Series:
        """Sembol bazında toplam net getiri katkısı (basit toplam)."""
        return self.symbol_contrib.sum().sort_values(ascending=False)


def default_slippage_bps(symbols, top20: set[str] | None = None) -> pd.Series:
    """Likidite kademesine göre kayma (bps): BTC/ETH 2, ilk 20 sembol 5, diğerleri 15."""
    top20 = top20 or set()
    values = {}
    for s in symbols:
        if s in TOP_TIER_SYMBOLS:
            values[s] = SLIPPAGE_BPS_BTC_ETH
        elif s in top20:
            values[s] = SLIPPAGE_BPS_TOP20
        else:
            values[s] = SLIPPAGE_BPS_OTHER
    return pd.Series(values, dtype=float)


def daily_funding(events: pd.DataFrame, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """Funding olaylarını (kolonlar: symbol, time, rate) günlük tutma dönemlerine toplar.
    d günü = (d 00:00, d+1 00:00] aralığındaki ödemelerin oran toplamı. Pozitif = long öder."""
    out = pd.DataFrame(0.0, index=index, columns=list(columns))
    if events is None or len(events) == 0:
        return out
    ev = events.copy()
    ev["day"] = (pd.to_datetime(ev["time"]) - pd.Timedelta(1, "ns")).dt.floor("D")
    pivot = ev.pivot_table(index="day", columns="symbol", values="rate", aggfunc="sum")
    pivot = pivot.reindex(index=index, columns=list(columns)).fillna(0.0)
    return pivot


def _as_frame(value, index: pd.DatetimeIndex, columns: pd.Index, default: float) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame(default, index=index, columns=columns)
    if isinstance(value, (int, float, np.floating)):
        return pd.DataFrame(float(value), index=index, columns=columns)
    if isinstance(value, pd.Series):
        return pd.DataFrame(np.tile(value.reindex(columns).fillna(default).to_numpy(), (len(index), 1)), index=index, columns=columns)
    return value.reindex(index=index, columns=columns).ffill().fillna(default)


def run(
    target: pd.DataFrame,
    open_prices: pd.DataFrame,
    *,
    fee_rate=None,
    slippage_bps=None,
    funding: pd.DataFrame | None = None,
    band: float | pd.Series = 0.0,
    band_relative: bool = False,
    rebalance_days: pd.DatetimeIndex | None = None,
    trade_mask: pd.DataFrame | None = None,
    account_size: float = 100_000.0,
    delay_bars: int = 0,
    min_notional: pd.Series | None = None,
    amount_step: pd.Series | None = None,
    lot_rounding: str = "floor",
    max_gross: float | None = None,
    allow_final_test: bool = False,
    final_test_experiment_id: str | None = None,
) -> EngineResult:
    """Hedef ağırlık matrisinden günlük NAV/getiri üretir.

    `target`: tarih × sembol hedef ağırlık (NaN = 0). `open_prices`: tarih × sembol günlük açılış.
    `fee_rate`: oran (0,0005 = %0,05); skaler/Series/DataFrame. `slippage_bps`: bps; aynı biçimler.
    `funding`: `daily_funding` çıktısı (oran; pozitif long öder). `band`: yeniden dengeleme bandı
    (|hedef − mevcut| ≤ band ise işlem yapılmaz; skaler ya da işlem günü indeksli Series); `band_relative=True` ise bant hedefe göreli: |hedef − mevcut| ≤
    band·|hedef| (hedef 0 = çıkış her zaman işlem görür). `rebalance_days`: verilirse YALNIZCA bu günlerin açılışında
    işlem yapılır (diğer günler pozisyon drift'le tutulur; fiyatı olmayan pozisyon yine tasfiye edilir). `trade_mask`: işlem günü ×
    sembol boolean; `rebalance_days` dışındaki günlerde YALNIZCA True hücrelerde işlem yapılır (ör. zarar limiti çıkışı; bant geçerli).
    `rebalance_days` yoksa her gün yalnızca maske hücrelerinde işlem yapılır. `min_notional`/`amount_step`: sembol limitleri;
    asgari emrin altında kalan YENİ/ARTAN pozisyonlar açılmaz ve `skipped`'a yazılır.
    `lot_rounding`: "floor" (varsayılan; miktar adıma aşağı yuvarlanır, asgari emrin altı atlanır) veya "nearest" (miktar en
    yakın lota yuvarlanır; hedef en küçük lotun (max(min notional, 1 adım)) yarısından küçükse 0 lot: yeni pozisyon açılmaz,
    mevcut pozisyon KAPATILIR (tam kapanış her zaman yapılabilir varsayılır); aksi halde en az bir lot. Yuvarlama pozisyonu
    hedefin 2 katına kadar büyütebilir: `diagnostics["lot_scale"]` = GERÇEKLEŞEN emirlerde Σ|yuvarlanmış| / Σ|hedef| raporlanır;
    atlanan emirler oranda yoktur).
    `max_gross`: kaldıraç tavanı (NAV katı). Hedeflerin brütü tavanı aşarsa önce TÜM hedefler orantılı küçültülür (sıra bağımlılığı
    yok); "nearest"te önce küçültme/kapanış emirleri, sonra artışlar işlenir; yuvarlama sonrası brütü artıran emir tavanı aşacaksa
    TAMAMEN atlanır ("leverage"; muhafazakâr: tavana sığan daha küçük bir lot denenmez); küçültme emirleri engellenmez."""
    assert_no_final_test(open_prices.index, allow_final_test, final_test_experiment_id)
    assert_no_final_test(target.index, allow_final_test, final_test_experiment_id)
    if delay_bars < 0:
        raise ValueError("delay_bars >= 0 olmalı")
    if lot_rounding not in ("floor", "nearest"):
        raise ValueError(f"lot_rounding 'floor' | 'nearest' olmalı: {lot_rounding!r}")

    index = open_prices.index
    columns = open_prices.columns
    n_days, n_sym = len(index), len(columns)
    fee = _as_frame(fee_rate, index, columns, FUTURES_TAKER_FEE).to_numpy()
    slip = _as_frame(slippage_bps, index, columns, SLIPPAGE_BPS_OTHER).to_numpy() / 1e4
    fund = (funding.reindex(index=index, columns=columns).fillna(0.0) if funding is not None else pd.DataFrame(0.0, index=index, columns=columns)).to_numpy()
    held_target = target.reindex(index=index, columns=columns).shift(1 + delay_bars).fillna(0.0).to_numpy()

    px = open_prices.to_numpy(dtype=float)
    ret = np.full((n_days, n_sym), np.nan)
    ret[:-1] = px[1:] / px[:-1] - 1.0  # açılıştan açılışa; son günün getirisi yok

    min_n = None if min_notional is None else min_notional.reindex(columns).to_numpy(dtype=float)
    step = None if amount_step is None else amount_step.reindex(columns).to_numpy(dtype=float)

    rebalance = None if rebalance_days is None else index.isin(pd.DatetimeIndex(rebalance_days))
    mask = None if trade_mask is None else trade_mask.reindex(index=index, columns=columns).fillna(False).astype(bool).to_numpy()
    lot_target = lot_final = 0.0  # "nearest" yuvarlamasının hedefi ne kadar büyüttüğü (lot_scale)
    band_by_day = (band.reindex(index).ffill().fillna(0.0).to_numpy(dtype=float) if isinstance(band, pd.Series)
                   else np.full(n_days, float(band)))
    w_prev = np.zeros(n_sym)  # önceki günün sonunda (drift sonrası) ağırlık
    nav = 1.0
    out_nav, out_ret, out_gross, out_turn = [], [], [], []
    out_comm, out_slip, out_fund = [], [], []
    out_trades = []  # gün başına GERÇEKTEN yapılan emir sayısı (Δağırlık ≠ 0)
    contrib = np.zeros((n_days - 1, n_sym))
    held = np.zeros((n_days - 1, n_sym))
    skipped: list[dict] = []
    missing_days = 0
    forced_liquidations = 0

    for d in range(n_days - 1):
        wanted = held_target[d].copy()
        r = ret[d]
        tradable = np.isfinite(px[d]) & np.isfinite(px[d + 1])
        # Fiyatı olmayan sembolde pozisyon tutulamaz: varsa tasfiye (getiri 0), yenisi açılmaz.
        forced_liquidations += int(np.sum(~tradable & (np.abs(w_prev) > 0)))
        missing_days += int(np.sum(~tradable & ((np.abs(wanted) > 0) | (np.abs(w_prev) > 0))))
        wanted[~tradable] = 0.0
        w_cur = np.where(tradable, w_prev, 0.0)

        # yeniden dengeleme bandı: küçük sapmalarda mevcut pozisyon korunur
        if max_gross is not None:  # sinyalin kendisi kaldıraç tavanını aşıyorsa orantılı küçült (sütun sırasına bağlı kalmasın)
            g = float(np.sum(np.abs(wanted)))
            if g > max_gross > 0:
                wanted = wanted * (max_gross / g)
        diff = wanted - w_cur
        b = band_by_day[d]
        trade = np.abs(diff) > (b * np.abs(wanted) if band_relative else b)
        if mask is not None and (rebalance is None or not rebalance[d]):
            trade = trade & mask[d]
        elif rebalance is not None and not rebalance[d]:
            trade = np.zeros(n_sym, dtype=bool)
        w_new = np.where(trade, wanted, w_cur)

        # asgari emir / adım yuvarlaması (yalnızca limit verildiyse)
        if min_n is not None or step is not None:
            account_nav = account_size * nav
            settled = np.where(trade, w_cur, w_new)  # brüt kontrolü: işlenmiş semboller + henüz işlenmemişlerin MEVCUT ağırlığı
            order = np.flatnonzero(trade)
            if lot_rounding == "nearest":  # iki geçiş: önce küçültme/kapanışlar (brütü düşürür), sonra artışlar (borsadaki sıra)
                order = sorted(order, key=lambda k: abs(w_new[k]) > abs(w_cur[k]))
            for j in order:
                target_notional = abs(w_new[j]) * account_nav
                if lot_rounding == "nearest":
                    target_w = w_new[j]
                    w_new[j] = _round_to_lot(j, target_w, w_cur, settled, target_notional, px[d, j], account_nav, min_n, step,
                                             max_gross, skipped, index[d], columns[j])
                    settled[j] = w_new[j]
                    if target_w != 0 and w_new[j] != w_cur[j]:
                        lot_target += abs(target_w)
                        lot_final += abs(w_new[j])
                    continue
                if w_new[j] != 0 and min_n is not None and np.isfinite(min_n[j]) and target_notional < min_n[j]:
                    skipped.append({"date": index[d], "symbol": columns[j], "reason": "min_notional", "notional": float(target_notional), "min_notional": float(min_n[j])})
                    w_new[j] = w_cur[j]
                    continue
                if step is not None and np.isfinite(step[j]) and step[j] > 0 and np.isfinite(px[d, j]) and w_new[j] != 0:
                    qty = np.floor(target_notional / px[d, j] / step[j]) * step[j]
                    w_new[j] = np.sign(w_new[j]) * qty * px[d, j] / account_nav
                    if qty == 0:
                        skipped.append({"date": index[d], "symbol": columns[j], "reason": "amount_step", "notional": float(target_notional), "min_notional": None})

        delta = w_new - w_cur
        comm = float(np.sum(np.abs(delta) * fee[d]))
        slp = float(np.sum(np.abs(delta) * slip[d]))
        fnd = float(np.sum(w_new * fund[d]))  # long öder (w>0, oran>0 → maliyet)
        r_safe = np.where(tradable, r, 0.0)
        gross = float(np.sum(w_new * r_safe))
        day_ret = gross - comm - slp - fnd
        end_factor = 1.0 + day_ret
        # drift: pozisyon değerleri büyür, nakit maliyet/funding'le değişir
        w_prev = (w_new * (1.0 + r_safe)) / end_factor if end_factor > 0 else np.zeros(n_sym)
        w_prev = np.where(np.isfinite(w_prev), w_prev, 0.0)

        nav *= end_factor
        out_nav.append(nav)
        out_ret.append(day_ret)
        out_gross.append(gross)
        out_turn.append(float(np.sum(np.abs(delta))))
        out_trades.append(int(np.count_nonzero(np.abs(delta) > 1e-12)))
        out_comm.append(comm)
        out_slip.append(slp)
        out_fund.append(fnd)
        contrib[d] = w_new * r_safe - np.abs(delta) * (fee[d] + slip[d]) - w_new * fund[d]
        held[d] = w_new

    days = index[:-1]
    result = EngineResult(
        nav=pd.Series(out_nav, index=days, name="nav"),
        returns=pd.Series(out_ret, index=days, name="return"),
        turnover=pd.Series(out_turn, index=days, name="turnover"),
        costs=pd.DataFrame({"commission": out_comm, "slippage": out_slip, "funding": out_fund}, index=days),
        gross_returns=pd.Series(out_gross, index=days, name="gross_return"),
        symbol_contrib=pd.DataFrame(contrib, index=days, columns=columns),
        weights_held=pd.DataFrame(held, index=days, columns=columns),
        skipped=skipped,
        diagnostics={
            "missing_price_symbol_days": missing_days,
            "forced_liquidations": forced_liquidations,
            "skipped_positions": len(skipped),
            "trades_per_day": pd.Series(out_trades, index=days, name="trades", dtype="int64"),
            "delay_bars": delay_bars,
            "band": float(np.mean(band_by_day)) if isinstance(band, pd.Series) else band,
            "band_relative": band_relative,
            "lot_rounding": lot_rounding,
            "lot_scale": (lot_final / lot_target) if lot_target > 0 else float("nan"),
            "n_days": len(days),
        },
    )
    return result


def _round_to_lot(j, w, w_cur, settled, target_notional, price, account_nav, min_n, step, max_gross, skipped, day, symbol) -> float:
    """En yakın lota yuvarlama (bkz. `run(lot_rounding="nearest")`). Döner: j'nin yeni ağırlığı."""
    if w == 0:
        return 0.0  # çıkış her zaman
    if not np.isfinite(price) or price <= 0:
        skipped.append({"date": day, "symbol": symbol, "reason": "price", "notional": float(target_notional), "min_notional": None})
        return w_cur[j]
    lot = step[j] if (step is not None and np.isfinite(step[j]) and step[j] > 0) else 0.0
    mn = min_n[j] if (min_n is not None and np.isfinite(min_n[j])) else 0.0
    min_qty = mn / price
    if lot > 0:
        min_qty = max(lot, np.ceil(min_qty / lot - 1e-9) * lot)
    q_exact = target_notional / price
    if min_qty > 0 and q_exact < 0.5 * min_qty:  # en yakın lot = 0
        if w_cur[j] == 0:
            skipped.append({"date": day, "symbol": symbol, "reason": "min_lot", "notional": float(target_notional),
                            "min_notional": float(min_qty * price)})
        return 0.0  # yeni pozisyon açılmaz; mevcut pozisyon (yönü ne olursa olsun) kapatılır
    q = np.round(q_exact / lot) * lot if lot > 0 else q_exact
    q = max(q, min_qty)
    w_try = float(np.sign(w) * q * price / account_nav)
    if max_gross is not None and abs(w_try) > abs(w_cur[j]):  # yalnızca brütü ARTIRAN emirler kontrol edilir
        gross = float(np.sum(np.abs(settled)) - abs(settled[j]) + abs(w_try))
        if gross > max_gross + 1e-12:
            skipped.append({"date": day, "symbol": symbol, "reason": "leverage", "notional": float(q * price), "min_notional": None})
            return w_cur[j]
    return w_try


def cost_summary(result: EngineResult) -> dict:
    """Toplam maliyet kırılımı ve brüt kâra oranı (KURALLAR raporu için)."""
    c = result.costs.sum()
    gross = float(result.gross_returns.sum())
    total_cost = float(c.sum())
    return {
        "commission": float(c["commission"]),
        "slippage": float(c["slippage"]),
        "funding": float(c["funding"]),
        "total_cost": total_cost,
        "gross_pnl": gross,
        "cost_to_gross": (total_cost / gross) if gross > 0 else float("nan"),
    }
