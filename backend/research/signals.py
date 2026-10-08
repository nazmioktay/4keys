"""Sinyal adaptörleri: model tahmini -> kol (arm) hedef ağırlığı (NAV oranı, işaretli). Karar t kapanışında; motor t+1
açılışında doldurur. Üyelik dışı (evren dışı) semboller her zaman 0 ağırlık alır.

    vol_target        vol tahmini -> volatilite hedefleme (long-only, eşit risk): w_i = min(cap, hedef_günlük_vol/σ̂_i) / N_üye
    rank_long_short   sıralama -> kesitsel ağırlık: üst q uzun, alt q kısa; her bacak 0,5 brüt (dolar-nötr, brüt 1) — long_only=True ise üst q uzun, brüt 1
    prob_filter       olasılık -> filtre: p > eşik olan üyeler eşit ağırlıkla uzun (taban yön +1)
    prob_size         olasılık -> boyut: w_i = clip(2(p−0,5), 0, 1) / N_üye
    edge_threshold    σ birimli getiri tahmini -> yalnızca beklenen getiri maliyeti aşan sembollerde long/short (context ister)
Yeni adaptör = fonksiyon + `ADAPTERS`'a kayıt."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import config


def _members(pred: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    return membership.reindex(index=pred.index, columns=pred.columns).fillna(False).astype(bool)


def vol_target(pred: pd.DataFrame, membership: pd.DataFrame, target_vol_annual: float = 0.40, cap: float = 1.0) -> pd.DataFrame:
    """`pred`: tarih × sembol GÜNLÜK vol tahmini (`next_rv`). Hedef yıllık vol (ör. %40) günlüğe çevrilir."""
    m = _members(pred, membership)
    daily_target = target_vol_annual / np.sqrt(config.ANNUALIZATION_DAYS)
    lev = (daily_target / pred.where(pred > 0)).clip(upper=cap)
    lev = lev.where(m).fillna(0.0)
    n = m.sum(axis=1).replace(0, np.nan)
    return lev.div(n, axis=0).fillna(0.0)


def rank_long_short(pred: pd.DataFrame, membership: pd.DataFrame, q: float = 0.2, long_only: bool = False) -> pd.DataFrame:
    m = _members(pred, membership)
    score = pred.where(m)
    pct = score.rank(axis=1, pct=True, method="first")
    n = score.notna().sum(axis=1).replace(0, np.nan)
    top = (pct > 1 - q) & score.notna()
    bottom = (pct <= q) & score.notna()
    n_top, n_bot = top.sum(axis=1).replace(0, np.nan), bottom.sum(axis=1).replace(0, np.nan)
    if long_only:
        return top.astype(float).div(n_top, axis=0).fillna(0.0)
    w = top.astype(float).div(n_top, axis=0) * 0.5 - bottom.astype(float).div(n_bot, axis=0) * 0.5
    _ = n
    return w.fillna(0.0)


def prob_filter(pred: pd.DataFrame, membership: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    m = _members(pred, membership)
    on = (pred > threshold) & m
    n = m.sum(axis=1).replace(0, np.nan)
    return on.astype(float).div(n, axis=0).fillna(0.0)


def prob_size(pred: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    m = _members(pred, membership)
    size = (2 * (pred - 0.5)).clip(0, 1).where(m).fillna(0.0)
    n = m.sum(axis=1).replace(0, np.nan)
    return size.div(n, axis=0).fillna(0.0)


def edge_threshold(pred: pd.DataFrame, membership: pd.DataFrame, *, context: dict, h: int, k: float = 1.0,
                   target_vol_annual: float = 0.20, vol_window: int = 20, vol_span: int = 60, base_cap: float = 0.10,
                   gross_cap: float = 2.0) -> pd.DataFrame:
    """Maliyet sonrası beklenen getiri (ön kayıt `ml_kol`). `pred`: σ birimli ileri h gün getiri tahmini (`vol_adj_return`).
    Maliyet eşiği c_i = 2 × (taker ücreti + kayma_i) / (σ_i · √h); |p| > k·c_i ise yön = işaret(p), ham ağırlık işaret(p)·|p|/σ_i;
    kol hedef vol (ham portföyün geçmiş getirisinin EWMA vol'ü, yalnızca ≤ t), isim tavanı `sizing.asset_caps`, brüt tavan.
    `context`: close (tarih × sembol), slip_bps (tarih × sembol), nav, min_notional, amount_step (yoksa None)."""
    from . import sizing
    from .targets import trailing_vol

    m = _members(pred, membership)
    close = context["close"].reindex(columns=pred.columns)
    sigma = trailing_vol(close, vol_window).reindex(pred.index)
    slip = context["slip_bps"].reindex(index=pred.index, columns=pred.columns) / 1e4
    cost = 2.0 * (config.FUTURES_TAKER_FEE + slip) / (sigma * np.sqrt(h))
    active = m & pred.notna() & sigma.gt(0) & (pred.abs() > k * cost)
    raw = (np.sign(pred) * pred.abs() / sigma).where(active).fillna(0.0)
    simple = close.pct_change(fill_method=None).reindex(pred.index).fillna(0.0)
    port = (raw.shift(1).fillna(0.0) * simple).sum(axis=1)
    # ısınma: ilk geçerli tahminden ÖNCEKİ günler (tahmin yok -> getiri 0) vol tahminine girmez; aksi halde ilk haftalarda vol düşük
    # görünür ve ölçek şişer. Sonuç: OOF'un ilk `vol_span` gününde pozisyon yok.
    port = port.where(pred.notna().any(axis=1).cummax())
    port_vol = port.ewm(span=vol_span, min_periods=vol_span).std() * np.sqrt(config.ANNUALIZATION_DAYS)
    scale = (target_vol_annual / port_vol.clip(lower=1e-4)).where(port_vol.notna(), 0.0)
    w = raw.mul(scale, axis=0)
    caps = sizing.asset_caps(close.reindex(pred.index), float(context["nav"]), context.get("min_notional"), context.get("amount_step"),
                             base_cap=base_cap)
    w = sizing.clip_to_caps(w, caps)
    gross = w.abs().sum(axis=1)
    return w.mul((gross_cap / gross).clip(upper=1.0).where(gross > 0, 1.0), axis=0)


edge_threshold.needs_context = True

ADAPTERS: dict[str, Callable[..., pd.DataFrame]] = {
    "vol_target": vol_target,
    "rank_long_short": rank_long_short,
    "prob_filter": prob_filter,
    "prob_size": prob_size,
    "edge_threshold": edge_threshold,
}


def make_weights(adapter: str, pred_wide: pd.DataFrame, membership: pd.DataFrame, context: dict | None = None, **params) -> pd.DataFrame:
    """`context` yalnızca `needs_context` işaretli adaptörlere geçer (fiyat, kayma kademesi, hesap limitleri)."""
    if adapter not in ADAPTERS:
        raise KeyError(f"Bilinmeyen sinyal adaptörü '{adapter}'. Kayıtlı: {sorted(ADAPTERS)}")
    fn = ADAPTERS[adapter]
    if getattr(fn, "needs_context", False):
        if context is None:
            raise ValueError(f"'{adapter}' adaptörü context (fiyat/kayma/hesap) ister")
        return fn(pred_wide, membership, context=context, **params)
    return fn(pred_wide, membership, **params)
