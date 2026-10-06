"""Sinyal adaptörleri: model tahmini -> kol (arm) hedef ağırlığı (NAV oranı, işaretli). Karar t kapanışında; motor t+1
açılışında doldurur. Üyelik dışı (evren dışı) semboller her zaman 0 ağırlık alır.

    vol_target        vol tahmini -> volatilite hedefleme (long-only, eşit risk): w_i = min(cap, hedef_günlük_vol/σ̂_i) / N_üye
    rank_long_short   sıralama -> kesitsel ağırlık: üst q uzun, alt q kısa; her bacak 0,5 brüt (dolar-nötr, brüt 1) — long_only=True ise üst q uzun, brüt 1
    prob_filter       olasılık -> filtre: p > eşik olan üyeler eşit ağırlıkla uzun (taban yön +1)
    prob_size         olasılık -> boyut: w_i = clip(2(p−0,5), 0, 1) / N_üye
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


ADAPTERS: dict[str, Callable[..., pd.DataFrame]] = {
    "vol_target": vol_target,
    "rank_long_short": rank_long_short,
    "prob_filter": prob_filter,
    "prob_size": prob_size,
}


def make_weights(adapter: str, pred_wide: pd.DataFrame, membership: pd.DataFrame, **params) -> pd.DataFrame:
    if adapter not in ADAPTERS:
        raise KeyError(f"Bilinmeyen sinyal adaptörü '{adapter}'. Kayıtlı: {sorted(ADAPTERS)}")
    return ADAPTERS[adapter](pred_wide, membership, **params)
