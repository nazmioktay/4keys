"""Küçük hesap için varlık başına tavan (KURALLAR.md §3; kullanıcı kararı 2026-10-07: BTC evrenden çıkarılmaz, tavanı en küçük
lota göre ayarlanır).

Varlık başına tavan kullanan tasarımlar `asset_caps` + `clip_to_caps` kullanır: tavan_t = max(base_cap, bir lotun notional'ı_t / NAV).
Böylece BTC gibi büyük lotlu bir sembol, sinyal tavana dayandığında en az BİR LOT alabilir (motor `lot_rounding="nearest"` ile
yuvarlar; brüt kaldıraç tavanı ayrıca geçerlidir). Yalnızca t günündeki fiyat kullanılır (ileri bakış yok)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def min_lot_notional(price: pd.DataFrame, min_notional: pd.Series | None, amount_step: pd.Series | None) -> pd.DataFrame:
    """Tarih × sembol: o günün fiyatıyla açılabilecek EN KÜÇÜK pozisyonun notional'ı = max(min notional, 1 adım), adıma yukarı."""
    cols = price.columns
    mn = (min_notional.reindex(cols) if min_notional is not None else pd.Series(np.nan, index=cols)).astype(float).fillna(0.0)
    lot = (amount_step.reindex(cols) if amount_step is not None else pd.Series(np.nan, index=cols)).astype(float)
    qty = (1.0 / price).mul(mn, axis=1)  # min notional / fiyat
    has_lot = lot.notna() & (lot > 0)
    lot_v = lot.where(has_lot, 1.0)
    qty_lots = np.ceil(qty.div(lot_v, axis=1) - 1e-9).clip(lower=1.0).mul(lot_v, axis=1)
    qty = qty_lots.where(np.broadcast_to(has_lot.to_numpy(), qty.shape), qty)
    return (qty * price).where(price > 0)


def asset_caps(price: pd.DataFrame, nav: float, min_notional: pd.Series | None, amount_step: pd.Series | None,
               base_cap: float = config.ONE_X_ASSET_CAP) -> pd.DataFrame:
    """Tarih × sembol varlık tavanı (NAV oranı) = max(base_cap, en küçük lot notional'ı / NAV). Fiyat yoksa base_cap."""
    lot_frac = min_lot_notional(price, min_notional, amount_step) / float(nav)
    return lot_frac.clip(lower=base_cap).fillna(base_cap)


def clip_to_caps(weights: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    """|w| <= tavan (işaret korunur)."""
    c = caps.reindex_like(weights).fillna(config.ONE_X_ASSET_CAP)
    return weights.clip(lower=-c, upper=c)
