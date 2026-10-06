"""Noktasal-zamanlı (point-in-time) evren: `date` günü için YALNIZCA `date`'ten ÖNCEKİ veriyle."""

from __future__ import annotations

import re

import pandas as pd

from ..config import INDEX_SYMBOLS, STABLE_BASES

_LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def base_asset(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def is_basic_eligible(symbol: str) -> bool:
    """Yalnızca sembol adına bakan elemeler: USDT quote, teslimli kontrat değil, stablecoin/endeks değil."""
    if not symbol.endswith("USDT") or "_" in symbol:
        return False
    if symbol in INDEX_SYMBOLS or re.search(r"INDEXUSDT$", symbol):
        return False
    return base_asset(symbol) not in STABLE_BASES


def _leveraged_tokens(symbols: set[str]) -> set[str]:
    """`<TABAN><UP|DOWN|BULL|BEAR>USDT` ve `<TABAN>USDT` birlikte listeliyse kaldıraçlı tokendir."""
    out = set()
    for s in symbols:
        base = base_asset(s)
        for suf in _LEVERAGED_SUFFIXES:
            if base.endswith(suf) and (base[: -len(suf)] + "USDT") in symbols:
                out.add(s)
    return out


def universe_at(
    date,
    n: int,
    quote_volume: pd.DataFrame,
    window_days: int = 30,
    min_history_days: int = 90,
    min_window_obs: int = 10,
) -> list[str]:
    """`date` günü işlem görecek ilk `n` perpetual (noktasal-zamanlı).

    - Yalnızca `index < date` verisi kullanılır (karar günün kapanışında verilir; o günün hacmi bilinmez).
    - Sıralama: son `window_days` gün ortalama quote-volume (en az `min_window_obs` gözlem).
    - Şart: sembolün ilk verisi en az `min_history_days` gün eski.
    - Hariç: stablecoin, kaldıraçlı token, endeks, USDT dışı quote, teslimli kontrat.
    `quote_volume`: tarih × sembol panel (eksik gün = NaN)."""
    date = pd.Timestamp(date)
    past = quote_volume[quote_volume.index < date]  # GELECEK BİLGİSİ YOK
    if past.empty:
        return []
    candidates = {s for s in past.columns if is_basic_eligible(s)}
    candidates -= _leveraged_tokens(set(past.columns))
    window = past[past.index >= date - pd.Timedelta(days=window_days)]
    first_seen = past.apply(lambda col: col.first_valid_index())
    rows = []
    for s in candidates:
        first = first_seen.get(s)
        if first is None or pd.isna(first) or (date - first).days < min_history_days:
            continue
        obs = window[s].dropna()
        if len(obs) < min_window_obs:
            continue
        rows.append((s, float(obs.mean())))
    rows.sort(key=lambda t: (-t[1], t[0]))
    return [s for s, _ in rows[:n]]


def ever_in_top_n(quote_volume: pd.DataFrame, n: int, step_days: int = 30, **kwargs) -> list[str]:
    """Tarih boyunca (her `step_days` günde bir) top-N'e hiç girmiş semboller — 1h indirme kapsamı için."""
    seen: set[str] = set()
    dates = quote_volume.index[quote_volume.index >= quote_volume.index.min() + pd.Timedelta(days=kwargs.get("min_history_days", 90))]
    for d in dates[::step_days]:
        seen.update(universe_at(d, n, quote_volume, **kwargs))
    return sorted(seen)


def slippage_tier_symbols(date, quote_volume: pd.DataFrame, top: int = 20) -> set[str]:
    """O günkü ilk `top` sembol (kayma kademesi; noktasal-zamanlı)."""
    return set(universe_at(date, top, quote_volume))
