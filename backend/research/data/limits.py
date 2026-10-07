"""Sembol limitleri (min notional, adım): `app.exchanges.binance.fetch_market_limits` yeniden kullanılır, JSON'a önbelleğe alınır."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .. import config


def _cache_file() -> Path:
    return Path(config.CACHE_DIR) / "limits.json"


def to_ccxt_symbol(symbol: str) -> str:
    """BTCUSDT -> BTC/USDT:USDT (USDT-M perpetual)."""
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    return f"{base}/USDT:USDT"


def _default_fetcher() -> Callable[[str], dict | None]:
    from app.exchanges import (
        get_exchange,  # yeniden kullanım: kimlik doğrulamasız, GERÇEK piyasa verisi
    )

    exchange = get_exchange("binance")
    return lambda ccxt_symbol: exchange.fetch_market_limits(ccxt_symbol, "future")


class LimitsUnavailableError(RuntimeError):
    """Limitler okunamadı (ağ/borsa hatası). Sessizce 'limitsiz' koşmak küçük hesap kısıtını gizlerdi: koşu durur."""


def _is_unknown_symbol(exc: Exception) -> bool:
    """ccxt `BadSymbol` (ve alt sınıfları): sembol borsada YOK (delist) -> kısıtsız varsayılır. Diğer her hata geçicidir."""
    return any(c.__name__ == "BadSymbol" for c in type(exc).__mro__)


def get_limits(symbols: list[str], fetcher: Callable[[str], dict | None] | None = None, refresh: bool = False) -> dict[str, dict | None]:
    """{sembol: {amount_step, amount_min, price_tick, cost_min} | None}. None = borsada yok (delist; motor kısıtsız varsayar,
    çağıran raporlar). Ağ/borsa hatası önbelleğe YAZILMAZ ve `LimitsUnavailableError` fırlatılır (bir sonraki koşu yeniden dener)."""
    path = _cache_file()
    cached: dict = {}
    if path.exists() and not refresh:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            cached = {}
    missing = [s for s in symbols if s not in cached]
    if missing:
        fetch = fetcher or _default_fetcher()
        failed: dict[str, str] = {}
        for s in missing:
            try:
                cached[s] = fetch(to_ccxt_symbol(s))
            except Exception as exc:  # noqa: BLE001 - delist ile geçici hatayı ayır
                if _is_unknown_symbol(exc):
                    cached[s] = None
                else:  # ağ/borsa hatası: kalan sembolleri tek tek deneyip beklemek yerine hemen dur
                    failed[s] = f"{type(exc).__name__}: {exc}"
                    break
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached, sort_keys=True), encoding="utf-8")  # yalnızca başarılı/delist sonuçlar
        if failed:
            sample = "; ".join(f"{k} ({v})" for k, v in list(failed.items())[:3])
            raise LimitsUnavailableError(f"{len(failed)} sembolün limiti okunamadı (önbelleğe yazılmadı; yeniden deneyin): {sample}")
    return {s: cached.get(s) for s in symbols}


def engine_limits(limits: dict[str, dict | None]) -> tuple[pd.Series, pd.Series, list[str]]:
    """(min_notional, amount_step, limitsiz_semboller) — `research.engine.run` girdisi."""
    min_notional = pd.Series({s: (v or {}).get("cost_min") for s, v in limits.items()}, dtype=float)
    step = pd.Series({s: (v or {}).get("amount_step") for s, v in limits.items()}, dtype=float)
    unknown = sorted(s for s, v in limits.items() if not v)
    return min_notional, step, unknown
