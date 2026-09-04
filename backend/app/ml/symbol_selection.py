import logging

import numpy as np
import pandas as pd

from app.core.config import settings
from app.exchanges.base import Exchange

logger = logging.getLogger(__name__)

# Uyumluluk/likidite kontrolü için kullanılan kısa, ucuz pencere — asıl
# eğitim veri setinin `ml_train_lookback` (10.000) derinliğinden BİLİNÇLİ
# OLARAK farklı ve çok daha küçük: burada amaç yalnızca "bu sembol BTC ile
# aynı rejimde mi hareket ediyor ve yeterince likit mi" sorusuna hızlıca
# cevap vermek, tüm adaylar için tam derinlikte veri çekmek gereksiz
# ağ/CPU maliyeti olurdu.
_PROBE_TIMEFRAME = "1h"
_PROBE_LOOKBACK = 100


def _quote_volume(ohlcv: pd.DataFrame) -> float:
    return float((ohlcv["close"] * ohlcv["volume"]).mean())


def _correlation_with_primary(primary_returns: pd.Series, candidate_ohlcv: pd.DataFrame) -> float:
    candidate_returns = candidate_ohlcv["close"].pct_change().dropna()
    n = min(len(primary_returns), len(candidate_returns))
    if n < 20:
        return float("nan")
    corr = np.corrcoef(primary_returns.tail(n), candidate_returns.tail(n))[0, 1]
    return float(corr) if np.isfinite(corr) else float("nan")


def select_training_symbols(
    exchange: Exchange,
    candidates: list[str],
    max_symbols: int | None = None,
) -> list[str]:
    """BTC-öncelikli eğitim sembol seçimi.

    Her zaman `settings.ml_primary_symbol` (varsayılan BTC/USDT) ilk sırada
    döner. `candidates` listesindeki diğer semboller yalnızca şu ikisini
    SAĞLARSA ek olarak eğitime katılır:

    1. Likidite: ortalama (close * volume) >= `ml_min_quote_volume_24h`.
    2. Uyumluluk: BTC ile saatlik getiri korelasyonu
       >= `ml_min_correlation_with_primary` (ince/manipüle edilebilir ya da
       BTC'den tamamen bağımsız hareket eden semboller elenir — bunları
       BTC üzerinde öğrenilmiş bir modelin veri setine katmak "zayıf
       seçilmiş bir grup" ile eğitim riskini taşır).

    Kalan adaylar korelasyona göre azalan sırada sıralanır ve
    `max_symbols` (varsayılan `settings.ml_train_max_symbols`) sınırına
    kadar (BTC dahil) alınır.
    """
    primary = settings.ml_primary_symbol
    limit = max_symbols if max_symbols is not None else settings.ml_train_max_symbols

    try:
        primary_ohlcv = exchange.fetch_ohlcv(primary, _PROBE_TIMEFRAME, _PROBE_LOOKBACK)
        primary_returns = primary_ohlcv["close"].pct_change().dropna()
    except Exception as exc:  # noqa: BLE001 - BTC verisi alınamazsa bile eğitim BTC sembolüyle devam edebilmeli
        logger.warning("symbol_selection: primary sembol (%s) verisi alınamadı: %s", primary, exc)
        primary_returns = pd.Series(dtype=float)

    scored: list[tuple[str, float]] = []
    for symbol in candidates:
        if symbol == primary:
            continue
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, _PROBE_TIMEFRAME, _PROBE_LOOKBACK)
            if len(ohlcv) < 20:
                continue
            if _quote_volume(ohlcv) < settings.ml_min_quote_volume_24h:
                continue
            if primary_returns.empty:
                continue
            corr = _correlation_with_primary(primary_returns, ohlcv)
            if not np.isfinite(corr) or corr < settings.ml_min_correlation_with_primary:
                continue
            scored.append((symbol, corr))
        except Exception as exc:  # noqa: BLE001 - tek sembol hatası tüm seçimi durdurmamalı
            logger.warning("symbol_selection: skipping %s: %s", symbol, exc)
            continue

    scored.sort(key=lambda item: item[1], reverse=True)
    selected = [primary] + [symbol for symbol, _corr in scored[: max(limit - 1, 0)]]
    return selected
