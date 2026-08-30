import logging

from app.core.config import settings
from app.exchanges.base import Exchange

from .indicators import compute_indicators, composite_score
from .schemas import ScreenerResult

logger = logging.getLogger(__name__)


def scan_market(exchange: Exchange) -> list[ScreenerResult]:
    """Borsadaki tüm sembolleri tarayıp her biri için yön skoru üretir.

    Sonuç, çağıran tarafından skor sırasına göre Long/Short Top N'e
    bölünebilecek şekilde ham liste olarak döner.
    """
    symbols = exchange.list_symbols(settings.quote_currency, settings.market_type)
    results: list[ScreenerResult] = []

    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, settings.candle_timeframe, settings.candle_lookback)
            if len(ohlcv) < 60:
                continue
            indicators = compute_indicators(ohlcv)
            last = indicators.iloc[-1]
            score = composite_score(indicators)
            results.append(
                ScreenerResult(
                    symbol=symbol,
                    score=score,
                    close=float(last["close"]),
                    rsi=float(last["rsi"]),
                    trend="up" if last["ema_fast"] > last["ema_slow"] else "down",
                )
            )
        except Exception as exc:  # noqa: BLE001 - tek sembol hatası taramayı durdurmamalı
            logger.warning("skipping %s: %s", symbol, exc)
            continue

    return results


def top_long(results: list[ScreenerResult], limit: int) -> list[ScreenerResult]:
    return sorted(results, key=lambda r: r.score, reverse=True)[:limit]


def top_short(results: list[ScreenerResult], limit: int) -> list[ScreenerResult]:
    return sorted(results, key=lambda r: r.score)[:limit]
