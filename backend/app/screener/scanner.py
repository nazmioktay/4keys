import logging
import math

from app.core.config import settings
from app.db import repository as db
from app.exchanges.base import Exchange

from app.ml.features import build_features

from .indicators import compute_indicators, composite_score
from .schemas import ScreenerResult

logger = logging.getLogger(__name__)


def _select_candidate_symbols(exchange: Exchange) -> list[str]:
    """Pahalı `fetch_ohlcv`+gösterge taramasına girecek ADAY sembolleri
    ucuz bir ön-filtreyle daraltır — bkz. `Exchange.fetch_tickers`
    docstring'i (önceden `scan_market` PİYASADAKİ TÜM sembolleri tek tek
    tarıyordu, bu da taramanın kendi periyodundan uzun sürüp hiç
    bitmemesine yol açabiliyordu).

    1. `screener_min_price`'ın altındaki (ör. 0.1 USDT) düşük değerli
       semboller elenir.
    2. Kalanlar 24 saatlik işlem hacmine göre sıralanır, en yüksek
       `screener_volume_top_pct` (varsayılan %20) tutulur.
    """
    tickers = exchange.fetch_tickers(settings.quote_currency, settings.market_type)
    candidates = [
        (symbol, data.get("quote_volume", 0.0))
        for symbol, data in tickers.items()
        if data.get("last", 0.0) >= settings.screener_min_price
    ]
    if not candidates:
        return []

    candidates.sort(key=lambda item: item[1], reverse=True)
    keep_count = max(1, math.ceil(len(candidates) * settings.screener_volume_top_pct / 100))
    return [symbol for symbol, _volume in candidates[:keep_count]]


def scan_market(exchange: Exchange) -> list[ScreenerResult]:
    """Borsadaki (hacim/fiyat ön-filtresinden geçen) sembolleri tarayıp her
    biri için yön skoru üretir.

    Sonuç, çağıran tarafından skor sırasına göre Long/Short Top N'e
    bölünebilecek şekilde ham liste olarak döner.
    """
    symbols = _select_candidate_symbols(exchange)
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
            db.record_latest_candle(symbol, settings.candle_timeframe, last)
            if symbol in settings.feature_snapshot_symbols_list:
                ml_features = build_features(ohlcv).dropna()
                if not ml_features.empty:
                    db.record_feature_snapshot(symbol, settings.candle_timeframe, last["timestamp"], ml_features.iloc[-1])
            db.record_signal(
                symbol,
                source="screener",
                direction="long" if score > 0 else ("short" if score < 0 else "neutral"),
                confidence=min(abs(score) / 100, 1.0),
                price=float(last["close"]),
            )
        except Exception as exc:  # noqa: BLE001 - tek sembol hatası taramayı durdurmamalı
            logger.warning("skipping %s: %s", symbol, exc)
            continue

    return results


def top_long(results: list[ScreenerResult], limit: int) -> list[ScreenerResult]:
    return sorted(results, key=lambda r: r.score, reverse=True)[:limit]


def top_short(results: list[ScreenerResult], limit: int) -> list[ScreenerResult]:
    return sorted(results, key=lambda r: r.score)[:limit]
