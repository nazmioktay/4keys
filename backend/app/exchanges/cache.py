import logging

import pandas as pd

from app.db import repository as db

from .base import Exchange

logger = logging.getLogger(__name__)

_TIMEFRAME_UNIT_MINUTES = {"m": 1, "h": 60, "d": 1440, "w": 10080}

# Bir mumun "bayat" sayılması için kaç bar geçmesi gerektiği — timeframe'in
# kendisinden biraz pay bırakır (borsa/işleme küçük gecikmeler için).
_FRESHNESS_TOLERANCE_BARS = 2


def _timeframe_minutes(timeframe: str) -> int:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    return value * _TIMEFRAME_UNIT_MINUTES.get(unit, 60)


def fetch_ohlcv_cached(exchange: Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """`Exchange.fetch_ohlcv`'in ÖNBELLEKLİ hali: `ohlcv_raw` tablosunda
    önceden kaydedilmiş mumlar varsa borsaya HİÇ gitmez (veya yalnızca
    EKSİK/YENİ kuyruğu, sırayla tamamlayarak çeker) — önceden her eğitim/
    karar döngüsü, `ml_train_lookback` (varsayılan 10.000) mumu HER
    SEFERİNDE baştan borsadan çekiyordu; bu hem gereksiz ağ/CPU maliyeti
    hem de borsanın rate limitine çarpma riski taşıyordu (bkz. screener
    taramasının aynı sorundan etkilenmesi).

    DB kapalıysa (`FOURKEYS_DATABASE_URL` boş) doğrudan `exchange.fetch_ohlcv`'e
    düşer — bu katman tamamen opsiyoneldir, DB olmadan da sistem çalışır.
    """
    if not db.is_enabled():
        return exchange.fetch_ohlcv(symbol, timeframe, limit)

    cached = db.get_ohlcv(symbol, timeframe, limit)

    if cached.empty:
        fresh = exchange.fetch_ohlcv(symbol, timeframe, limit)
        db.save_ohlcv_bulk(symbol, timeframe, fresh)
        return fresh

    last_ts = pd.Timestamp(cached["timestamp"].iloc[-1])
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    now = pd.Timestamp.now(tz="UTC")
    timeframe_minutes = _timeframe_minutes(timeframe)
    is_stale = (now - last_ts).total_seconds() > _FRESHNESS_TOLERANCE_BARS * timeframe_minutes * 60

    if len(cached) >= limit and not is_stale:
        return cached.iloc[-limit:].reset_index(drop=True)

    # Yalnızca son kaydedilen mumdan SONRAKİ (eksik) kuyruğu çek — sıra ile,
    # tamamlayarak: `since` son önbellek zaman damgasının hemen ardından
    # başlar, `exchange.fetch_ohlcv` (bkz. BinanceExchange) `limit > 1000`
    # olduğunda bu noktadan borsanın izin verdiği kadar ileriye doğru
    # kendi içinde sayfalar.
    since_ms = int(last_ts.timestamp() * 1000) + 1
    try:
        fresh = exchange.fetch_ohlcv(symbol, timeframe, limit, since=since_ms)
    except Exception:  # noqa: BLE001 - borsa erişilemezse, en azından ELİMİZDEKİ önbellekle devam edilebilir
        logger.warning("fetch_ohlcv_cached: %s için yeni kuyruk çekilemedi, önbellekle devam ediliyor", symbol)
        fresh = pd.DataFrame(columns=cached.columns)

    if not fresh.empty:
        db.save_ohlcv_bulk(symbol, timeframe, fresh)
        combined = pd.concat([cached, fresh], ignore_index=True)
        combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    else:
        combined = cached

    if len(combined) < limit:
        # Önbellek + yeni kuyruk toplamı hâlâ istenenden az — DB muhtemelen
        # soğuk/kısmi (ör. ilk kurulum) — tam geçmişi bir kez borsadan çekip
        # DB'yi bu vesileyle tamamen doldur.
        full = exchange.fetch_ohlcv(symbol, timeframe, limit)
        if not full.empty:
            db.save_ohlcv_bulk(symbol, timeframe, full)
        return full

    return combined.iloc[-limit:].reset_index(drop=True)
