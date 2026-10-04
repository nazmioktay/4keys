import logging

import pandas as pd

from app.db import repository as db

from .base import Exchange

logger = logging.getLogger(__name__)

_TIMEFRAME_UNIT_MINUTES = {"m": 1, "h": 60, "d": 1440, "w": 10080}

# Her çağrıda önbellekteki son kaç barın borsadan YENİDEN çekilip üzerine
# yazılacağı. Eski önbellek, bir mumu henüz oluşurken yakalayıp kalıcı
# kaydediyordu (ON CONFLICT DO NOTHING) ve son mum 2 bar "taze" sayıldığı
# için borsaya hiç gitmiyordu — canlı karar motoru 1-2 saatlik donuk/yarım
# veriyle karar veriyor, stop-loss'lar seviyenin çok ötesinde tetikleniyordu.
_REFETCH_TAIL_BARS = 3


def timeframe_minutes(timeframe: str) -> int:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    return value * _TIMEFRAME_UNIT_MINUTES.get(unit, 60)


def _utc_now_naive() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_convert(None)


def drop_unclosed_bars(ohlcv: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Henüz kapanmamış (açılış zamanı + timeframe > şimdi) mumları atar.

    Eğitim etiketleri ve backtest yalnızca kapanmış mumlar üzerinden
    hesaplanır; canlı karar da aynı kuralı kullanmalı, yoksa model eğitimde
    hiç görmediği yarım-mum özellikleriyle tahmin üretir."""
    if ohlcv.empty:
        return ohlcv
    now = _utc_now_naive() if now is None else now
    timestamps = pd.to_datetime(ohlcv["timestamp"])
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert(None)
    cutoff = now - pd.Timedelta(minutes=timeframe_minutes(timeframe))
    return ohlcv.loc[timestamps <= cutoff].reset_index(drop=True)


def fetch_ohlcv_cached(exchange: Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """`Exchange.fetch_ohlcv`'in ÖNBELLEKLİ hali: `ohlcv_raw` tablosundaki
    geçmiş okunur, borsadan yalnızca son `_REFETCH_TAIL_BARS` bar ve sonrası
    çekilip upsert edilir. Dönen veri YALNIZCA kapanmış mumları içerir.

    DB kapalıysa (`FOURKEYS_DATABASE_URL` boş) doğrudan `exchange.fetch_ohlcv`'e
    düşer — bu katman tamamen opsiyoneldir, DB olmadan da sistem çalışır.
    """
    if not db.is_enabled():
        return drop_unclosed_bars(exchange.fetch_ohlcv(symbol, timeframe, limit + 1), timeframe).iloc[-limit:].reset_index(
            drop=True
        )

    cached = db.get_ohlcv(symbol, timeframe, limit)

    if cached.empty:
        return _fetch_full(exchange, symbol, timeframe, limit)

    bar_minutes = timeframe_minutes(timeframe)
    tail_start_index = max(0, len(cached) - _REFETCH_TAIL_BARS)
    tail_start = pd.Timestamp(cached["timestamp"].iloc[tail_start_index])
    if tail_start.tzinfo is not None:
        tail_start = tail_start.tz_convert(None)
    since_ms = int(tail_start.tz_localize("UTC").timestamp() * 1000)
    bars_needed = int((_utc_now_naive() - tail_start) / pd.Timedelta(minutes=bar_minutes)) + 2
    try:
        fresh = exchange.fetch_ohlcv(symbol, timeframe, min(max(bars_needed, 1), max(limit, 1)), since=since_ms)
    except Exception:  # noqa: BLE001 - borsa erişilemezse, en azından ELİMİZDEKİ önbellekle devam edilebilir
        logger.warning("fetch_ohlcv_cached: %s için yeni kuyruk çekilemedi, önbellekle devam ediliyor", symbol)
        return drop_unclosed_bars(cached, timeframe).iloc[-limit:].reset_index(drop=True)

    fresh = drop_unclosed_bars(fresh, timeframe)
    if not fresh.empty:
        db.save_ohlcv_bulk(symbol, timeframe, fresh)
        kept = cached.loc[~pd.to_datetime(cached["timestamp"]).isin(pd.to_datetime(fresh["timestamp"]))]
        combined = pd.concat([kept, fresh], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    else:
        combined = cached

    combined = drop_unclosed_bars(combined, timeframe)
    if len(combined) < limit:
        # Önbellek + yeni kuyruk toplamı hâlâ istenenden az — DB muhtemelen
        # soğuk/kısmi (ör. ilk kurulum) — tam geçmişi bir kez borsadan çekip
        # DB'yi bu vesileyle tamamen doldur.
        return _fetch_full(exchange, symbol, timeframe, limit)

    return combined.iloc[-limit:].reset_index(drop=True)


def _fetch_full(exchange: Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    full = drop_unclosed_bars(exchange.fetch_ohlcv(symbol, timeframe, limit + 1), timeframe)
    if not full.empty:
        db.save_ohlcv_bulk(symbol, timeframe, full)
    return full.iloc[-limit:].reset_index(drop=True)


def repair_ohlcv_cache(exchange: Exchange, symbol: str, timeframe: str, limit: int) -> int:
    """Son `limit` kapanmış mumu borsadan baştan çekip önbelleğin üzerine
    yazar. Eski önbellek yarım (henüz oluşurken yakalanmış) mumları kalıcı
    kaydettiği için geçmişe yayılmış bozuk satırları onarmak içindir."""
    full = drop_unclosed_bars(exchange.fetch_ohlcv(symbol, timeframe, limit), timeframe)
    return db.save_ohlcv_bulk(symbol, timeframe, full)
