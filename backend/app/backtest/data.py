import pandas as pd

from app.exchanges.base import Exchange

_TIMEFRAME_UNIT_MINUTES = {"m": 1, "h": 60, "d": 1440, "w": 10080}
_FAR_PAST_MS = int(pd.Timestamp("2017-01-01", tz="UTC").timestamp() * 1000)


def timeframe_to_minutes(timeframe: str) -> int:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit not in _TIMEFRAME_UNIT_MINUTES:
        raise ValueError(f"Bilinmeyen timeframe birimi: {timeframe}")
    return value * _TIMEFRAME_UNIT_MINUTES[unit]


def fetch_full_history(
    exchange: Exchange,
    symbol: str,
    timeframe: str,
    max_candles: int,
    batch_size: int = 1000,
    start_since_ms: int | None = None,
) -> pd.DataFrame:
    """Bir sembolün mevcut olan en eski veriden başlayarak `max_candles`'a
    kadar (veya borsada o kadar geçmiş yoksa mevcut olan tüm) OHLCV verisini
    ileriye doğru sayfalayarak çeker.

    Binance gibi borsalar tek istekte sınırlı sayıda mum döner (genelde
    ~1000-1500); bu fonksiyon `since` parametresini her turda bir sonraki
    mumun zaman damgasına ilerleterek daha uzun geçmişi birleştirir.
    """
    since = start_since_ms or _FAR_PAST_MS
    frames: list[pd.DataFrame] = []
    fetched = 0

    while fetched < max_candles:
        limit = min(batch_size, max_candles - fetched)
        batch = exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
        if batch.empty:
            break

        frames.append(batch)
        fetched += len(batch)

        if len(batch) < limit:
            break  # borsa güncel veriye yetişti, daha fazla mum yok

        last_ts_ms = int(batch["timestamp"].iloc[-1].value // 10**6)
        next_since = last_ts_ms + 1
        if next_since <= since:
            break  # ilerleme yok, sonsuz döngüyü önle
        since = next_since

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return full.iloc[:max_candles].reset_index(drop=True)
