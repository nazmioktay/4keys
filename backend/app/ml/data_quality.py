import logging

import pandas as pd

logger = logging.getLogger(__name__)


def count_missing_candles(ohlcv: pd.DataFrame, timeframe_minutes: int) -> int:
    """Bir OHLCV serisindeki (kronolojik sıralı) zaman damgaları arasında,
    beklenen aralıktan (`timeframe_minutes`) daha büyük boşluklardan
    kaynaklanan YAKLAŞIK kayıp mum sayısını döner.

    Kesin bir sayı değildir (borsa/DB kaynaklı küçük zaman damgası
    sapmaları toleranslı yuvarlanır), ama "bu sembolün geçmişinde ciddi
    bir kesinti var mı" sorusuna hızlı bir cevap verir — eğitim verisinin
    (özellikle DB önbelleğinden okunanların, bkz. `app.exchanges.cache`)
    sessizce eksik/parçalı olmadığından emin olmak için.
    """
    if len(ohlcv) < 2:
        return 0
    ts = pd.to_datetime(ohlcv["timestamp"])
    diffs_minutes = ts.diff().dropna().dt.total_seconds() / 60
    if timeframe_minutes <= 0:
        return 0
    ratios = (diffs_minutes / timeframe_minutes).round()
    missing = (ratios - 1).clip(lower=0).sum()
    return int(missing)


def warn_if_gaps(symbol: str, timeframe: str, ohlcv: pd.DataFrame, timeframe_minutes: int) -> int:
    """`count_missing_candles`'ı çalıştırır, anlamlı bir boşluk varsa
    (satır sayısının >%1'i) log'a uyarı yazar. Eğitim akışını KESMEZ —
    yalnızca gözlemlenebilirlik içindir (bkz. çağıranların log'ları)."""
    missing = count_missing_candles(ohlcv, timeframe_minutes)
    if missing > 0 and missing > max(1, len(ohlcv) * 0.01):
        logger.warning(
            "data_quality: %s (%s) serisinde ~%d kayıp mum tespit edildi (%d satır üzerinden) — "
            "eğitim verisi kesintili/parçalı olabilir",
            symbol,
            timeframe,
            missing,
            len(ohlcv),
        )
    return missing
