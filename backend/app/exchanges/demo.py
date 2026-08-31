import hashlib

import numpy as np
import pandas as pd

from app.exchanges.base import Exchange

_DEMO_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "XRP/USDT:USDT",
    "ADA/USDT:USDT",
    "DOGE/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "TON/USDT:USDT",
    "DOT/USDT:USDT",
    "MATIC/USDT:USDT",
]

_TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def _seed_for(symbol: str) -> int:
    return int(hashlib.sha256(symbol.encode()).hexdigest(), 16) % (2**32)


class DemoExchange(Exchange):
    """Ağ erişimi olmadan çalışan sentetik borsa.

    Gerçek borsalara (ör. Binance) bu sandbox/ortamdan erişilemediğinde
    (ağ politikası engeli, geçici kesinti, offline geliştirme) tüm
    boru hattını (screener -> ML eğitim -> karar motoru) uçtan uca
    canlı olarak göstermek için kullanılır. Sembol başına sabit bir
    seed ile deterministik bir geometrik rastgele yürüyüş (GBM benzeri)
    fiyat serisi üretir; gerçek piyasa verisi DEĞİLDİR ve gerçek emir
    verme/hesap işlemlerinde ASLA kullanılmaz (yalnızca `get_exchange`
    üzerinden, salt-okunur piyasa verisi arayüzüne bağlıdır).
    """

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        return list(_DEMO_SYMBOLS)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        minutes = _TIMEFRAME_MINUTES.get(timeframe, 60)
        rng = np.random.default_rng(_seed_for(symbol))

        drift = rng.uniform(-0.00005, 0.00005)
        volatility = rng.uniform(0.003, 0.012)
        start_price = rng.uniform(1.0, 60000.0)

        n = limit
        returns = rng.normal(drift, volatility, size=n)
        close = start_price * np.exp(np.cumsum(returns))

        open_ = np.empty(n)
        open_[0] = start_price
        open_[1:] = close[:-1]

        intrabar_noise = rng.uniform(0.0005, 0.004, size=n)
        high = np.maximum(open_, close) * (1 + intrabar_noise)
        low = np.minimum(open_, close) * (1 - intrabar_noise)
        volume = rng.uniform(10, 5000, size=n)

        end_ms = since if since is not None else 0
        period_ms = minutes * 60_000
        if since is not None:
            timestamps = since + np.arange(n) * period_ms
        else:
            now_ms = pd.Timestamp.now("UTC").value // 1_000_000
            timestamps = now_ms - (n - np.arange(n) - 1) * period_ms

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
