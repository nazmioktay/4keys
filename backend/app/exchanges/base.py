from abc import ABC, abstractmethod

import pandas as pd


class Exchange(ABC):
    """Borsa erişimini soyutlayan arayüz.

    Yeni bir borsa (BIST, VIOP, başka bir dünya borsası) eklemek için bu
    sınıfı implemente eden yeni bir adapter yazmak yeterlidir; screener,
    ML ve bot motorları bu arayüz üzerinden çalışır, borsaya özel kod
    içermez.
    """

    @abstractmethod
    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        """Verilen kote para birimi ve piyasa tipi için işlem gören sembolleri döner."""

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        """Bir sembol için OHLCV mum verisini DataFrame olarak döner.

        Kolonlar: timestamp, open, high, low, close, volume

        `since` verilirse (Unix ms), o zamandan itibaren ileriye doğru en
        fazla `limit` mum döner — geçmişe doğru sayfalama (bkz.
        `app.backtest.data.fetch_full_history`) bunu kullanır.
        """
