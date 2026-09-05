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

    def fetch_tickers(self, quote_currency: str, market_type: str) -> dict[str, dict]:
        """TÜM sembollerin güncel fiyat/24s işlem hacmini TEK bir toplu
        (bulk) istekle döner — `{symbol: {"last": float, "quote_volume": float}}`.

        `list_symbols` + her sembol için ayrı `fetch_ohlcv` çağırmanın
        AKSİNE (yüzlerce sembolde bu, borsanın rate limitine çarpıp
        taramanın kendi periyodundan uzun sürmesine/asla bitmemesine yol
        açabiliyordu — bkz. `app.screener.scanner.scan_market`), bu tek
        istekle ucuz bir ön-filtre (hacim sıralaması + fiyat tabanı)
        yapılabilir; pahalı `fetch_ohlcv`+gösterge hesaplaması yalnızca bu
        filtreyi geçen KÜÇÜK bir alt kümede çalıştırılır.

        Varsayılan (geriye dönük uyumlu) uygulama: toplu ticker desteği
        olmayan/henüz eklenmemiş borsalar için `list_symbols`'u sarar.
        `last=inf` döner (fiyat tabanı filtresini HER ZAMAN geçer — 0.0
        dönseydi, `screener_min_price`'ın altında kalıp TÜM semboller
        yanlışlıkla elenirdi); `quote_volume=0.0` (hepsi eşit, sıralama
        etkisizleşir — yalnızca `screener_volume_top_pct` oranı kadarı,
        orijinal sırayla tutulur). Gerçek toplu veri sağlayabilen borsalar
        (ör. Binance) bunu override eder.
        """
        return {
            symbol: {"last": float("inf"), "quote_volume": 0.0}
            for symbol in self.list_symbols(quote_currency, market_type)
        }

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        """Bir sembol için OHLCV mum verisini DataFrame olarak döner.

        Kolonlar: timestamp, open, high, low, close, volume

        `since` verilirse (Unix ms), o zamandan itibaren ileriye doğru en
        fazla `limit` mum döner — geçmişe doğru sayfalama (bkz.
        `app.backtest.data.fetch_full_history`) bunu kullanır.
        """

    def fetch_open_interest(self, symbol: str) -> dict | None:
        """Perpetual futures'a özgü açık pozisyon (open interest) verisinin
        ŞU ANKİ anlık görüntüsünü döner: `{"open_interest": float,
        "open_interest_value": float}` (miktar + notional/USDT değeri).

        `app.ml.orderbook_features` ile AYNI mantık: borsalar geçmişe dönük
        open interest saklamaz/satmaz, bu yüzden bu değer yalnızca
        toplamaya BAŞLADIĞIMIZ andan itibaren (bkz. `app.openinterest`)
        birikir. Varsayılan (geriye dönük uyumlu) uygulama `None` döner —
        yalnızca gerçek destek sağlayan borsalar (ör. Binance futures)
        bunu override eder; spot sembollerde de anlamsızdır."""
        return None
