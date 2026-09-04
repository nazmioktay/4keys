import ccxt
import pandas as pd

from .base import Exchange


class BinanceExchange(Exchange):
    """Binance adapter'ı.

    İki kullanım modu vardır:
    - **Kimlik doğrulamasız (varsayılan)**: Sadece herkese açık piyasa verisi
      (semboller, OHLCV). Screener, ML ve strateji modülleri bu modu kullanır
      ve API anahtarınıza asla dokunmaz.
    - **Kimlik doğrulamalı** (`api_key`/`api_secret` verildiğinde): Bakiye
      okuma, pozisyon okuma ve gerçek emir gönderme. Bu mod yalnızca
      `app/trading/executor.py` üzerinden, açık güvenlik kapılarından
      geçirilerek kullanılmalıdır — doğrudan burada değil.
    """

    def __init__(self, api_key: str | None = None, api_secret: str | None = None, testnet: bool = True) -> None:
        auth = {"apiKey": api_key, "secret": api_secret} if api_key and api_secret else {}
        # `enableRateLimit`/`timeout` açıkça verilmezse ccxt'nin varsayılanı
        # (rate limit KAPALI, 10sn timeout) kullanılır — art arda yüzlerce
        # sembol taranan `scan_market`'te bu, borsanın kendi limitine
        # sessizce (istisna fırlatmadan sadece yavaşlayarak) çarpıp
        # taramanın kendi periyodundan çok uzun sürmesine yol açabiliyordu.
        # `enableRateLimit=True` ccxt'nin kendi trafiğini borsanın izin
        # verdiği hıza göre otomatik aralıklandırmasını sağlar.
        auth = {**auth, "enableRateLimit": True, "timeout": 15000}
        self._spot = ccxt.binance(auth)
        self._futures = ccxt.binance({**auth, "options": {"defaultType": "future"}})
        if testnet:
            self._spot.set_sandbox_mode(True)
            self._futures.set_sandbox_mode(True)
        self._authenticated = bool(api_key and api_secret)
        self.testnet = testnet

    def _client(self, market_type: str) -> ccxt.binance:
        return self._futures if market_type == "future" else self._spot

    def _require_auth(self) -> None:
        if not self._authenticated:
            raise PermissionError(
                "Bu işlem Binance API anahtarı gerektirir. BinanceExchange'i "
                "api_key/api_secret ile oluşturun (bkz. app/trading/executor.py)."
            )

    # ---- Herkese açık piyasa verisi (Exchange arayüzü) ----

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        client = self._client(market_type)
        markets = client.load_markets()
        return [
            m["symbol"]
            for m in markets.values()
            if m.get("quote") == quote_currency
            and m.get("active", True)
            and (market_type != "future" or m.get("swap"))
        ]

    def fetch_tickers(self, quote_currency: str, market_type: str) -> dict[str, dict]:
        """`client.fetch_tickers()` — TEK istekte TÜM sembollerin 24 saatlik
        ticker'ını (son fiyat + hacim) döner. `list_symbols` + sembol başına
        `fetch_ohlcv` döngüsünün AKSİNE (bkz. base sınıf docstring'i), bu
        borsanın kendi toplu uç noktasını kullandığı için yüzlerce ayrı
        istek yerine 1 istekle sonuçlanır."""
        client = self._client(market_type)
        tickers = client.fetch_tickers()
        result: dict[str, dict] = {}
        for symbol, ticker in tickers.items():
            market = client.markets.get(symbol, {})
            if market.get("quote") != quote_currency:
                continue
            if not market.get("active", True):
                continue
            if market_type == "future" and not market.get("swap"):
                continue
            last = ticker.get("last") or ticker.get("close") or 0.0
            quote_volume = ticker.get("quoteVolume")
            if quote_volume is None:
                base_volume = ticker.get("baseVolume") or 0.0
                quote_volume = base_volume * (last or 0.0)
            result[symbol] = {"last": float(last or 0.0), "quote_volume": float(quote_volume or 0.0)}
        return result

    _MAX_CANDLES_PER_CALL = 1000  # Binance futures REST API'sinin tek istekteki üst sınırı

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        """`limit` mum döner. `limit`, Binance'in tek istekteki üst sınırını
        (1000) aşarsa, geriye doğru sayfalama (pagination) yaparak birden
        fazla istekle birleştirir — böylece 1000'den çok mumluk (aylar/yıllar
        süren) geçmiş veri de ücretsiz ve güvenle çekilebilir.
        """
        client = self._client("future")
        if limit <= self._MAX_CANDLES_PER_CALL:
            raw = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
            return self._to_frame(raw)

        timeframe_ms = client.parse_timeframe(timeframe) * 1000
        end_ms = client.milliseconds()
        start_ms = since if since is not None else end_ms - limit * timeframe_ms

        all_rows: list[list] = []
        cursor = start_ms
        while len(all_rows) < limit and cursor < end_ms:
            batch = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=self._MAX_CANDLES_PER_CALL, since=cursor)
            if not batch:
                break
            all_rows.extend(batch)
            last_ts = batch[-1][0]
            if last_ts <= cursor:  # ilerleme yoksa sonsuz döngüyü önle
                break
            cursor = last_ts + timeframe_ms

        return self._to_frame(all_rows[-limit:])

    @staticmethod
    def _to_frame(raw: list[list]) -> pd.DataFrame:
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def fetch_taker_flow(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        """Binance'in HAM kline uç noktasından (ccxt'nin `fetch_ohlcv` ile
        normalize ettiği 6 sütunun ÖTESİNDE) "taker buy base asset volume"
        alanını çeker — bir mumdaki toplam hacmin ne kadarının AGRESİF
        (piyasa emriyle, en iyi teklif/talep fiyatını vuran) alıcılardan
        geldiğini gösterir.

        Bu, `fetch_order_book_metrics`teki anlık emir defteri
        dengesizliğinden (statik, yalnızca periyodik anlık görüntü olarak
        toplanabilen, geçmişi olmayan) FARKLI ve TAMAMLAYICI bir mikro
        yapı sinyalidir: mum bazlı olduğu için TAM GEÇMİŞE sahiptir ve
        `ml_train_lookback` derinliğinde geriye dönük backfill edilebilir
        (bkz. `app.ml.features.taker_flow_features`).

        Döner: `timestamp`, `volume` (toplam), `taker_buy_base_volume`
        kolonlarıyla bir DataFrame — oran (`taker_buy_ratio`) çağıran
        tarafından hesaplanır.
        """
        client = self._client("future")
        client.load_markets()
        market_id = client.market(symbol)["id"]

        def _fetch_page(page_limit: int, page_since: int | None) -> list[list]:
            params: dict = {"symbol": market_id, "interval": timeframe, "limit": page_limit}
            if page_since is not None:
                params["startTime"] = page_since
            return client.fapiPublicGetKlines(params)

        if limit <= self._MAX_CANDLES_PER_CALL:
            raw = _fetch_page(limit, since)
            return self._taker_flow_frame(raw)

        timeframe_ms = client.parse_timeframe(timeframe) * 1000
        end_ms = client.milliseconds()
        start_ms = since if since is not None else end_ms - limit * timeframe_ms

        all_rows: list[list] = []
        cursor = start_ms
        while len(all_rows) < limit and cursor < end_ms:
            batch = _fetch_page(self._MAX_CANDLES_PER_CALL, cursor)
            if not batch:
                break
            all_rows.extend(batch)
            last_ts = int(batch[-1][0])
            if last_ts <= cursor:  # ilerleme yoksa sonsuz döngüyü önle
                break
            cursor = last_ts + timeframe_ms

        return self._taker_flow_frame(all_rows[-limit:])

    @staticmethod
    def _taker_flow_frame(raw: list[list]) -> pd.DataFrame:
        columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        df = pd.DataFrame(raw, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
        df["volume"] = df["volume"].astype(float)
        df["taker_buy_base_volume"] = df["taker_buy_base_volume"].astype(float)
        return df[["timestamp", "volume", "taker_buy_base_volume"]]

    def fetch_funding_rate(self, symbol: str) -> float | None:
        """Şu anki (bir sonraki ödemede uygulanacak) funding rate'i döner
        (ör. 0.0001 = %0.01). Kimlik doğrulama gerektirmez, herkese açık
        veridir. Perpetual futures'a özgüdür — spot sembollerde None döner.
        """
        try:
            result = self._futures.fetch_funding_rate(symbol)
            rate = result.get("fundingRate")
            return float(rate) if rate is not None else None
        except Exception:  # noqa: BLE001 - makro veri opsiyoneldir, hata ana akışı bozmamalı
            return None

    def fetch_order_book_metrics(self, symbol: str, depth: int = 20) -> dict | None:
        """Emir defterinin (order book) ANLIK bir özetini döner — geçmişe
        dönük emir defteri borsalarda saklanmaz/satılmaz, bu yüzden yalnızca
        periyodik anlık görüntü (`app.orderbook.service`) olarak toplanabilir.
        Kimlik doğrulama gerektirmez, herkese açık veridir.

        Döner: {bid_volume, ask_volume, imbalance (-1..1), spread_pct} ya da
        erişilemezse None.
        """
        try:
            book = self._futures.fetch_order_book(symbol, limit=depth)
            bids = book.get("bids") or []
            asks = book.get("asks") or []
            if not bids or not asks:
                return None
            bid_volume = float(sum(qty for _price, qty in bids))
            ask_volume = float(sum(qty for _price, qty in asks))
            total = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total if total > 0 else 0.0
            best_bid, best_ask = bids[0][0], asks[0][0]
            mid = (best_bid + best_ask) / 2
            spread_pct = ((best_ask - best_bid) / mid * 100) if mid > 0 else 0.0
            return {
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "imbalance": imbalance,
                "spread_pct": spread_pct,
            }
        except Exception:  # noqa: BLE001 - order book verisi opsiyoneldir, hata ana akışı bozmamalı
            return None

    # ---- Kimlik doğrulamalı hesap/emir işlemleri ----

    def fetch_balance(self, market_type: str = "future") -> dict:
        self._require_auth()
        return self._client(market_type).fetch_balance()

    def fetch_positions(self) -> list[dict]:
        self._require_auth()
        return self._futures.fetch_positions()

    def fetch_open_orders(self, symbol: str | None = None, market_type: str = "future") -> list[dict]:
        self._require_auth()
        return self._client(market_type).fetch_open_orders(symbol)

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        market_type: str = "future",
    ) -> dict:
        self._require_auth()
        client = self._client(market_type)
        return client.create_order(symbol, order_type, side, amount, price)

    def cancel_order(self, order_id: str, symbol: str, market_type: str = "future") -> dict:
        self._require_auth()
        return self._client(market_type).cancel_order(order_id, symbol)

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        self._require_auth()
        return self._futures.set_leverage(leverage, symbol)

    def get_api_key_permissions(self) -> dict:
        """Binance'e bu API anahtarının izinlerini sorar (Güvenlik Protokolü
        Bölüm 9.1 — çekim izninin kapalı olduğunu KOD İÇİNDE doğrulamak için).

        ccxt'nin `sapiGetAccountApiRestrictions` uç noktasını sarar; bazı alt
        hesap/izin kombinasyonlarında Binance bu uç noktayı kısıtlayabilir,
        bu durumda çağıran taraf hatayı "doğrulanamadı" olarak ele almalıdır.
        """
        self._require_auth()
        return self._spot.sapiGetAccountApiRestrictions()
