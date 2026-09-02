"""Ücretsiz makro/piyasa bağlamı veri kaynakları.

Her fonksiyon izole çalışır ve ASLA exception fırlatmaz — bir kaynak
(ör. Yahoo Finance geçici olarak erişilemez) başarısız olursa yalnızca
`None` döner, diğer kaynakları veya ana işlem akışını etkilemez. Bu,
projenin geri kalanındaki "kalıcılık/ek veri asla ana akışı bozmaz"
prensibiyle tutarlıdır (bkz. `app.db.repository`).
"""

import logging

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 10


def get_total_market_cap_and_btc_dominance() -> dict:
    """CoinGecko `/global` — toplam kripto piyasa değeri (TOTAL) ve BTC
    dominansı. Ücretsiz, API anahtarı gerekmez."""
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()["data"]
        return {
            "total_market_cap": float(data["total_market_cap"]["usd"]),
            "btc_dominance": float(data["market_cap_percentage"]["btc"]),
        }
    except Exception:  # noqa: BLE001
        logger.warning("coingecko global verisi alınamadı", exc_info=True)
        return {"total_market_cap": None, "btc_dominance": None}


def get_yfinance_last_close(ticker: str) -> float | None:
    """Yahoo Finance'ten bir ticker'ın en son kapanış değerini döner."""
    try:
        import yfinance as yf

        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:  # noqa: BLE001
        logger.warning("yfinance %s alınamadı", ticker, exc_info=True)
        return None


def get_vix() -> float | None:
    return get_yfinance_last_close("^VIX")


def get_gold_price() -> float | None:
    return get_yfinance_last_close("GC=F")


def get_world_indices() -> dict:
    return {
        "sp500": get_yfinance_last_close("^GSPC"),
        "nasdaq": get_yfinance_last_close("^IXIC"),
        "nikkei": get_yfinance_last_close("^N225"),
        "dax": get_yfinance_last_close("^GDAXI"),
    }


def get_fed_funds_rate(api_key: str | None) -> float | None:
    """FRED (ABD Merkez Bankası) API'sinden efektif federal fon oranı
    (DFF serisi). Ücretsiz ama bir API key gerektirir
    (https://fred.stlouisfed.org/docs/api/api_key.html — anında, ücretsiz).
    Key ayarlanmamışsa (FOURKEYS_FRED_API_KEY boş) None döner."""
    if not api_key:
        return None
    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "DFF", "api_key": api_key, "file_type": "json", "sort_order": "desc", "limit": 1},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        obs = resp.json()["observations"][0]
        return float(obs["value"])
    except Exception:  # noqa: BLE001
        logger.warning("FRED fed funds rate alınamadı", exc_info=True)
        return None


def get_ecb_deposit_rate() -> float | None:
    """ECB'nin kamuya açık İstatistik Veri Ambarı (SDW) REST API'si —
    mevduat faizi (DFR). API anahtarı gerekmez."""
    try:
        resp = requests.get(
            "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.DFR.LEV",
            params={"format": "jsondata", "lastNObservations": 1},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        series = next(iter(payload["dataSets"][0]["series"].values()))
        obs = next(iter(series["observations"].values()))
        return float(obs[0])
    except Exception:  # noqa: BLE001
        logger.warning("ECB deposit rate alınamadı", exc_info=True)
        return None


def get_binance_btc_funding_rate(exchange) -> float | None:
    """Verilen borsa adaptöründen (bkz. `app.exchanges.binance.BinanceExchange`)
    BTC perpetual futures funding rate'ini döner. Adaptör bu metodu
    desteklemiyorsa (ör. demo/sentetik borsa) None döner."""
    fetch = getattr(exchange, "fetch_funding_rate", None)
    if fetch is None:
        return None
    try:
        return fetch("BTC/USDT:USDT")
    except Exception:  # noqa: BLE001
        logger.warning("funding rate alınamadı", exc_info=True)
        return None
