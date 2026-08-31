import pandas as pd
import requests

from .base import Exchange


class AlgoLabExchange(Exchange):
    """Denizbank AlgoLab BIST/VIOP adapter'ı.

    ÖNEMLİ — dürüstlük notu: AlgoLab'ın (Denizbank'ın retail algoritmik
    trading API'si) tek, sabit ve halka açık bir OpenAPI şeması yoktur;
    erişim bir API key başvurusu sonrası verilen dokümantasyona dayanır. Bu
    sınıf, AlgoLab'ın YAYGIN BİLİNEN genel kimlik doğrulama akışını
    (API key + kullanıcı adı/şifre -> SMS/e-posta doğrulama kodu -> oturum
    hash'i) ve tipik uç nokta kalıbını uygular. Gerçek kullanım öncesi API
    key başvurunuzdan erişeceğiniz güncel dokümantasyonla `_endpoints`
    sözlüğündeki yolları ve yanıt alan adlarını (`_parse_*` metodları)
    teyit edip gerekirse güncelleyin.

    Kimlik doğrulama akışı (2 adım, Binance'ten farklı olarak burada piyasa
    verisi bile oturum gerektirir):
    1. `login(username, password)` -> SMS/e-posta doğrulama kodu tetiklenir,
       geçici bir `token` döner.
    2. `login_control(token, sms_code)` -> oturum `hash`'i döner ve bu
       nesneye kaydedilir; sonraki tüm istekler bu hash'i kullanır.

    Oturum süresi sınırlıdır; uzun süre işlem yapılmazsa `session_refresh()`
    ile tazelenmelidir (AlgoLab dokümantasyonundaki süreye göre periyodik
    çağırın — bkz. `app/bist/session_store.py`).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://www.algolab.com.tr/api",
        session_hash: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._session_hash = session_hash
        self._endpoints = {
            "login": "/LoginUser",
            "login_control": "/LoginUserControl",
            "session_refresh": "/SessionRefresh",
            "symbol_info": "/GetEquityInfo",
            "candle_data": "/GetCandleData",
            "instant_position": "/InstantPosition",
            "send_order": "/SendOrder",
        }

    # ---- Oturum yönetimi ----

    def _api_key_headers(self) -> dict:
        return {"APIKEY": self.api_key}

    def _auth_headers(self) -> dict:
        return {"APIKEY": self.api_key, "Authorization": self._session_hash or ""}

    def is_authenticated(self) -> bool:
        return bool(self._session_hash)

    def _require_auth(self) -> None:
        if not self.is_authenticated():
            raise PermissionError(
                "AlgoLab oturumu açılmamış. Önce login() ve login_control() ile "
                "SMS/e-posta doğrulamasını tamamlayın (bkz. /bist/login, /bist/login/verify)."
            )

    def login(self, username: str, password: str) -> str:
        """1. adım: kullanıcı adı/şifre ile giriş, doğrulama kodu gönderilmesini
        tetikler. Döner: bir sonraki adımda kullanılacak geçici `token`."""
        response = requests.post(
            f"{self.base_url}{self._endpoints['login']}",
            json={"username": username, "password": password},
            headers=self._api_key_headers(),
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        token = data.get("content", {}).get("token") or data.get("token")
        if not token:
            raise RuntimeError(f"AlgoLab login yanıtında token bulunamadı: {data}")
        return token

    def login_control(self, token: str, sms_code: str) -> str:
        """2. adım: SMS/e-posta doğrulama kodunu doğrulayıp oturum hash'i alır."""
        response = requests.post(
            f"{self.base_url}{self._endpoints['login_control']}",
            json={"token": token, "code": sms_code},
            headers=self._api_key_headers(),
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        session_hash = data.get("content", {}).get("hash") or data.get("hash")
        if not session_hash:
            raise RuntimeError(f"AlgoLab login_control yanıtında oturum hash'i bulunamadı: {data}")
        self._session_hash = session_hash
        return session_hash

    def session_refresh(self) -> None:
        self._require_auth()
        response = requests.post(
            f"{self.base_url}{self._endpoints['session_refresh']}", headers=self._auth_headers(), timeout=15
        )
        response.raise_for_status()

    # ---- Herkese açık(-a yakın) piyasa verisi (Exchange arayüzü) ----
    # NOT: AlgoLab'da piyasa verisi de oturum gerektirir; bu yüzden bu
    # metodlar da _require_auth() çağırır — Binance adapter'ından farkı budur.

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        """`market_type`: "equity" (BIST hisse) veya "viop" (vadeli işlem)."""
        self._require_auth()
        response = requests.get(
            f"{self.base_url}{self._endpoints['symbol_info']}",
            params={"market": market_type},
            headers=self._auth_headers(),
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        items = data.get("content", data if isinstance(data, list) else [])
        return [item["symbol"] for item in items if "symbol" in item]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        self._require_auth()
        params: dict = {"symbol": symbol, "period": timeframe, "limit": limit}
        if since is not None:
            params["since"] = since
        response = requests.get(
            f"{self.base_url}{self._endpoints['candle_data']}", params=params, headers=self._auth_headers(), timeout=15
        )
        response.raise_for_status()
        data = response.json()
        bars = data.get("content", data if isinstance(data, list) else [])

        df = pd.DataFrame(bars)
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        rename_map = {"date": "timestamp", "time": "timestamp"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    # ---- Kimlik doğrulamalı hesap/emir işlemleri ----

    def fetch_positions(self) -> list[dict]:
        self._require_auth()
        response = requests.get(
            f"{self.base_url}{self._endpoints['instant_position']}", headers=self._auth_headers(), timeout=15
        )
        response.raise_for_status()
        data = response.json()
        return data.get("content", data if isinstance(data, list) else [])

    def send_order(
        self,
        symbol: str,
        direction: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        market_type: str = "equity",
    ) -> dict:
        """`direction`: "buy" | "sell". `order_type`: "market" | "limit"."""
        self._require_auth()
        payload = {
            "symbol": symbol,
            "direction": direction,
            "orderType": order_type,
            "quantity": quantity,
            "price": price,
            "market": market_type,
        }
        response = requests.post(
            f"{self.base_url}{self._endpoints['send_order']}",
            json=payload,
            headers=self._auth_headers(),
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
