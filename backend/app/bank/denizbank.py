from urllib.parse import urlencode

import requests

from .schemas import BankAccountSummary, BankBalance, TokenResponse


class DenizbankOpenBankingClient:
    """Denizbank Açık Bankacılık (TPP) istemcisi — bakiye/hesap görüntüleme.

    ÖNEMLİ: Denizbank'ın tek, halka açık ve belgelenmiş bir "trading API"si
    yoktur; banka hesabı bilgisine erişim Türkiye'de BDDK'nın düzenlediği
    Açık Bankacılık çerçevesi üzerinden, bir TPP (Third Party Provider /
    fintech) olarak kayıt olup OAuth2 tabanlı bir onay (consent) akışıyla
    yapılır. Bu sınıf, o standart akışın (yetkilendirme kodu grant'ı ->
    access token -> hesap/bakiye sorgusu) GENEL kalıbını uygular.

    Aşağıdaki uç nokta yolları (`/oauth2/authorize`, `/oauth2/token`,
    `/accounts`, `/accounts/{id}/balances`) Türkiye Açık Bankacılık
    ekosisteminde yaygın kalıptır ancak **Denizbank'a özgü kesin yollar
    değildir** — TPP başvurunuz onaylandığında Denizbank'ın geliştirici
    portalından aldığınız gerçek `base_url` ve uç nokta yollarıyla bu
    metodları güncelleyin (veya `endpoint_overrides` ile geçersiz kılın).
    """

    def __init__(
        self,
        base_url: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        endpoint_overrides: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self._endpoints = {
            "authorize": "/oauth2/authorize",
            "token": "/oauth2/token",
            "accounts": "/accounts",
            "balances": "/accounts/{account_id}/balances",
            **(endpoint_overrides or {}),
        }

    def get_authorization_url(self, scope: str = "accounts", state: str | None = None) -> str:
        """Kullanıcının bankada onay (consent) vermesi için ziyaret etmesi gereken URL."""
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": scope,
        }
        if state:
            params["state"] = state
        return f"{self.base_url}{self._endpoints['authorize']}?{urlencode(params)}"

    def exchange_code_for_token(self, code: str) -> TokenResponse:
        response = requests.post(
            f"{self.base_url}{self._endpoints['token']}",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=15,
        )
        response.raise_for_status()
        return TokenResponse(**response.json())

    def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        response = requests.post(
            f"{self.base_url}{self._endpoints['token']}",
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=15,
        )
        response.raise_for_status()
        return TokenResponse(**response.json())

    def get_accounts(self, access_token: str) -> list[BankAccountSummary]:
        response = requests.get(
            f"{self.base_url}{self._endpoints['accounts']}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15,
        )
        response.raise_for_status()
        return [BankAccountSummary(**item) for item in response.json().get("accounts", [])]

    def get_balance(self, access_token: str, account_id: str) -> BankBalance:
        path = self._endpoints["balances"].format(account_id=account_id)
        response = requests.get(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15,
        )
        response.raise_for_status()
        return BankBalance(**response.json())
