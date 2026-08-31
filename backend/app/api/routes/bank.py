from fastapi import APIRouter, HTTPException

from app.bank import token_store
from app.bank.denizbank import DenizbankOpenBankingClient
from app.bank.schemas import BankAccountSummary, BankBalance
from app.core.config import settings

router = APIRouter(prefix="/bank/denizbank", tags=["bank"])


def _get_client() -> DenizbankOpenBankingClient:
    if not (settings.denizbank_base_url and settings.denizbank_client_id):
        raise HTTPException(
            status_code=409,
            detail=(
                "Denizbank Açık Bankacılık ayarları eksik. FOURKEYS_DENIZBANK_BASE_URL, "
                "FOURKEYS_DENIZBANK_CLIENT_ID ve FOURKEYS_DENIZBANK_CLIENT_SECRET'i "
                "TPP başvurunuzdan aldığınız değerlerle .env dosyasına girin."
            ),
        )
    return DenizbankOpenBankingClient(
        base_url=settings.denizbank_base_url,
        client_id=settings.denizbank_client_id,
        client_secret=settings.denizbank_client_secret.get_secret_value(),
        redirect_uri=settings.denizbank_redirect_uri,
    )


@router.get("/authorize")
def authorize(state: str | None = None) -> dict:
    """Kullanıcının bankada onay vermesi için ziyaret etmesi gereken URL'i döner.

    Bu bir tarayıcı yönlendirmesi DEĞİL, URL'in kendisidir — kullanıcı bu
    linki tarayıcıda açıp bankada onay verdikten sonra Denizbank onu
    `redirect_uri`'ye (varsayılan: /bank/denizbank/callback) bir `code`
    parametresiyle geri yönlendirir.
    """
    client = _get_client()
    return {"authorization_url": client.get_authorization_url(state=state)}


@router.get("/callback")
def callback(code: str, state: str | None = None) -> dict:
    client = _get_client()
    token = client.exchange_code_for_token(code)
    token_store.save_token(token)
    return {"status": "ok", "expires_in": token.expires_in}


@router.get("/accounts", response_model=list[BankAccountSummary])
def accounts() -> list[BankAccountSummary]:
    token = token_store.get_token()
    if token is None:
        raise HTTPException(status_code=401, detail="Önce /bank/denizbank/authorize akışıyla onay verin.")
    client = _get_client()
    return client.get_accounts(token.access_token)


@router.get("/balances/{account_id}", response_model=BankBalance)
def balance(account_id: str) -> BankBalance:
    token = token_store.get_token()
    if token is None:
        raise HTTPException(status_code=401, detail="Önce /bank/denizbank/authorize akışıyla onay verin.")
    client = _get_client()
    return client.get_balance(token.access_token, account_id)
