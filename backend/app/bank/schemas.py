from pydantic import BaseModel


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str | None = None
    expires_in: int
    token_type: str = "Bearer"


class BankAccountSummary(BaseModel):
    account_id: str
    iban: str
    account_name: str
    currency: str


class BankBalance(BaseModel):
    account_id: str
    available_balance: float
    currency: str
    as_of: str
