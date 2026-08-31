from unittest.mock import MagicMock, patch

from app.bank.denizbank import DenizbankOpenBankingClient


def _client() -> DenizbankOpenBankingClient:
    return DenizbankOpenBankingClient(
        base_url="https://api.denizbank.example",
        client_id="test-client",
        client_secret="test-secret",
        redirect_uri="http://localhost:8000/bank/denizbank/callback",
    )


def test_authorization_url_contains_required_params():
    client = _client()
    url = client.get_authorization_url(scope="accounts", state="xyz")
    assert url.startswith("https://api.denizbank.example/oauth2/authorize?")
    assert "client_id=test-client" in url
    assert "state=xyz" in url
    assert "scope=accounts" in url


@patch("app.bank.denizbank.requests.post")
def test_exchange_code_for_token(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "at-123",
        "refresh_token": "rt-456",
        "expires_in": 3600,
        "token_type": "Bearer",
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    client = _client()
    token = client.exchange_code_for_token("auth-code-abc")

    assert token.access_token == "at-123"
    assert token.expires_in == 3600
    mock_post.assert_called_once()
    called_url = mock_post.call_args.args[0]
    assert called_url == "https://api.denizbank.example/oauth2/token"


@patch("app.bank.denizbank.requests.get")
def test_get_accounts_parses_response(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "accounts": [
            {"account_id": "1", "iban": "TR000000000000000000000001", "account_name": "Vadesiz", "currency": "TRY"}
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    client = _client()
    accounts = client.get_accounts("at-123")

    assert len(accounts) == 1
    assert accounts[0].iban == "TR000000000000000000000001"
    headers = mock_get.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer at-123"
