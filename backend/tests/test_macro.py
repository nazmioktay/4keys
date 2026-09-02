import pandas as pd
import pytest

from app.core.config import settings
from app.db import repository as db
from app.db.session import init_db, reset_for_tests
from app.macro import data
from app.macro.service import fetch_macro_snapshot, refresh_and_record_macro_snapshot


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "sqlite:///:memory:")
    reset_for_tests()
    init_db()
    yield
    reset_for_tests()


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_get_total_market_cap_and_btc_dominance_parses_coingecko(monkeypatch):
    payload = {"data": {"total_market_cap": {"usd": 2_500_000_000_000.0}, "market_cap_percentage": {"btc": 54.3}}}
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: _FakeResponse(payload))

    result = data.get_total_market_cap_and_btc_dominance()
    assert result["total_market_cap"] == pytest.approx(2_500_000_000_000.0)
    assert result["btc_dominance"] == pytest.approx(54.3)


def test_get_total_market_cap_returns_none_on_failure(monkeypatch):
    def _raise(*a, **k):
        raise ConnectionError("boom")

    monkeypatch.setattr(data.requests, "get", _raise)
    result = data.get_total_market_cap_and_btc_dominance()
    assert result == {"total_market_cap": None, "btc_dominance": None}


def test_get_fed_funds_rate_returns_none_without_api_key():
    assert data.get_fed_funds_rate(None) is None
    assert data.get_fed_funds_rate("") is None


def test_get_fed_funds_rate_parses_fred_response(monkeypatch):
    payload = {"observations": [{"value": "5.33"}]}
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: _FakeResponse(payload))
    assert data.get_fed_funds_rate("fake-key") == pytest.approx(5.33)


def test_get_ecb_deposit_rate_parses_sdw_response(monkeypatch):
    payload = {"dataSets": [{"series": {"0:0:0:0:0:0": {"observations": {"0": [4.0]}}}}]}
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: _FakeResponse(payload))
    assert data.get_ecb_deposit_rate() == pytest.approx(4.0)


def test_get_yfinance_last_close_returns_none_on_empty_history(monkeypatch):
    class _FakeTicker:
        def history(self, period):
            return pd.DataFrame()

    import sys
    import types

    fake_yf = types.SimpleNamespace(Ticker=lambda ticker: _FakeTicker())
    sys.modules["yfinance"] = fake_yf
    try:
        assert data.get_yfinance_last_close("^VIX") is None
    finally:
        del sys.modules["yfinance"]


def test_get_binance_btc_funding_rate_uses_exchange_method():
    class _FakeExchange:
        def fetch_funding_rate(self, symbol):
            assert symbol == "BTC/USDT:USDT"
            return 0.0001

    assert data.get_binance_btc_funding_rate(_FakeExchange()) == pytest.approx(0.0001)


def test_get_binance_btc_funding_rate_returns_none_without_method():
    class _FakeExchangeNoFunding:
        pass

    assert data.get_binance_btc_funding_rate(_FakeExchangeNoFunding()) is None


def test_get_binance_btc_funding_rate_returns_none_on_error():
    class _FakeExchangeErrors:
        def fetch_funding_rate(self, symbol):
            raise RuntimeError("network down")

    assert data.get_binance_btc_funding_rate(_FakeExchangeErrors()) is None


def test_fetch_macro_snapshot_combines_all_sources(monkeypatch):
    monkeypatch.setattr(data, "get_total_market_cap_and_btc_dominance", lambda: {"total_market_cap": 1.0, "btc_dominance": 50.0})
    monkeypatch.setattr(data, "get_world_indices", lambda: {"sp500": 5000.0, "nasdaq": 16000.0, "nikkei": 39000.0, "dax": 18000.0})
    monkeypatch.setattr(data, "get_vix", lambda: 15.0)
    monkeypatch.setattr(data, "get_gold_price", lambda: 2000.0)
    monkeypatch.setattr(data, "get_fed_funds_rate", lambda key: 5.33)
    monkeypatch.setattr(data, "get_ecb_deposit_rate", lambda: 4.0)
    monkeypatch.setattr(data, "get_binance_btc_funding_rate", lambda ex: 0.0001)

    class _FakeExchange:
        pass

    snapshot = fetch_macro_snapshot(_FakeExchange())
    assert snapshot["total_market_cap"] == 1.0
    assert snapshot["btc_dominance"] == 50.0
    assert snapshot["vix"] == 15.0
    assert snapshot["sp500"] == 5000.0
    assert snapshot["fed_funds_rate"] == 5.33
    assert snapshot["ecb_deposit_rate"] == 4.0
    assert snapshot["funding_rate_btc"] == 0.0001


def test_refresh_and_record_macro_snapshot_persists_to_db(monkeypatch):
    monkeypatch.setattr(data, "get_total_market_cap_and_btc_dominance", lambda: {"total_market_cap": 1.0, "btc_dominance": 50.0})
    monkeypatch.setattr(data, "get_world_indices", lambda: {"sp500": None, "nasdaq": None, "nikkei": None, "dax": None})
    monkeypatch.setattr(data, "get_vix", lambda: None)
    monkeypatch.setattr(data, "get_gold_price", lambda: None)
    monkeypatch.setattr(data, "get_fed_funds_rate", lambda key: None)
    monkeypatch.setattr(data, "get_ecb_deposit_rate", lambda: None)
    monkeypatch.setattr(data, "get_binance_btc_funding_rate", lambda ex: None)

    class _FakeExchange:
        pass

    refresh_and_record_macro_snapshot(_FakeExchange())

    latest = db.get_latest_macro_snapshot()
    assert latest is not None
    assert latest["total_market_cap"] == 1.0
    assert latest["btc_dominance"] == 50.0
    assert latest["vix"] is None  # eksik kaynak sessizce None kalmalı, hata fırlatmamalı
