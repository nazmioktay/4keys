import pandas as pd
import pytest

from app.core.config import settings
from app.db import repository as db
from app.db.session import check_connection, init_db, is_enabled, reset_for_tests
from app.portfolio.manager import PortfolioManager
from app.portfolio.schemas import RiskRules


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch):
    """Her testi paylaşılan bellek içi SQLite veritabanıyla izole çalıştırır."""
    monkeypatch.setattr(settings, "database_url", "sqlite:///:memory:")
    reset_for_tests()
    init_db()
    yield
    reset_for_tests()


@pytest.fixture
def _db_disabled(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "")
    reset_for_tests()
    yield
    reset_for_tests()


def test_is_enabled_reflects_database_url():
    assert is_enabled() is True


def test_check_connection_succeeds_on_valid_db():
    assert check_connection() is True


def test_init_db_is_idempotent():
    assert init_db() is True
    assert init_db() is True  # ikinci çağrı da hata vermemeli


def test_record_and_read_trade_roundtrip():
    record = {
        "symbol": "BTC/USDT",
        "direction": "long",
        "entry_price": 100.0,
        "exit_price": 106.0,
        "size_quote": 200.0,
        "pnl_pct": 6.0,
        "pnl_quote": 12.0,
        "opened_at": "2024-01-01T00:00:00+00:00",
        "closed_at": "2024-01-01T04:00:00+00:00",
        "source": "test",
    }
    db.record_trade(record)

    trades = db.get_recent_trades(limit=10)
    assert len(trades) == 1
    assert trades[0]["symbol"] == "BTC/USDT"
    assert trades[0]["pnl_pct"] == pytest.approx(6.0)


def test_record_and_read_signal_roundtrip():
    db.record_signal("BTC/USDT", source="ml", direction="long", confidence=0.72, price=100.0)
    db.record_signal("ETH/USDT", source="screener", direction="short", confidence=0.55, price=50.0)

    all_signals = db.get_recent_signals(limit=10)
    assert len(all_signals) == 2

    ml_only = db.get_recent_signals(limit=10, source="ml")
    assert len(ml_only) == 1
    assert ml_only[0]["symbol"] == "BTC/USDT"

    btc_only = db.get_recent_signals(limit=10, symbol="BTC/USDT")
    assert len(btc_only) == 1


def test_record_latest_candle_deduplicates_on_conflict():
    row = pd.Series(
        {"timestamp": pd.Timestamp("2024-01-01T00:00:00Z"), "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 100}
    )
    db.record_latest_candle("BTC/USDT", "4h", row)
    db.record_latest_candle("BTC/USDT", "4h", row)  # aynı time/symbol/timeframe -> IntegrityError yutulmalı

    from app.db.models import OHLCVRaw
    from app.db.session import session_scope

    with session_scope() as s:
        count = s.query(OHLCVRaw).count()
    assert count == 1


def test_portfolio_manager_close_persists_trade_to_db():
    portfolio = PortfolioManager(starting_equity=1000, rules=RiskRules())
    decision = portfolio.propose_open("BTC/USDT", "long", entry_price=100, stop_loss_price=95)
    portfolio.open("BTC/USDT", "long", 100, decision.size_quote)
    portfolio.close("BTC/USDT", exit_price=106)

    trades = db.get_recent_trades(limit=10)
    assert len(trades) == 1
    assert trades[0]["symbol"] == "BTC/USDT"


def test_repository_functions_are_noop_when_disabled(_db_disabled):
    db.record_signal("BTC/USDT", source="ml", direction="long", confidence=0.5, price=100.0)
    db.record_trade(
        {
            "symbol": "BTC/USDT", "direction": "long", "entry_price": 100, "exit_price": 105,
            "size_quote": 100, "pnl_pct": 5, "pnl_quote": 5,
            "opened_at": "2024-01-01T00:00:00+00:00", "closed_at": "2024-01-01T01:00:00+00:00",
        }
    )
    assert db.get_recent_trades() == []
    assert db.get_recent_signals() == []


def test_repository_never_raises_when_db_url_invalid(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "postgresql+psycopg2://invalid:invalid@127.0.0.1:1/doesnotexist")
    reset_for_tests()
    try:
        # Bağlanılamasa bile bu çağrılar exception fırlatmamalı (loglayıp yutmalı).
        db.record_signal("BTC/USDT", source="ml", direction="long", confidence=0.5, price=100.0)
        db.record_trade(
            {
                "symbol": "BTC/USDT", "direction": "long", "entry_price": 100, "exit_price": 105,
                "size_quote": 100, "pnl_pct": 5, "pnl_quote": 5,
                "opened_at": "2024-01-01T00:00:00+00:00", "closed_at": "2024-01-01T01:00:00+00:00",
            }
        )
        assert db.get_recent_trades() == []
    finally:
        reset_for_tests()
