import pytest

from app.core.config import settings
from app.engine.decision import DecisionEngine
from app.engine.positions import PaperPositionStore
from app.portfolio.manager import PortfolioManager
from app.portfolio.schemas import RiskRules
from app.security import kill_switch
from app.security.safety import MAX_LEVERAGE, check_withdrawals_disabled, enforce_leverage_cap
from app.trading.executor import LiveTradingDisabled, place_live_order, set_live_leverage
from app.trading.schemas import LeverageRequest, OrderRequest


@pytest.fixture(autouse=True)
def _reset_security_state():
    kill_switch.reset_for_tests()
    yield
    kill_switch.reset_for_tests()


def test_kill_switch_starts_inactive():
    assert kill_switch.is_active() is False


def test_kill_switch_activate_and_deactivate():
    kill_switch.activate("test sebebi", triggered_by="manual")
    assert kill_switch.is_active() is True
    assert kill_switch.status().reason == "test sebebi"
    assert kill_switch.status().triggered_by == "manual"

    kill_switch.deactivate()
    assert kill_switch.is_active() is False


def test_enforce_leverage_cap_allows_within_limit():
    assert enforce_leverage_cap(MAX_LEVERAGE) == MAX_LEVERAGE
    assert enforce_leverage_cap(1) == 1


def test_enforce_leverage_cap_rejects_above_limit():
    with pytest.raises(ValueError, match="güvenlik tavanını"):
        enforce_leverage_cap(MAX_LEVERAGE + 1)


class _FakeExchangePermissions:
    def __init__(self, enable_withdrawals: bool, raise_error: bool = False):
        self._enable_withdrawals = enable_withdrawals
        self._raise_error = raise_error

    def get_api_key_permissions(self) -> dict:
        if self._raise_error:
            raise RuntimeError("borsa erişilemedi")
        return {"enableWithdrawals": self._enable_withdrawals}


def test_check_withdrawals_disabled_when_actually_disabled():
    safe, message = check_withdrawals_disabled(_FakeExchangePermissions(enable_withdrawals=False))
    assert safe is True


def test_check_withdrawals_disabled_flags_when_enabled():
    safe, message = check_withdrawals_disabled(_FakeExchangePermissions(enable_withdrawals=True))
    assert safe is False
    assert "ÇEKİM" in message


def test_check_withdrawals_disabled_fails_closed_on_verification_error():
    safe, message = check_withdrawals_disabled(_FakeExchangePermissions(enable_withdrawals=False, raise_error=True))
    assert safe is False  # doğrulanamayan durum güvenli sayılmaz


def test_portfolio_manager_auto_trips_kill_switch_on_drawdown(monkeypatch):
    monkeypatch.setattr(settings, "kill_switch_daily_drawdown_pct", 10.0)
    portfolio = PortfolioManager(starting_equity=1000, rules=RiskRules(max_symbol_exposure_pct=100, max_total_exposure_pct=100))

    portfolio.open("BTC/USDT", "long", entry_price=100, size_quote=1000)
    assert kill_switch.is_active() is False

    portfolio.close("BTC/USDT", exit_price=88)  # -%12 pnl -> equity 1000 -> 880, drawdown %12 > %10

    assert kill_switch.is_active() is True
    assert kill_switch.status().triggered_by == "auto_drawdown"


def test_portfolio_manager_does_not_trip_kill_switch_below_threshold(monkeypatch):
    monkeypatch.setattr(settings, "kill_switch_daily_drawdown_pct", 50.0)
    portfolio = PortfolioManager(starting_equity=1000, rules=RiskRules(max_symbol_exposure_pct=100, max_total_exposure_pct=100))

    portfolio.open("BTC/USDT", "long", entry_price=100, size_quote=1000)
    portfolio.close("BTC/USDT", exit_price=95)  # -%5 pnl, limit %50

    assert kill_switch.is_active() is False


class _TrendExchangeForDecision:
    def list_symbols(self, quote_currency, market_type):
        return ["BTCUSDT"]

    def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
        import numpy as np
        import pandas as pd

        n = max(limit, 220)
        close = np.linspace(100, 200, n)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
                "open": close, "high": close + 1, "low": close - 1, "close": close,
                "volume": np.full(n, 1000.0),
            }
        )


def test_decision_engine_blocks_open_when_kill_switch_active():
    from app.ml.dataset import build_training_dataset
    from app.ml.model import SignalModel

    exchange = _TrendExchangeForDecision()
    X, y = build_training_dataset(exchange, ["BTCUSDT"], "4h", 220, horizon=5, threshold_pct=1.0)
    model = SignalModel()
    model.fit(X, y)

    kill_switch.activate("test", triggered_by="manual")

    engine = DecisionEngine(
        exchange=exchange, model=model, positions=PaperPositionStore(),
        timeframe="4h", lookback=220, open_confidence=0.0, close_confidence=0.0,
    )
    actions = engine.run_cycle(["BTCUSDT"])
    assert len(actions) == 1
    assert actions[0].type == "blocked"
    assert "kill switch" in actions[0].reason


def test_place_live_order_blocked_by_kill_switch(monkeypatch):
    monkeypatch.setattr(settings, "enable_live_trading", True)
    monkeypatch.setattr(settings, "binance_api_key", type(settings.binance_api_key)("x"))
    monkeypatch.setattr(settings, "binance_api_secret", type(settings.binance_api_secret)("y"))
    kill_switch.activate("acil durum", triggered_by="manual")

    request = OrderRequest(symbol="BTC/USDT:USDT", side="buy", amount=0.001, confirm=True)
    with pytest.raises(LiveTradingDisabled, match="Kill switch aktif"):
        place_live_order(request)


def test_place_live_order_blocked_when_withdrawals_enabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_live_trading", True)
    monkeypatch.setattr(settings, "binance_api_key", type(settings.binance_api_key)("x"))
    monkeypatch.setattr(settings, "binance_api_secret", type(settings.binance_api_secret)("y"))

    import app.trading.executor as executor_module

    monkeypatch.setattr(executor_module, "get_trading_exchange", lambda: _FakeExchangePermissions(enable_withdrawals=True))

    request = OrderRequest(symbol="BTC/USDT:USDT", side="buy", amount=0.001, confirm=True)
    with pytest.raises(LiveTradingDisabled, match="ÇEKİM"):
        place_live_order(request)


def test_set_live_leverage_rejects_above_cap(monkeypatch):
    monkeypatch.setattr(settings, "enable_live_trading", True)
    request = LeverageRequest(symbol="BTC/USDT:USDT", leverage=MAX_LEVERAGE + 5, confirm=True)
    with pytest.raises(ValueError, match="güvenlik tavanını"):
        set_live_leverage(request)
