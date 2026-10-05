from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import settings
from app.db.session import init_db, reset_for_tests
from app.portfolio import shared
from app.portfolio.manager import PortfolioManager
from app.portfolio.schemas import RiskRules


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "sqlite:///:memory:")
    reset_for_tests()
    init_db()
    monkeypatch.setattr(shared, "_portfolio", None)
    yield
    reset_for_tests()


def _rules() -> RiskRules:
    return RiskRules(position_sizing_method="fixed_risk", entry_tranche_weights=[1.0], exit_tranche_weights=[1.0])


def test_state_roundtrip_preserves_positions_history_and_rules():
    pm = PortfolioManager(starting_equity=1000, rules=_rules())
    pm.open("BTC/USDT", "long", 100.0, 200.0, stop_loss_price=95.0)
    pm.open("ETH/USDT", "short", 50.0, 100.0, stop_loss_price=55.0)
    pm.close("ETH/USDT", 45.0, reason="test")

    restored = PortfolioManager.from_state(pm.to_state())

    assert restored.equity == pytest.approx(pm.equity)
    assert set(restored.positions) == {"BTC/USDT"}
    position = restored.positions["BTC/USDT"]
    assert position.stop_loss_price == 95.0
    assert position.opened_at == pm.positions["BTC/USDT"].opened_at
    assert restored.closed_history == pm.closed_history
    assert restored.rules == pm.rules
    assert restored.trade_stats() == pm.trade_stats()


def test_shared_portfolio_survives_restart_via_db():
    portfolio = shared.reset_portfolio(starting_equity=1000, rules=_rules())
    portfolio.open("BTC/USDT", "long", 100.0, 200.0, stop_loss_price=95.0)

    shared._portfolio = None  # süreç yeniden başladı
    restored = shared.get_portfolio()

    assert "BTC/USDT" in restored.positions
    assert restored.positions["BTC/USDT"].entry_price == 100.0
    assert restored.persist_enabled is True


def test_rules_update_is_persisted():
    shared.reset_portfolio(starting_equity=1000, rules=_rules())
    shared.get_portfolio().rules = RiskRules(leverage=2, position_sizing_method="fixed_risk")

    shared._portfolio = None
    assert shared.get_portfolio().rules.leverage == 2


def test_daily_loss_counter_resets_at_utc_day_boundary():
    pm = PortfolioManager(starting_equity=1000, rules=RiskRules(daily_loss_limit_pct=5.0, position_sizing_method="fixed_risk"))
    pm.open("BTC/USDT", "long", 100.0, 1000.0, stop_loss_price=90.0)
    pm.close("BTC/USDT", 93.0)  # ~ -%7 x 1000 = -70 USDT, limit 50
    assert pm.realized_pnl_today < -50

    blocked = pm.propose_open("ETH/USDT", "long", 100.0, 95.0)
    assert blocked.allowed is False

    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    pm._roll_day(tomorrow)
    assert pm.realized_pnl_today == 0.0
    # oturum toplamı korunur, yalnızca günlük sayaç sıfırlanır
    assert pm.realized_pnl_session < -50


def test_disabled_new_entries_block_opens_but_stops_still_work():
    from tests.test_portfolio import _engine_with, _FixedModel, _TickerExchange

    rules = RiskRules(entry_tranche_weights=[1.0], max_symbol_exposure_pct=100, max_total_exposure_pct=100, allow_new_entries=False)
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)
    engine = _engine_with(_TickerExchange(live_price=None), _FixedModel("long", 0.9), portfolio, {})

    action = engine.run_cycle(["BTC/USDT"])[0]
    assert action.type == "blocked"
    assert "yeni pozisyon açma kapalı" in action.reason
    assert portfolio.get("BTC/USDT") is None

    # önceden açılmış bir pozisyonun stop'u hâlâ çalışır
    portfolio.open("ETH/USDT", "long", entry_price=110, size_quote=100, stop_loss_price=105)
    engine = _engine_with(_TickerExchange(live_price=104.0), _FixedModel("long", 0.9), portfolio)
    assert engine.evaluate("ETH/USDT").type == "close"


def test_allow_new_entries_survives_restart():
    shared.reset_portfolio(starting_equity=1000, rules=RiskRules(allow_new_entries=False))
    shared._portfolio = None
    assert shared.get_portfolio().rules.allow_new_entries is False
