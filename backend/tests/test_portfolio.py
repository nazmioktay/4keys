import numpy as np
import pandas as pd
import pytest

from app.engine.decision import DecisionEngine
from app.engine.positions import PaperPositionStore
from app.exchanges.base import Exchange
from app.portfolio.manager import PortfolioManager
from app.portfolio.risk_manager import calculate_position_size, evaluate_risk
from app.portfolio.schemas import PositionExposure, RiskRules


def test_calculate_position_size_matches_risk_amount():
    size_quote, risk_amount, stop_distance_pct = calculate_position_size(
        equity=1000, entry_price=100, stop_loss_price=95, risk_per_trade_pct=1.0, direction="long"
    )
    assert risk_amount == pytest.approx(10.0)
    assert stop_distance_pct == pytest.approx(5.0)
    # SL'e çarpılırsa kaybedilen tutar tam olarak risk_amount olmalı
    loss_if_sl_hit = size_quote * (stop_distance_pct / 100)
    assert loss_if_sl_hit == pytest.approx(risk_amount)


def test_calculate_position_size_rejects_invalid_stop_loss():
    with pytest.raises(ValueError):
        calculate_position_size(1000, 100, 105, 1.0, direction="long")


def test_evaluate_risk_blocks_after_daily_loss_limit():
    rules = RiskRules(daily_loss_limit_pct=5.0)
    decision = evaluate_risk(
        equity=1000,
        open_positions=[],
        realized_pnl_session=-60.0,  # -%6, limit %5
        proposed_symbol="BTC/USDT",
        proposed_size_quote=100,
        rules=rules,
    )
    assert decision.allowed is False


def test_evaluate_risk_caps_symbol_exposure():
    rules = RiskRules(max_symbol_exposure_pct=10.0, max_total_exposure_pct=100.0)
    decision = evaluate_risk(
        equity=1000,
        open_positions=[PositionExposure(symbol="BTC/USDT", size_quote=80)],
        realized_pnl_session=0.0,
        proposed_symbol="BTC/USDT",
        proposed_size_quote=100,
        rules=rules,
    )
    assert decision.allowed is True
    assert decision.size_quote == pytest.approx(20.0)  # 1000*%10 - 80 = 20


def test_evaluate_risk_blocks_new_symbol_over_concurrent_limit():
    rules = RiskRules(max_concurrent_positions=1)
    decision = evaluate_risk(
        equity=1000,
        open_positions=[PositionExposure(symbol="BTC/USDT", size_quote=50)],
        realized_pnl_session=0.0,
        proposed_symbol="ETH/USDT",
        proposed_size_quote=50,
        rules=rules,
    )
    assert decision.allowed is False


def test_portfolio_manager_open_close_updates_equity():
    portfolio = PortfolioManager(starting_equity=1000, rules=RiskRules())
    decision = portfolio.propose_open("BTC/USDT", "long", entry_price=100, stop_loss_price=97)
    assert decision.allowed is True
    portfolio.open("BTC/USDT", "long", 100, decision.size_quote)
    assert portfolio.get("BTC/USDT") is not None

    record = portfolio.close("BTC/USDT", exit_price=106)  # +%6 kâr, avg_price bazlı değil doğrudan pnl
    assert record is not None
    assert record["pnl_pct"] == pytest.approx(6.0)
    assert portfolio.equity > portfolio.starting_equity


class _TrendExchange(Exchange):
    def __init__(self) -> None:
        self._rng = np.random.default_rng(11)

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        return ["UPUSDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        n = max(limit, 220)
        base = np.linspace(100, 260, n)
        close = base + self._rng.normal(0, 1.0, n)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": self._rng.uniform(800, 1200, n),
            }
        )


def test_decision_engine_uses_portfolio_manager_for_sizing():
    from app.ml.dataset import build_training_dataset
    from app.ml.model import SignalModel

    exchange = _TrendExchange()
    X, y = build_training_dataset(exchange, ["UPUSDT"], "4h", 220, horizon=5, threshold_pct=1.0)
    model = SignalModel()
    model.fit(X, y)

    portfolio = PortfolioManager(starting_equity=1000, rules=RiskRules(max_risk_per_trade_pct=1.0))
    engine = DecisionEngine(
        exchange=exchange,
        model=model,
        positions=PaperPositionStore(),
        timeframe="4h",
        lookback=220,
        open_confidence=0.0,
        close_confidence=0.0,
        portfolio=portfolio,
    )

    actions = engine.run_cycle(["UPUSDT"])
    assert len(actions) == 1
    action = actions[0]
    if action.type == "open_long":
        position = portfolio.get("UPUSDT")
        assert position is not None
        assert position.size_quote > 0
        # %1 risk, %3 varsayılan SL mesafesi -> boyut equity'nin küçük bir kısmı olmalı
        assert position.size_quote < portfolio.equity
