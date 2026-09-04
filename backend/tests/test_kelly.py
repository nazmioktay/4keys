import numpy as np
import pandas as pd
import pytest

from app.engine.decision import DecisionEngine
from app.engine.positions import PaperPositionStore
from app.exchanges.base import Exchange
from app.portfolio.manager import PortfolioManager
from app.portfolio.risk_manager import calculate_kelly_position_size, kelly_fraction, resolve_kelly_multiplier
from app.portfolio.schemas import RiskRules, TradeStats


def test_kelly_fraction_known_example():
    # %60 kazanma oranı, ortalama kazanç %4, ortalama kayıp -%2 -> b=2
    # f* = 0.6 - 0.4/2 = 0.4
    f = kelly_fraction(win_rate_pct=60, avg_win_pct=4, avg_loss_pct=-2)
    assert f == pytest.approx(0.4)


def test_kelly_fraction_negative_edge_clamped_to_zero():
    # Düşük kazanma oranı + düşük odds -> negatif Kelly -> 0 dönmeli (işlem açma)
    f = kelly_fraction(win_rate_pct=30, avg_win_pct=1, avg_loss_pct=-3)
    assert f == 0.0


def test_kelly_fraction_invalid_inputs_return_zero():
    assert kelly_fraction(60, avg_win_pct=0, avg_loss_pct=-2) == 0.0
    assert kelly_fraction(60, avg_win_pct=4, avg_loss_pct=0) == 0.0


def test_resolve_kelly_multiplier_variants():
    assert resolve_kelly_multiplier("quarter", None) == 0.25
    assert resolve_kelly_multiplier("half", None) == 0.5
    assert resolve_kelly_multiplier("full", None) == 1.0
    assert resolve_kelly_multiplier("custom", 0.33) == 0.33
    with pytest.raises(ValueError):
        resolve_kelly_multiplier("custom", None)


def test_calculate_kelly_position_size_applies_multiplier_and_cap():
    # full Kelly = %40, yarım Kelly uygulanırsa %20 olmalı
    size_quote, applied_pct, full_kelly_pct = calculate_kelly_position_size(
        equity=1000, win_rate_pct=60, avg_win_pct=4, avg_loss_pct=-2,
        kelly_multiplier=0.5, max_kelly_fraction_pct=25.0,
    )
    assert full_kelly_pct == pytest.approx(40.0)
    assert applied_pct == pytest.approx(20.0)
    assert size_quote == pytest.approx(200.0)


def test_calculate_kelly_position_size_respects_safety_cap():
    # full Kelly = %40, tam Kelly uygulanırsa %40 olur ama cap %25'te kesilmeli
    size_quote, applied_pct, full_kelly_pct = calculate_kelly_position_size(
        equity=1000, win_rate_pct=60, avg_win_pct=4, avg_loss_pct=-2,
        kelly_multiplier=1.0, max_kelly_fraction_pct=25.0,
    )
    assert full_kelly_pct == pytest.approx(40.0)
    assert applied_pct == pytest.approx(25.0)  # cap devrede
    assert size_quote == pytest.approx(250.0)


def test_portfolio_manager_falls_back_to_fixed_risk_without_enough_history():
    rules = RiskRules(position_sizing_method="kelly", kelly_min_trades=20, max_risk_per_trade_pct=1.0, max_symbol_exposure_pct=100.0, max_total_exposure_pct=100.0)
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)

    # Hiç kapanmış işlem yok -> Kelly istatistiği güvenilmez, fixed_risk'e düşmeli
    decision = portfolio.propose_open("BTC/USDT", "long", entry_price=100, stop_loss_price=95)
    # fixed_risk: risk_amount=%1*1000=10, stop_distance=%5 -> size=200
    assert decision.size_quote == pytest.approx(200.0)


def test_portfolio_manager_uses_kelly_once_enough_history():
    rules = RiskRules(position_sizing_method="kelly", kelly_min_trades=5, kelly_multiplier=0.5, max_kelly_fraction_pct=50.0, max_symbol_exposure_pct=100.0, max_total_exposure_pct=100.0)
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)

    # Karışık (kazanan+kaybeden) bir geçmişi doğrudan closed_history'ye yazarak simüle et.
    portfolio.closed_history = [
        {"pnl_pct": 4.0}, {"pnl_pct": -2.0}, {"pnl_pct": 4.0}, {"pnl_pct": -2.0}, {"pnl_pct": 4.0},
    ]
    stats = portfolio.trade_stats()
    assert stats.num_trades == 5
    assert stats.win_rate_pct == pytest.approx(60.0)
    assert stats.avg_win_pct == pytest.approx(4.0)
    assert stats.avg_loss_pct == pytest.approx(-2.0)

    decision = portfolio.propose_open("ETH/USDT", "long", entry_price=100, stop_loss_price=95)
    # full Kelly=%40, yarım Kelly=%20, equity=1000 -> 200
    assert decision.size_quote == pytest.approx(200.0)


def test_portfolio_manager_kelly_stats_override():
    rules = RiskRules(position_sizing_method="kelly", kelly_min_trades=10, kelly_multiplier=1.0, max_kelly_fraction_pct=100.0, max_symbol_exposure_pct=100.0, max_total_exposure_pct=100.0)
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)

    override = TradeStats(num_trades=50, win_rate_pct=60, avg_win_pct=4, avg_loss_pct=-2)
    decision = portfolio.propose_open(
        "BTC/USDT", "long", entry_price=100, stop_loss_price=95, kelly_stats_override=override
    )
    assert decision.size_quote == pytest.approx(400.0)  # full kelly %40, equity 1000


class _TrendExchange(Exchange):
    """Karar motoru testleri için basit, sürekli yükselen trendli sahte borsa."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng(11)

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        return ["UPUSDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
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


def test_run_cycle_sizes_new_position_with_kelly():
    """DecisionEngine.run_cycle, portföy Kelly moduna alındığında açılış
    sinyalini Kelly kriterine göre boyutlandırmalı (sabit-risk değil)."""
    from app.ml.dataset import build_training_dataset
    from app.ml.model import SignalModel

    exchange = _TrendExchange()
    X, y = build_training_dataset(exchange, ["UPUSDT"], "4h", 220, horizon=5, threshold_pct=1.0)
    model = SignalModel()
    model.fit(X, y)

    rules = RiskRules(
        position_sizing_method="kelly",
        kelly_multiplier=0.5,
        kelly_min_trades=5,
        max_kelly_fraction_pct=50.0,
        max_symbol_exposure_pct=100.0,
        max_total_exposure_pct=100.0,
        entry_tranche_weights=[1.0],  # bu test saf Kelly boyutlandırmasını ölçüyor, kademeli alımı değil
    )
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)
    # Kelly'nin devreye girmesi için sisteme yeterli (ve karışık) bir canlı
    # işlem geçmişi "yaşatıyoruz" — gerçek hayatta bu geçmiş zamanla,
    # run_cycle çağrıları sonucu kapanan işlemlerden birikir.
    portfolio.closed_history = [
        {"pnl_pct": 4.0}, {"pnl_pct": -2.0}, {"pnl_pct": 4.0}, {"pnl_pct": -2.0}, {"pnl_pct": 4.0},
    ]

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

    # Yükselen trend + %0 güven eşiği -> model bir yön üretiyorsa open_long beklenir.
    assert action.type in ("open_long", "hold")
    if action.type == "open_long":
        position = portfolio.get("UPUSDT")
        assert position is not None
        # Stop-loss mesafesinden bağımsız: Kelly boyutlandırması sadece
        # kazanma oranı + ort. kazanç/kayıptan hesaplanır.
        # full Kelly = 0.6 - 0.4/2 = 0.4 -> %40; yarım Kelly -> %20; equity=1000 -> 200
        assert position.size_quote == pytest.approx(200.0)
