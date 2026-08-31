import pytest

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
