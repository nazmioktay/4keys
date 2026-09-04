import numpy as np
import pytest

from app.dca.optimizer import optimize_dca
from app.dca.schemas import DCAParams
from app.dca.simulator import simulate_dca


def test_dip_and_recovery_hits_take_profit():
    # 100 -> düşer, bir averaging order tetiklenir -> toparlanıp TP'ye ulaşır
    prices = np.array([100, 98, 96, 94, 97, 100, 103, 106])
    params = DCAParams(
        base_order_size=100,
        deviation_pct=3.0,
        deviation_multiplier=1.0,
        order_size_multiplier=1.0,
        max_safety_orders=2,
        take_profit_pct=2.0,
        direction="long",
    )
    result = simulate_dca(prices, params)
    assert result.trades_closed >= 1
    assert result.win_rate_pct == 100.0
    assert result.total_profit_pct > 0


def test_continuous_downtrend_without_stop_loss_stays_open():
    prices = np.array([100, 95, 90, 85, 80, 75, 70])
    params = DCAParams(
        base_order_size=100,
        deviation_pct=3.0,
        deviation_multiplier=1.2,
        order_size_multiplier=1.5,
        max_safety_orders=3,
        take_profit_pct=2.0,
        direction="long",
    )
    result = simulate_dca(prices, params)
    assert result.trades_closed == 0
    assert result.trades_open_at_end == 1


def test_stop_loss_closes_losing_trade():
    prices = np.array([100, 95, 90, 85, 80, 75, 70, 65])
    params = DCAParams(
        base_order_size=100,
        deviation_pct=50.0,  # averaging order tetiklenmesin, saf SL testi
        deviation_multiplier=1.0,
        order_size_multiplier=1.0,
        max_safety_orders=0,
        take_profit_pct=2.0,
        stop_loss_pct=10.0,
        direction="long",
    )
    result = simulate_dca(prices, params)
    # Sürekli düşüşte SL tekrar tekrar tetiklenebilir (her defasında yeni işlem açılır)
    assert result.trades_closed >= 1
    assert result.win_rate_pct == 0.0
    assert result.total_profit_pct < 0


def test_zero_cost_take_profit_matches_gross_pnl():
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    params = DCAParams(
        base_order_size=100,
        deviation_pct=2.0,
        deviation_multiplier=1.5,
        order_size_multiplier=1.5,
        max_safety_orders=2,
        take_profit_pct=5.0,
        direction="long",
        commission_pct=0.0,
        slippage_pct=0.0,
    )
    result = simulate_dca(prices, params)
    assert result.trades_closed == 1
    assert result.trade_pnls_pct[0] == pytest.approx(5.0)


def test_cost_reduces_realized_pnl():
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    base_kwargs = dict(
        base_order_size=100,
        deviation_pct=2.0,
        deviation_multiplier=1.5,
        order_size_multiplier=1.5,
        max_safety_orders=2,
        take_profit_pct=5.0,
        direction="long",
    )
    zero_cost = simulate_dca(prices, DCAParams(**base_kwargs, commission_pct=0.0, slippage_pct=0.0))
    with_cost = simulate_dca(prices, DCAParams(**base_kwargs, commission_pct=0.05, slippage_pct=0.03))

    # Base order + exit = 2 dolu bacak -> cost_pct = (0.05+0.03)*2 = 0.16
    expected_cost_pct = (0.05 + 0.03) * 2
    assert with_cost.trade_pnls_pct[0] == pytest.approx(zero_cost.trade_pnls_pct[0] - expected_cost_pct)
    assert with_cost.total_profit_pct < zero_cost.total_profit_pct


def test_cost_scales_with_filled_safety_orders():
    # Fiyat önce düşüp bir safety order tetikler, sonra toparlanıp take-profit'e ulaşır.
    prices = np.array([100.0, 97.0, 95.0, 100.0, 103.0, 106.0])
    params = DCAParams(
        base_order_size=100,
        deviation_pct=3.0,
        deviation_multiplier=1.0,
        order_size_multiplier=1.0,
        max_safety_orders=1,
        take_profit_pct=5.0,
        direction="long",
        commission_pct=0.05,
        slippage_pct=0.03,
    )
    zero_cost_params = params.model_copy(update={"commission_pct": 0.0, "slippage_pct": 0.0})

    result = simulate_dca(prices, params)
    gross_result = simulate_dca(prices, zero_cost_params)

    assert result.trades_closed == 1
    assert gross_result.trades_closed == 1
    # base order + 1 safety order + exit = 3 dolu bacak
    expected_cost_pct = (0.05 + 0.03) * 3
    assert result.trade_pnls_pct[0] == pytest.approx(gross_result.trade_pnls_pct[0] - expected_cost_pct)


def test_optimizer_returns_ranked_candidates_within_balance():
    rng = np.random.default_rng(7)
    # Hafif yukarı trendli, dalgalı bir seri -> averaging + TP fırsatları olsun
    base = np.linspace(100, 130, 150)
    noise = rng.normal(0, 3.0, 150)
    prices = base + noise

    candidates = optimize_dca(prices, balance=1000.0, direction="long", top_n=5)
    assert len(candidates) > 0
    assert len(candidates) <= 5
    for c in candidates:
        assert c.max_capital_used <= 1000.0 + 1e-6
    # sıralama azalan olmalı (profit_over_drawdown varsayılan skor -> total_profit_pct kabaca artmalı)
    assert candidates[0].total_profit_pct >= candidates[-1].total_profit_pct - 50
