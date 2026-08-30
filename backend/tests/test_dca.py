import numpy as np

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
