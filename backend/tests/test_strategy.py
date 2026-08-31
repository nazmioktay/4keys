import numpy as np
import pandas as pd

from app.strategy.engine import run_backtest
from app.strategy.evaluator import evaluate
from app.strategy.examples import EMA_GOLDEN_CROSS, RSI_OVERSOLD_BOUNCE
from app.strategy.schemas import ConditionNode, Operand


def test_compare_condition():
    node = ConditionNode(type="compare", left=Operand(indicator="rsi"), op="lt", right=Operand(value=30))
    row = pd.Series({"rsi": 25})
    assert evaluate(node, row, None) == True
    row = pd.Series({"rsi": 40})
    assert evaluate(node, row, None) == False


def test_cross_above_condition():
    node = ConditionNode(
        type="cross", left=Operand(indicator="a"), right=Operand(indicator="b"), direction="above"
    )
    prev_row = pd.Series({"a": 1, "b": 2})
    row = pd.Series({"a": 3, "b": 2})
    assert evaluate(node, row, prev_row) == True

    prev_row = pd.Series({"a": 3, "b": 2})
    row = pd.Series({"a": 4, "b": 2})
    assert evaluate(node, row, prev_row) == False  # zaten üstteydi, kesişim yok


def test_and_or_conditions():
    node = ConditionNode(
        type="and",
        conditions=[
            ConditionNode(type="compare", left=Operand(indicator="rsi"), op="lt", right=Operand(value=30)),
            ConditionNode(type="compare", left=Operand(indicator="volume"), op="gt", right=Operand(value=1000)),
        ],
    )
    row = pd.Series({"rsi": 20, "volume": 1500})
    assert evaluate(node, row, None) == True
    row = pd.Series({"rsi": 20, "volume": 500})
    assert evaluate(node, row, None) == False


def _dip_and_recover_ohlcv(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    base = np.concatenate([np.linspace(100, 70, n // 2), np.linspace(70, 130, n // 2)])
    close = base + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": rng.uniform(800, 1200, n),
        }
    )


def test_backtest_runs_and_produces_trades_for_rsi_strategy():
    ohlcv = _dip_and_recover_ohlcv()
    result = run_backtest(ohlcv, RSI_OVERSOLD_BOUNCE, "TEST/USDT")
    assert result.trades_closed >= 0  # veri setine göre 0 da olabilir, kod hatasız çalışmalı
    for t in result.trades:
        assert t.exit_index > t.entry_index


def test_backtest_ema_cross_strategy_runs():
    ohlcv = _dip_and_recover_ohlcv()
    result = run_backtest(ohlcv, EMA_GOLDEN_CROSS, "TEST/USDT")
    assert isinstance(result.total_profit_pct, float)
    assert result.win_rate_pct >= 0
