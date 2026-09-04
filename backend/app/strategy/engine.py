import pandas as pd

from app.screener.indicators import compute_indicators

from .evaluator import evaluate
from .schemas import StrategyBacktestResult, StrategyDefinition, TradeRecord


def _pnl_pct(direction: str, entry_price: float, current_price: float) -> float:
    if direction == "long":
        return (current_price / entry_price - 1) * 100
    return (entry_price / current_price - 1) * 100


def run_backtest(ohlcv: pd.DataFrame, strategy: StrategyDefinition, symbol: str) -> StrategyBacktestResult:
    """JSON kural ağacıyla tanımlanmış bir stratejiyi geçmiş veri üzerinde çalıştırır.

    TradingView Pine Script'e ihtiyaç duymadan; `entry`/`exit` kural ağaçları
    ve opsiyonel take-profit/stop-loss yüzdeleriyle strateji anında test edilir.
    """
    indicators = compute_indicators(ohlcv)

    trades: list[TradeRecord] = []
    equity_curve = [0.0]
    in_position = False
    entry_index = 0
    entry_price = 0.0

    for i in range(1, len(indicators)):
        row = indicators.iloc[i]
        prev_row = indicators.iloc[i - 1]

        if not in_position:
            if evaluate(strategy.entry, row, prev_row):
                in_position = True
                entry_index = i
                entry_price = float(row["close"])
            continue

        price = float(row["close"])
        # Bariyer kontrolleri (take-profit/stop-loss/kural) BRÜT pnl_pct
        # üzerinden yapılır — gerçek bir emir de bu seviyelere brüt fiyat
        # hareketiyle ulaşır; işlem maliyeti SONRADAN, kaydedilen PnL'e
        # yansıtılır (aşağıda).
        pnl_pct = _pnl_pct(strategy.direction, entry_price, price)
        exit_reason: str | None = None

        if strategy.exit is not None and evaluate(strategy.exit, row, prev_row):
            exit_reason = "rule"
        elif strategy.take_profit_pct is not None and pnl_pct >= strategy.take_profit_pct:
            exit_reason = "take_profit"
        elif strategy.stop_loss_pct is not None and pnl_pct <= -strategy.stop_loss_pct:
            exit_reason = "stop_loss"

        if exit_reason:
            # Round-trip maliyet (giriş BACAĞI + çıkış BACAĞI) — önceden
            # PnL yalnızca brüt fiyat farkından hesaplanıyordu.
            cost_pct = (strategy.commission_pct + strategy.slippage_pct) * 2
            net_pnl_pct = pnl_pct - cost_pct
            trades.append(
                TradeRecord(
                    entry_index=entry_index,
                    exit_index=i,
                    entry_price=entry_price,
                    exit_price=price,
                    pnl_pct=round(net_pnl_pct, 3),
                    exit_reason=exit_reason,
                )
            )
            equity_curve.append(equity_curve[-1] + net_pnl_pct)
            in_position = False

    trades_closed = len(trades)
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    win_rate_pct = (wins / trades_closed * 100) if trades_closed else 0.0
    total_profit_pct = round(sum(t.pnl_pct for t in trades), 3)

    peak = 0.0
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, peak - value)

    return StrategyBacktestResult(
        symbol=symbol,
        strategy_name=strategy.name,
        trades=trades,
        trades_closed=trades_closed,
        win_rate_pct=round(win_rate_pct, 2),
        total_profit_pct=total_profit_pct,
        max_drawdown_pct=round(max_drawdown, 3),
    )
