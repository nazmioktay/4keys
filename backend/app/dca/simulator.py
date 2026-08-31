import numpy as np

from .schemas import DCABacktestResult, DCAParams


def _pnl_pct(direction: str, avg_price: float, current_price: float) -> float:
    if direction == "long":
        return (current_price / avg_price - 1) * 100
    return (avg_price / current_price - 1) * 100


def _trigger_price(direction: str, reference_price: float, deviation_pct: float) -> float:
    if direction == "long":
        return reference_price * (1 - deviation_pct / 100)
    return reference_price * (1 + deviation_pct / 100)


def simulate_dca(prices: np.ndarray, params: DCAParams) -> DCABacktestResult:
    """Bir DCA botunun `prices` (kapanış fiyatları) üzerindeki davranışını simüle eder.

    Basitleştirmeler: mum içi fitil hareketleri değil kapanış fiyatları
    kullanılır; komisyon ve slipaj hesaba katılmaz; bir seferde tek pozisyon
    açıktır — bir işlem kapanınca hemen bir sonraki mumda yeni işlem açılır
    (ASAP start condition davranışı).
    """
    direction = params.direction
    n = len(prices)

    trades_closed = 0
    wins = 0
    realized_profit_quote = 0.0
    max_capital_used = 0.0
    durations: list[int] = []
    equity_curve = [0.0]
    trade_pnls_pct: list[float] = []

    i = 0
    while i < n:
        entry_price = prices[i]
        avg_price = entry_price
        position_qty = params.base_order_size / entry_price
        total_invested = params.base_order_size
        last_order_price = entry_price
        safety_orders_filled = 0
        entry_index = i

        closed = False
        j = i + 1
        while j < n:
            price_j = prices[j]

            if safety_orders_filled < params.max_safety_orders:
                step_deviation = params.deviation_pct * (params.deviation_multiplier**safety_orders_filled)
                trigger = _trigger_price(direction, last_order_price, step_deviation)
                triggered = price_j <= trigger if direction == "long" else price_j >= trigger
                if triggered:
                    order_size = params.base_order_size * (params.order_size_multiplier**safety_orders_filled)
                    qty = order_size / price_j
                    total_invested += order_size
                    position_qty += qty
                    avg_price = total_invested / position_qty
                    last_order_price = price_j
                    safety_orders_filled += 1
                    max_capital_used = max(max_capital_used, total_invested)

            pnl_pct = _pnl_pct(direction, avg_price, price_j)

            if pnl_pct >= params.take_profit_pct:
                profit_quote = total_invested * (pnl_pct / 100)
                realized_profit_quote += profit_quote
                trades_closed += 1
                wins += 1
                durations.append(j - entry_index)
                equity_curve.append(equity_curve[-1] + profit_quote)
                trade_pnls_pct.append(round(pnl_pct, 4))
                i = j + 1
                closed = True
                break

            if params.stop_loss_pct is not None and pnl_pct <= -params.stop_loss_pct:
                loss_quote = total_invested * (pnl_pct / 100)
                realized_profit_quote += loss_quote
                trades_closed += 1
                durations.append(j - entry_index)
                equity_curve.append(equity_curve[-1] + loss_quote)
                trade_pnls_pct.append(round(pnl_pct, 4))
                i = j + 1
                closed = True
                break

            j += 1

        max_capital_used = max(max_capital_used, total_invested)

        if not closed:
            break  # veri sonunda açık pozisyon kaldı

    trades_open_at_end = 1 if (trades_closed == 0 and n > 1) or (i < n) else 0

    peak = 0.0
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = peak - value
        max_drawdown = max(max_drawdown, drawdown)

    capital_base = max_capital_used if max_capital_used > 0 else params.base_order_size
    total_profit_pct = (realized_profit_quote / capital_base) * 100 if capital_base else 0.0
    max_drawdown_pct = (max_drawdown / capital_base) * 100 if capital_base else 0.0
    win_rate_pct = (wins / trades_closed * 100) if trades_closed else 0.0
    avg_duration = float(np.mean(durations)) if durations else 0.0

    return DCABacktestResult(
        params=params,
        trades_closed=trades_closed,
        trades_open_at_end=trades_open_at_end,
        win_rate_pct=round(win_rate_pct, 2),
        total_profit_pct=round(total_profit_pct, 2),
        max_drawdown_pct=round(max_drawdown_pct, 2),
        max_capital_used=round(max_capital_used, 4),
        avg_trade_duration_candles=round(avg_duration, 2),
        trade_pnls_pct=trade_pnls_pct,
    )
