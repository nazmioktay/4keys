import numpy as np

from .schemas import DCABacktestResult, DCAParams
from .simulator import simulate_dca

DEVIATION_GRID = [1.0, 1.5, 2.0, 3.0, 5.0]
DEVIATION_MULTIPLIER_GRID = [1.0, 1.2, 1.5, 2.0]
ORDER_SIZE_MULTIPLIER_GRID = [1.0, 1.3, 1.5, 2.0]
MAX_SAFETY_ORDERS_GRID = [3, 5, 7, 10]
TAKE_PROFIT_GRID = [1.0, 1.5, 2.0, 3.0, 4.0]
STOP_LOSS_GRID = [None, 8.0, 15.0]


def _max_theoretical_capital(base_order_size: float, order_size_multiplier: float, max_safety_orders: int) -> float:
    multipliers = order_size_multiplier ** np.arange(max_safety_orders + 1)
    return base_order_size * float(multipliers.sum())


def _score(result: DCABacktestResult, objective: str) -> float:
    if objective == "profit":
        return result.total_profit_pct
    if objective == "win_rate":
        return result.win_rate_pct + result.total_profit_pct / 1000  # win_rate öncelikli, profit tiebreak
    # profit_over_drawdown (varsayılan)
    drawdown_floor = max(result.max_drawdown_pct, 0.1)
    return result.total_profit_pct / drawdown_floor


def optimize_dca(
    prices: np.ndarray,
    balance: float,
    direction: str,
    objective: str = "profit_over_drawdown",
    allow_stop_loss: bool = False,
    top_n: int = 5,
) -> list[DCABacktestResult]:
    """Verilen fiyat serisi ve sermaye üzerinde DCA parametre uzayını tarayıp
    en iyi `top_n` kombinasyonu döner.

    Her kombinasyon için base_order_size, tüm averaging order'ların teorik
    olarak dolması durumunda kullanılacak maksimum sermaye `balance`'ı tam
    kullanacak şekilde otomatik hesaplanır — yani optimizasyon hem
    parametreleri hem de bunlara bağlı pozisyon boyutunu birlikte arar.
    """
    stop_loss_options = STOP_LOSS_GRID if allow_stop_loss else [None]
    results: list[DCABacktestResult] = []

    for deviation_pct in DEVIATION_GRID:
        for deviation_multiplier in DEVIATION_MULTIPLIER_GRID:
            for order_size_multiplier in ORDER_SIZE_MULTIPLIER_GRID:
                for max_safety_orders in MAX_SAFETY_ORDERS_GRID:
                    theoretical_max = _max_theoretical_capital(1.0, order_size_multiplier, max_safety_orders)
                    base_order_size = balance / theoretical_max
                    if base_order_size <= 0:
                        continue

                    for take_profit_pct in TAKE_PROFIT_GRID:
                        for stop_loss_pct in stop_loss_options:
                            params = DCAParams(
                                base_order_size=base_order_size,
                                deviation_pct=deviation_pct,
                                deviation_multiplier=deviation_multiplier,
                                order_size_multiplier=order_size_multiplier,
                                max_safety_orders=max_safety_orders,
                                take_profit_pct=take_profit_pct,
                                stop_loss_pct=stop_loss_pct,
                                direction=direction,
                            )
                            result = simulate_dca(prices, params)
                            if result.trades_closed > 0:
                                results.append(result)

    if not results:
        return []

    results.sort(key=lambda r: _score(r, objective), reverse=True)
    return results[:top_n]
