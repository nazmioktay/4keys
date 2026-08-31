import math

from .schemas import PerformanceMetrics


def compute_metrics(trade_pnls_pct: list[float], num_candles: int, timeframe_minutes: int) -> PerformanceMetrics:
    """Bir işlem listesinin yüzde getirilerinden kapsamlı performans metrikleri üretir.

    Getiri serisi olarak mum bazlı değil **işlem bazlı** pnl% kullanılır
    (bu, düzensiz aralıklarla açılan pozisyonlar için standart ve pratik bir
    yaklaşımdır). Yıllıklaştırma, verinin kapsadığı toplam süreye göre işlem
    başına düşen frekans tahmin edilerek yapılır.
    """
    n = len(trade_pnls_pct)
    total_days = (num_candles * timeframe_minutes) / (60 * 24)
    years = total_days / 365 if total_days > 0 else 0.0

    if n == 0:
        return PerformanceMetrics(
            num_trades=0,
            win_rate_pct=0.0,
            total_return_pct=0.0,
            cagr_pct=None,
            sharpe_ratio=None,
            sortino_ratio=None,
            max_drawdown_pct=0.0,
            calmar_ratio=None,
            profit_factor=None,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            expectancy_pct=0.0,
        )

    wins = [p for p in trade_pnls_pct if p > 0]
    losses = [p for p in trade_pnls_pct if p < 0]

    win_rate_pct = len(wins) / n * 100
    total_return_pct = sum(trade_pnls_pct)
    avg_win_pct = sum(wins) / len(wins) if wins else 0.0
    avg_loss_pct = sum(losses) / len(losses) if losses else 0.0
    expectancy_pct = total_return_pct / n

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    # gross_loss == 0 durumunda oran matematiksel olarak tanımsız/sonsuz olur;
    # JSON'da güvenle taşınamayan Infinity yerine None (tanımsız) döndürülür.
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    equity_curve = [0.0]
    for p in trade_pnls_pct:
        equity_curve.append(equity_curve[-1] + p)
    peak = 0.0
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, peak - value)

    mean_pnl = total_return_pct / n
    variance = sum((p - mean_pnl) ** 2 for p in trade_pnls_pct) / n
    std_pnl = math.sqrt(variance)

    downside = [min(p, 0.0) for p in trade_pnls_pct]
    downside_variance = sum(d**2 for d in downside) / n
    downside_std = math.sqrt(downside_variance)

    trades_per_year = (n / years) if years > 0 else 0.0

    sharpe_ratio = (mean_pnl / std_pnl) * math.sqrt(trades_per_year) if std_pnl > 0 and trades_per_year > 0 else None
    sortino_ratio = (
        (mean_pnl / downside_std) * math.sqrt(trades_per_year) if downside_std > 0 and trades_per_year > 0 else None
    )

    growth_factor = 1.0
    for p in trade_pnls_pct:
        growth_factor *= 1 + p / 100
    cagr_pct = None
    if years > 0 and growth_factor > 0:
        cagr_pct = (growth_factor ** (1 / years) - 1) * 100

    calmar_ratio = (cagr_pct / max_drawdown) if (cagr_pct is not None and max_drawdown > 0) else None

    return PerformanceMetrics(
        num_trades=n,
        win_rate_pct=round(win_rate_pct, 2),
        total_return_pct=round(total_return_pct, 3),
        cagr_pct=round(cagr_pct, 3) if cagr_pct is not None else None,
        sharpe_ratio=round(sharpe_ratio, 3) if sharpe_ratio is not None else None,
        sortino_ratio=round(sortino_ratio, 3) if sortino_ratio is not None else None,
        max_drawdown_pct=round(max_drawdown, 3),
        calmar_ratio=round(calmar_ratio, 3) if calmar_ratio is not None else None,
        profit_factor=round(profit_factor, 3) if profit_factor is not None else None,
        avg_win_pct=round(avg_win_pct, 3),
        avg_loss_pct=round(avg_loss_pct, 3),
        expectancy_pct=round(expectancy_pct, 3),
    )
