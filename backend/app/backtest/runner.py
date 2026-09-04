import pandas as pd

from app.dca.simulator import simulate_dca
from app.exchanges.base import Exchange
from app.strategy.engine import run_backtest as run_strategy_backtest

from .data import fetch_full_history, timeframe_to_minutes
from .metrics import compute_metrics, monte_carlo_bootstrap
from .schemas import BacktestReport, BacktestRequest, DataSufficiency, PerformanceMetrics


def _simulate(ohlcv: pd.DataFrame, request: BacktestRequest) -> tuple[int, list[float]]:
    """Verilen istek tipine göre simülasyonu çalıştırıp (kapanan işlem sayısı,
    işlem başına yüzde getiri listesi) döner. DCA ve JSON-strateji
    motorlarını tek bir arayüz altında birleştirir."""
    if request.kind == "dca":
        result = simulate_dca(ohlcv["close"].to_numpy(), request.dca_params)
        return result.trades_closed, result.trade_pnls_pct

    result = run_strategy_backtest(ohlcv, request.strategy, request.symbol)
    return result.trades_closed, [t.pnl_pct for t in result.trades]


def _discover_sufficient_history(
    exchange: Exchange, request: BacktestRequest
) -> tuple[pd.DataFrame, DataSufficiency]:
    """Az veriyle başlayıp, verilen strateji/DCA parametreleriyle
    istatistiksel olarak yeterli sayıda işlem üretilene kadar (veya
    borsanın/limitin izin verdiği maksimuma ulaşana kadar) geçmiş veri
    miktarını kademeli olarak artırır.

    Bu, kullanıcının "geçmiş veri miktarını öğrenerek oluştur" isteğinin
    karşılığıdır: sabit bir mum sayısı varsaymak yerine, gerçekte kaç mumun
    gerektiğini bu strateji için deneyerek keşfeder.
    """
    timeframe = request.timeframe
    candles_needed = request.initial_candles
    ohlcv = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    trades_closed = 0

    while True:
        ohlcv = fetch_full_history(exchange, request.symbol, timeframe, max_candles=candles_needed)

        if len(ohlcv) < 30:
            return ohlcv, DataSufficiency(
                candles_used=len(ohlcv),
                candles_requested_final=candles_needed,
                trades_found=0,
                min_trades_target=request.min_trades,
                sufficient=False,
                reason=(
                    f"Borsada bu sembol/timeframe için yalnızca {len(ohlcv)} mumluk veri var; "
                    "anlamlı bir backtest için yetersiz."
                ),
            )

        trades_closed, _ = _simulate(ohlcv, request)

        exchange_exhausted = len(ohlcv) < candles_needed  # borsa istenenden azını verdi -> daha fazla geçmiş yok
        reached_cap = candles_needed >= request.max_candles

        if trades_closed >= request.min_trades or exchange_exhausted or reached_cap:
            sufficient = trades_closed >= request.min_trades
            if sufficient:
                reason = (
                    f"{len(ohlcv)} mum tarandı, {trades_closed} kapanan işlemle "
                    f"istatistiksel hedefe (min {request.min_trades}) ulaşıldı."
                )
            elif exchange_exhausted:
                reason = (
                    f"Borsada bu sembol için toplam {len(ohlcv)} mum mevcut ve bu da yalnızca "
                    f"{trades_closed} işlem üretti (hedef: {request.min_trades}); daha fazla geçmiş veri yok."
                )
            else:
                reason = (
                    f"Maksimum mum sınırına ({request.max_candles}) ulaşıldı, {trades_closed} işlem üretildi "
                    f"(hedef: {request.min_trades}); sonuçlar düşük istatistiksel güvenilirlikte olabilir."
                )
            return ohlcv, DataSufficiency(
                candles_used=len(ohlcv),
                candles_requested_final=candles_needed,
                trades_found=trades_closed,
                min_trades_target=request.min_trades,
                sufficient=sufficient,
                reason=reason,
            )

        candles_needed = min(candles_needed * 2, request.max_candles)


def _metrics_for(ohlcv: pd.DataFrame, request: BacktestRequest, timeframe_minutes: int) -> tuple[PerformanceMetrics, list[float]]:
    _, pnls = _simulate(ohlcv, request)
    return compute_metrics(pnls, len(ohlcv), timeframe_minutes), pnls


def run_backtest_report(exchange: Exchange, request: BacktestRequest) -> BacktestReport:
    """Verilen DCA parametreleri veya JSON stratejisi için uçtan uca güçlü
    bir backtest raporu üretir:

    1. Bu parametreler için ne kadar geçmiş veri gerektiğini keşfeder
       (bkz. `_discover_sufficient_history`).
    2. Veriyi kronolojik olarak eğitim (in-sample) / test (out-of-sample)
       olarak ikiye böler.
    3. Her iki dönem ve tüm veri için ayrı ayrı zengin performans metrikleri
       (Sharpe, Sortino, Calmar, profit factor, CAGR, max drawdown vb.) hesaplar.
    4. Eğitim ve test dönemleri arasında büyük bir performans uçurumu varsa
       (aşırı uyum / overfitting belirtisi) bunu uyarı olarak işaretler.
    """
    timeframe = request.timeframe
    timeframe_minutes = timeframe_to_minutes(timeframe)

    ohlcv, sufficiency = _discover_sufficient_history(exchange, request)

    if len(ohlcv) < 30:
        return BacktestReport(
            symbol=request.symbol,
            timeframe=timeframe,
            data_sufficiency=sufficiency,
            train_metrics=None,
            test_metrics=None,
            full_period_metrics=None,
            warnings=["Yeterli geçmiş veri olmadığı için metrikler hesaplanamadı."],
        )

    split_idx = int(len(ohlcv) * request.train_ratio)
    train_df = ohlcv.iloc[:split_idx].reset_index(drop=True)
    test_df = ohlcv.iloc[split_idx:].reset_index(drop=True)

    train_metrics, test_pnls = None, []
    if len(train_df) >= 30:
        train_metrics, _ = _metrics_for(train_df, request, timeframe_minutes)

    test_metrics = None
    if len(test_df) >= 30:
        test_metrics, test_pnls = _metrics_for(test_df, request, timeframe_minutes)

    full_metrics, _ = _metrics_for(ohlcv, request, timeframe_minutes)

    monte_carlo = (
        monte_carlo_bootstrap(test_pnls, num_simulations=request.monte_carlo_simulations) if test_pnls else None
    )

    warnings: list[str] = []
    if not sufficiency.sufficient:
        warnings.append(sufficiency.reason)

    if train_metrics is not None and test_metrics is not None:
        if test_metrics.num_trades < 5:
            warnings.append("Test (out-of-sample) döneminde çok az işlem var; doğrulama güvenilirliği düşük.")
        if train_metrics.total_return_pct > 0 and test_metrics.total_return_pct < 0:
            warnings.append(
                "Eğitim döneminde kârlı ama test (out-of-sample) döneminde zararlı — "
                "aşırı uyum (overfitting) riski yüksek, parametreleri temkinli kullanın."
            )
        elif (
            train_metrics.total_return_pct > 0
            and test_metrics.total_return_pct < train_metrics.total_return_pct * 0.3
        ):
            warnings.append(
                "Test dönemi getirisi eğitim döneminin çok altında kaldı — sonuçlar döneme özgü "
                "olabilir, canlıya almadan önce farklı sembol/dönemlerle de doğrulayın."
            )
    elif test_metrics is None:
        warnings.append("Test bölümünde göstergelerin ısınması için yeterli mum yok; out-of-sample doğrulama atlandı.")

    return BacktestReport(
        symbol=request.symbol,
        timeframe=timeframe,
        data_sufficiency=sufficiency,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        full_period_metrics=full_metrics,
        monte_carlo=monte_carlo,
        warnings=warnings,
    )
