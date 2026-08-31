import numpy as np
import pandas as pd
import pytest

from app.backtest.data import fetch_full_history, timeframe_to_minutes
from app.backtest.metrics import compute_metrics
from app.backtest.runner import _discover_sufficient_history, run_backtest_report
from app.backtest.schemas import BacktestRequest
from app.dca.schemas import DCAParams
from app.exchanges.base import Exchange
from app.strategy.examples import RSI_OVERSOLD_BOUNCE


class FakeHistoryExchange(Exchange):
    """Sabit uzunlukta, önceden üretilmiş bir geçmişi olan test borsası.

    `fetch_ohlcv`, gerçek borsaların `since` davranışını taklit eder: since
    verilirse o zamandan itibaren ileriye doğru en fazla `limit` mum döner;
    istenenden az veri varsa (geçmişin sonuna gelindiyse) daha az döner.
    """

    def __init__(self, total_candles: int, timeframe: str = "4h", seed: int = 1, oscillation: bool = True) -> None:
        freq_minutes = timeframe_to_minutes(timeframe)
        idx = pd.date_range("2020-01-01", periods=total_candles, freq=f"{freq_minutes}min", tz="UTC").tz_convert(None)
        rng = np.random.default_rng(seed)
        if oscillation:
            t = np.linspace(0, 40 * np.pi, total_candles)
            base = 100 + 20 * np.sin(t)
        else:
            base = np.linspace(100, 200, total_candles)
        close = base + rng.normal(0, 1.0, total_candles)
        self.full_df = pd.DataFrame(
            {
                "timestamp": idx,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": rng.uniform(800, 1200, total_candles),
            }
        )

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        return ["OSCUSDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        df = self.full_df
        if since is None:
            return df.iloc[-limit:].reset_index(drop=True)
        since_ts = pd.Timestamp(since, unit="ms")
        mask = df["timestamp"] >= since_ts
        return df.loc[mask].iloc[:limit].reset_index(drop=True)


def test_timeframe_to_minutes():
    assert timeframe_to_minutes("1m") == 1
    assert timeframe_to_minutes("15m") == 15
    assert timeframe_to_minutes("4h") == 240
    assert timeframe_to_minutes("1d") == 1440


def test_fetch_full_history_paginates_across_batches():
    exchange = FakeHistoryExchange(total_candles=2500)
    result = fetch_full_history(exchange, "OSCUSDT", "4h", max_candles=2200, batch_size=1000)
    assert len(result) == 2200
    assert result["timestamp"].is_monotonic_increasing
    assert result["timestamp"].is_unique


def test_fetch_full_history_stops_when_exchange_runs_out():
    exchange = FakeHistoryExchange(total_candles=800)
    result = fetch_full_history(exchange, "OSCUSDT", "4h", max_candles=5000, batch_size=1000)
    assert len(result) == 800  # borsada bundan fazlası yok, sonsuz döngüye girmedi


def test_compute_metrics_known_values():
    pnls = [2.0, -1.0, 3.0, -2.0, 1.0]
    metrics = compute_metrics(pnls, num_candles=1000, timeframe_minutes=240)
    assert metrics.num_trades == 5
    assert metrics.win_rate_pct == pytest.approx(60.0)
    assert metrics.total_return_pct == pytest.approx(3.0)
    assert metrics.avg_win_pct == pytest.approx(2.0)
    assert metrics.avg_loss_pct == pytest.approx(-1.5)
    assert metrics.profit_factor == pytest.approx(6.0 / 3.0)
    assert metrics.max_drawdown_pct >= 0


def test_compute_metrics_empty_returns_zeroed_report():
    metrics = compute_metrics([], num_candles=500, timeframe_minutes=240)
    assert metrics.num_trades == 0
    assert metrics.sharpe_ratio is None
    assert metrics.profit_factor is None


def test_discover_sufficient_history_stops_early_when_enough_trades():
    exchange = FakeHistoryExchange(total_candles=5000, oscillation=True)
    # Sık tetiklenen, TP'si düşük bir DCA -> çok sayıda kısa işlem üretir
    dca_params = DCAParams(
        base_order_size=10,
        deviation_pct=10,
        deviation_multiplier=1,
        order_size_multiplier=1,
        max_safety_orders=0,
        take_profit_pct=0.5,
        direction="long",
    )
    request = BacktestRequest(
        symbol="OSCUSDT", timeframe="4h", dca_params=dca_params, min_trades=10, max_candles=5000, initial_candles=500
    )
    ohlcv, sufficiency = _discover_sufficient_history(exchange, request)
    assert sufficiency.sufficient is True
    assert sufficiency.trades_found >= 10
    assert len(ohlcv) < 5000  # tüm maksimuma çıkmadan yeterli örnekleme ulaştı


def test_discover_sufficient_history_reports_insufficient_when_capped():
    exchange = FakeHistoryExchange(total_candles=5000, oscillation=False)  # tek yönlü trend -> az işlem
    dca_params = DCAParams(
        base_order_size=10,
        deviation_pct=1,
        deviation_multiplier=1,
        order_size_multiplier=1,
        max_safety_orders=0,
        take_profit_pct=50.0,  # neredeyse hiç tetiklenmez
        direction="short",
    )
    request = BacktestRequest(
        symbol="OSCUSDT", timeframe="4h", dca_params=dca_params, min_trades=50, max_candles=1000, initial_candles=500
    )
    _, sufficiency = _discover_sufficient_history(exchange, request)
    assert sufficiency.sufficient is False


def test_run_backtest_report_with_strategy_produces_train_test_split():
    exchange = FakeHistoryExchange(total_candles=3000, oscillation=True)
    request = BacktestRequest(
        symbol="OSCUSDT",
        timeframe="4h",
        strategy=RSI_OVERSOLD_BOUNCE,
        min_trades=5,
        max_candles=3000,
        initial_candles=800,
        train_ratio=0.7,
    )
    report = run_backtest_report(exchange, request)

    assert report.data_sufficiency.candles_used > 0
    assert report.full_period_metrics is not None
    if report.train_metrics and report.test_metrics:
        total_candles = report.data_sufficiency.candles_used
        assert report.train_metrics.num_trades + report.test_metrics.num_trades <= total_candles


def test_backtest_request_requires_exactly_one_target():
    with pytest.raises(ValueError):
        BacktestRequest(symbol="OSCUSDT")
    with pytest.raises(ValueError):
        BacktestRequest(symbol="OSCUSDT", dca_params=DCAParams(
            base_order_size=10, deviation_pct=1, deviation_multiplier=1,
            order_size_multiplier=1, max_safety_orders=0, take_profit_pct=1,
        ), strategy=RSI_OVERSOLD_BOUNCE)
