import pandas as pd
import pytest

from app.backtest import walk_forward as wf
from app.backtest.schemas import SystemBacktestRequest
from app.ml.model import SignalModel

from tests.test_system_backtest import FakeOscillatingExchange


def test_walk_forward_trains_only_on_past_and_keeps_final_slice_untouched(monkeypatch):
    exchange = FakeOscillatingExchange(total_candles=3000)
    request = SystemBacktestRequest(symbol="BTC/USDT:USDT", candles=3000, timeframe="1h")

    fit_sizes: list[int] = []
    original_fit = SignalModel.fit

    def recording_fit(self, X, y):
        fit_sizes.append(len(X))
        original_fit(self, X, y)

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    original_run = wf.run_system_backtest

    def recording_run(*args, **kwargs):
        windows.append(kwargs["eval_window"])
        return original_run(*args, **kwargs)

    monkeypatch.setattr(SignalModel, "fit", recording_fit)
    monkeypatch.setattr(wf, "run_system_backtest", recording_run)

    report = wf.run_walk_forward_system_backtest(
        exchange, request, n_folds=3, min_train_frac=0.4, final_test_frac=0.1, use_meta_label=False
    )

    assert len(report.folds) == 3
    assert all(f.error is None for f in report.folds), [f.error for f in report.folds]
    reserved = pd.Timestamp(report.final_test_reserved_from)
    for (start, end), fold in zip(windows, report.folds):
        assert end <= reserved  # ayrılmış son dilim hiçbir katmanda test edilmez
        # katman modeli yalnızca test başlangıcından (etiket ufku + embargo kadar) önceki satırları gördü
        assert fold.train_rows > 0
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert prev_end <= next_start  # test pencereleri çakışmaz
    # her katmanda eğitim seti büyür (genişleyen pencere)
    assert fit_sizes == sorted(fit_sizes)
    assert [f.train_rows for f in report.folds] == fit_sizes


def test_walk_forward_training_rows_respect_purge(monkeypatch):
    exchange = FakeOscillatingExchange(total_candles=2500)
    request = SystemBacktestRequest(symbol="BTC/USDT:USDT", candles=2500, timeframe="1h")
    captured: dict = {}
    original = wf.build_symbol_frame

    def capture_frame(*args, **kwargs):
        frame = original(*args, **kwargs)
        captured["frame"] = frame
        return frame

    monkeypatch.setattr(wf, "build_symbol_frame", capture_frame)
    report = wf.run_walk_forward_system_backtest(exchange, request, n_folds=2, use_meta_label=False, embargo_bars=10)

    stamps = pd.to_datetime(captured["frame"]["bar_timestamp"])
    for fold in report.folds:
        cutoff = pd.Timestamp(fold.test_start) - pd.Timedelta(hours=8 + 10)
        assert fold.train_rows == int((stamps < cutoff).sum())


def test_score_penalizes_drawdown():
    assert wf.score_result(10.0, 5.0) == pytest.approx(2.0)
    assert wf.score_result(10.0, 0.1) == pytest.approx(10.0)  # payda alt sınırı
