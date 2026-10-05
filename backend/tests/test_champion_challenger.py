import pandas as pd
import pytest

from app.backtest import system_runner
from app.core.config import settings
from app.ml import train as train_module
from app.ml.model import SignalModel


class _Report:
    def __init__(self, pnl, dd, trades=40):
        self.total_pnl_pct = pnl
        self.max_drawdown_pct = dd
        self.trades_closed = trades


@pytest.fixture
def champion_on_disk(tmp_path, monkeypatch):
    path = tmp_path / "signal_model.joblib"
    path.write_bytes(b"x")
    champion = SignalModel()
    monkeypatch.setattr(train_module, "DEFAULT_MODEL_PATH", path)
    monkeypatch.setattr(train_module.SignalModel, "load_from", classmethod(lambda cls, p=None: champion))
    return champion


def _scores(monkeypatch, champion, challenger_report, champion_report):
    windows = []

    def fake_run(exchange, model, meta, request, persist=True, eval_window=None, **kwargs):
        windows.append(eval_window)
        return champion_report if model is champion else challenger_report

    monkeypatch.setattr(system_runner, "run_system_backtest", fake_run)
    return windows


def test_challenger_worse_than_champion_is_rejected(champion_on_disk, monkeypatch):
    windows = _scores(monkeypatch, champion_on_disk, _Report(pnl=2.0, dd=4.0), _Report(pnl=6.0, dd=2.0))
    ok, reason = train_module._champion_challenger_verdict(None, SignalModel(), pd.Timestamp("2026-06-01", tz="UTC"))
    assert ok is False
    assert "champion/challenger" in reason
    # ikisi de AYNI holdout penceresinde ölçüldü
    assert windows[0] == windows[1]
    assert windows[0][0] == pd.Timestamp("2026-06-01")


def test_challenger_at_least_as_good_is_accepted(champion_on_disk, monkeypatch):
    _scores(monkeypatch, champion_on_disk, _Report(pnl=6.0, dd=2.0), _Report(pnl=2.0, dd=2.0))
    ok, reason = train_module._champion_challenger_verdict(None, SignalModel(), pd.Timestamp("2026-06-01"))
    assert ok is True and reason is None


def test_no_champion_means_accept(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "DEFAULT_MODEL_PATH", tmp_path / "missing.joblib")
    assert train_module._champion_challenger_verdict(None, SignalModel(), pd.Timestamp("2026-06-01")) == (True, None)


def test_champion_evaluation_failure_does_not_block_new_model(champion_on_disk, monkeypatch):
    def failing(exchange, model, *a, **k):
        if model is champion_on_disk:
            raise ValueError("eski özellik listesi")
        return _Report(1.0, 1.0)

    monkeypatch.setattr(system_runner, "run_system_backtest", failing)
    assert train_module._champion_challenger_verdict(None, SignalModel(), pd.Timestamp("2026-06-01")) == (True, None)


def test_rejected_challenger_is_not_saved_and_status_untouched(tmp_path, monkeypatch):
    from tests.test_xgboost import TrendExchange

    monkeypatch.setattr(train_module, "DEFAULT_MODEL_PATH", tmp_path / "signal_model.joblib")
    monkeypatch.setattr(settings, "ml_champion_challenger_enabled", True)
    monkeypatch.setattr(settings, "ml_min_balanced_accuracy", 0.0)
    monkeypatch.setattr(train_module, "_champion_challenger_verdict", lambda *a: (False, "champion/challenger: daha kötü"))
    saved, statuses = [], []
    monkeypatch.setattr(SignalModel, "save", lambda self, *a, **k: saved.append(1))
    monkeypatch.setattr(train_module, "write_model_status", lambda *a, **k: statuses.append(k))

    result = train_module.train_signal_model_validated(
        TrendExchange(seed=5), ["UPUSDT"], horizon=5, threshold_pct=0.5, timeframe="4h", lookback=400, walk_forward_splits=3
    )

    assert result.accepted is False
    assert "champion/challenger" in result.rejection_reason
    assert saved == [] and statuses == []
