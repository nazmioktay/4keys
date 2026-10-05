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


# --- Meta-label / online model sınırı üretimdeki birincil modele sabitlenir (denetim bulgusu 2) ---

from types import SimpleNamespace

import numpy as np

from app.ml.features import ALL_FEATURE_COLUMNS
from app.ml.model_status import write_model_status

_N_ROWS = 400
_BOUNDARY_ROW = 300  # üretimdeki modelin holdout sınırı; taze verinin %80'i (320) bundan SONRA


def _synthetic_dataset(symbol="BTC/USDT:USDT"):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(_N_ROWS, len(ALL_FEATURE_COLUMNS))), columns=ALL_FEATURE_COLUMNS)
    y = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=_N_ROWS))
    time_frac = pd.Series(np.arange(_N_ROWS) / (_N_ROWS - 1))
    stamps = pd.Series(pd.date_range("2026-01-01", periods=_N_ROWS, freq="1h"))
    return X, y, time_frac, stamps, pd.Series([symbol] * _N_ROWS)


@pytest.fixture
def production_boundary(tmp_path, monkeypatch):
    """Diskte üretim modeli + ondan ÖNCE kaydedilmiş holdout sınırı; taze veri seti daha ileri."""
    path = tmp_path / "signal_model.joblib"
    path.write_bytes(b"x")
    monkeypatch.setattr(train_module, "DEFAULT_MODEL_PATH", path)
    dataset = _synthetic_dataset()
    boundary = dataset[3].iloc[_BOUNDARY_ROW]
    write_model_status(path, enabled=True, balanced_accuracy=0.42, holdout_start_time=boundary.isoformat())
    monkeypatch.setattr(train_module, "build_training_dataset_with_time", lambda *a, **k: dataset)
    return boundary


def test_meta_label_trains_only_on_rows_before_production_holdout_boundary(production_boundary, monkeypatch):
    seen = {}

    def fake_oof(X, y, time_frac, factory, **kwargs):
        seen["rows"] = len(X)
        return X.iloc[:100], pd.Series([0, 1] * 50)

    class _Meta:
        def fit(self, *a, **k):
            pass

        def save(self, *a, **k):
            pass

    monkeypatch.setattr(train_module, "build_meta_dataset_out_of_fold", fake_oof)
    monkeypatch.setattr(train_module, "MetaLabelModel", _Meta)

    train_module.train_meta_label_model(None, ["BTC/USDT:USDT"], SignalModel(), holdout_frac=0.2)

    assert seen["rows"] == _BOUNDARY_ROW  # önceki davranış: 320 (taze verinin %80'i) -> 20 satır sızıntı


def test_online_snapshot_is_taken_at_production_holdout_boundary(production_boundary, monkeypatch):
    seen = {}

    class _Online:
        def save(self, *a, **k):
            pass

    def fake_prequential(X, y, **kwargs):
        seen["snapshot_at_row"] = kwargs["snapshot_at_row"]
        report = SimpleNamespace(
            overall_balanced_accuracy=0.5, overall_accuracy=0.5, rows_used=len(X), accepted=True, rejection_reason=None
        )
        return _Online(), report

    monkeypatch.setattr(train_module, "run_prequential_evaluation", fake_prequential)

    train_module.train_online_signal_model(None, ["BTC/USDT:USDT"], persist=False, holdout_frac=0.2)

    assert seen["snapshot_at_row"] == _BOUNDARY_ROW  # önceki davranış: 320


def test_auto_retrain_does_not_retrain_meta_after_rejected_challenger(monkeypatch):
    from app.scheduler import jobs, status

    class _Rejected:
        rows_used = 1000
        accepted = False
        rejection_reason = "champion/challenger: daha kötü"

        class out_of_sample:
            balanced_accuracy = 0.42

    called = []
    monkeypatch.setattr(jobs, "get_exchange", lambda *_a, **_k: object())
    monkeypatch.setattr(jobs, "_auto_retrain_symbols", lambda exchange: ["BTC/USDT:USDT"])
    monkeypatch.setattr(jobs, "train_signal_model_validated", lambda *a, **k: _Rejected())
    monkeypatch.setattr(jobs, "DEFAULT_META_MODEL_PATH", type("P", (), {"exists": staticmethod(lambda: True)})())
    monkeypatch.setattr(jobs.SignalModel, "load_from", classmethod(lambda cls, path=None: object()))
    monkeypatch.setattr(jobs, "train_meta_label_model", lambda *a, **k: called.append(1) or (object(), 500))

    jobs.job_auto_retrain()

    assert called == []
    assert "meta-label yeniden eğitilmedi" in status.get_all()[jobs.AUTO_RETRAIN_JOB_ID].detail
