"""`tie_neutral` / `label_cost_pct` (plan 3.1) eğitim hattına bağlı mı? (denetim bulgusu 4)

Casus (spy) testleri: asıl veri/etiket üretimi sahtelenir, yalnızca iki seçeneğin
her eğitim yolunda AYNI değerlerle veri kurucuya ulaştığı doğrulanır."""

import numpy as np
import pandas as pd
import pytest

from app.backtest import walk_forward
from app.backtest.schemas import SystemBacktestRequest
from app.core.config import settings
from app.ml import dataset as dataset_module
from app.ml import train as train_module
from app.ml.model import SignalModel


class _Stop(Exception):
    """Casus, argümanları kaydedip eğitimi durdurmak için fırlatır."""


def _recorder(sink: list, exc=_Stop):
    def spy(*args, **kwargs):
        sink.append(kwargs)
        raise exc("spy")

    return spy


@pytest.fixture
def labeling_options(monkeypatch):
    monkeypatch.setattr(settings, "ml_label_tie_neutral", True)
    monkeypatch.setattr(settings, "ml_label_cost_pct", 0.12)
    return {"tie_neutral": True, "label_cost_pct": 0.12}


def test_ensemble_labeling_defaults_keep_current_behaviour():
    labeling = train_module.ensemble_labeling()
    assert labeling["tie_neutral"] is False
    assert labeling["label_cost_pct"] == 0.0
    assert labeling["labeling_method"] == "atr_triple_barrier"  # mevcut ensemble etiketlemesi değişmedi


def test_ensemble_labeling_reads_settings(labeling_options):
    labeling = train_module.ensemble_labeling()
    assert labeling["tie_neutral"] is True and labeling["label_cost_pct"] == 0.12


def test_primary_model_training_passes_options(labeling_options, monkeypatch):
    calls = []
    monkeypatch.setattr(train_module, "build_training_dataset_with_time", _recorder(calls))
    with pytest.raises(_Stop):
        train_module.train_signal_model_validated(None, ["BTC/USDT:USDT"], persist=False, **train_module.ensemble_labeling())
    assert calls[0]["tie_neutral"] is True and calls[0]["label_cost_pct"] == 0.12


def test_meta_label_training_passes_same_options_as_primary(labeling_options, monkeypatch):
    calls = []
    monkeypatch.setattr(train_module, "build_training_dataset_with_time", _recorder(calls))
    with pytest.raises(_Stop):
        train_module.train_meta_label_model(None, ["BTC/USDT:USDT"], SignalModel(), **train_module.ensemble_labeling())
    assert calls[0]["tie_neutral"] is True and calls[0]["label_cost_pct"] == 0.12


def test_online_model_training_passes_same_options_as_primary(labeling_options, monkeypatch):
    calls = []
    monkeypatch.setattr(train_module, "build_training_dataset_with_time", _recorder(calls))
    with pytest.raises(_Stop):
        train_module.train_online_signal_model(None, ["BTC/USDT:USDT"], persist=False, **train_module.ensemble_labeling())
    assert calls[0]["tie_neutral"] is True and calls[0]["label_cost_pct"] == 0.12


def test_sequence_and_regime_members_accept_and_pass_options(labeling_options, monkeypatch):
    # `**ensemble_labeling()` ile çağrıldıkları için TypeError vermemeli ve aynı etiketi öğrenmeli.
    seq_calls, regime_calls = [], []
    monkeypatch.setattr(train_module, "build_sequence_dataset", _recorder(seq_calls))
    monkeypatch.setattr(train_module, "build_regime_labeled_dataset", _recorder(regime_calls))
    monkeypatch.setattr(train_module, "fit_regime_model", lambda *a, **k: (object(), []))
    with pytest.raises(_Stop):
        train_module.train_lstm_signal_model(None, ["BTC/USDT:USDT"], persist=False, **train_module.ensemble_labeling())
    with pytest.raises(_Stop):
        train_module.train_signal_models_by_regime(None, ["BTC/USDT:USDT"], persist=False, **train_module.ensemble_labeling())
    assert seq_calls[0]["tie_neutral"] is True and seq_calls[0]["label_cost_pct"] == 0.12
    assert regime_calls[0]["tie_neutral"] is True and regime_calls[0]["label_cost_pct"] == 0.12


def test_dataset_builder_forwards_options_to_symbol_frame(monkeypatch):
    ohlcv = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=100, freq="1h"),
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
        }
    )
    calls = []
    monkeypatch.setattr(dataset_module, "fetch_ohlcv_cached", lambda *a, **k: ohlcv)
    monkeypatch.setattr(dataset_module, "load_macro_history", lambda: None)
    monkeypatch.setattr(dataset_module, "build_symbol_frame", lambda *a, **k: calls.append(k) or pd.DataFrame())

    dataset_module.build_training_dataset_with_time(None, ["X"], "1h", 100, tie_neutral=True, label_cost_pct=0.12)

    assert calls[0]["tie_neutral"] is True and calls[0]["label_cost_pct"] == 0.12


def test_walk_forward_default_labeling_uses_settings(labeling_options, monkeypatch):
    ohlcv = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=500, freq="1h"), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}
    )
    calls = []
    monkeypatch.setattr(walk_forward, "build_symbol_frame", _recorder(calls))
    with pytest.raises(_Stop):
        walk_forward.prepare_walk_forward(None, SystemBacktestRequest(), ohlcv=ohlcv)
    assert calls[0]["tie_neutral"] is True and calls[0]["label_cost_pct"] == 0.12


@pytest.mark.parametrize(
    "train",
    [
        lambda: train_module.train_signal_model_validated(None, ["BTC/USDT:USDT"], persist=False),
        lambda: train_module.train_meta_label_model(None, ["BTC/USDT:USDT"], SignalModel()),
        lambda: train_module.train_online_signal_model(None, ["BTC/USDT:USDT"], persist=False),
    ],
    ids=["xgboost", "meta_label", "online"],
)
def test_training_without_explicit_options_follows_settings(labeling_options, monkeypatch, train):
    # Elle çağrılan rotalar (/ml/train, /ml/train-meta, ...) ve taramalar seçenekleri vermese de
    # üç model AYNI ayarı kullanır — birbirinden sapamaz.
    calls = []
    monkeypatch.setattr(train_module, "build_training_dataset_with_time", _recorder(calls))
    with pytest.raises(_Stop):
        train()
    assert calls[0]["tie_neutral"] is True and calls[0]["label_cost_pct"] == 0.12
