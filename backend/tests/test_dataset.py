import pandas as pd

import app.db.repository as repository
from app.core.config import settings
from app.ml.dataset import _persist_feature_snapshots
from app.ml.features import FEATURE_COLUMNS


def _fake_features_frame() -> pd.DataFrame:
    return pd.DataFrame({col: [1.0, 2.0, 3.0] for col in FEATURE_COLUMNS})


def test_persist_feature_snapshots_is_noop_when_disabled(monkeypatch):
    """Bkz. README 'OOM üretim olayı' — LSTM/RL şu an devre dışı olduğundan
    bu tabloyu okuyan hiçbir eğitim kodu yok; varsayılan (False) iken
    hiçbir DB yazma çağrısı yapılmamalı."""
    monkeypatch.setattr(settings, "ml_persist_feature_snapshots", False)
    calls = []
    monkeypatch.setattr(repository, "record_feature_snapshots_bulk", lambda *a, **k: calls.append((a, k)))

    _persist_feature_snapshots("BTC/USDT:USDT", "1h", _fake_features_frame())

    assert calls == []


def test_persist_feature_snapshots_writes_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "ml_persist_feature_snapshots", True)
    calls = []
    monkeypatch.setattr(repository, "record_feature_snapshots_bulk", lambda *a, **k: calls.append((a, k)))

    _persist_feature_snapshots("BTC/USDT:USDT", "1h", _fake_features_frame())

    assert len(calls) == 1
    assert calls[0][0][0] == "BTC/USDT:USDT"
