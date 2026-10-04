import numpy as np
import pandas as pd

from app.ml import train as train_module
from app.ml.features import ALL_FEATURE_COLUMNS
from app.ml.meta_label import build_meta_dataset_out_of_fold
from app.ml.online_model import OnlineSignalModel, run_prequential_evaluation


def _xy(n: int, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(ALL_FEATURE_COLUMNS))), columns=ALL_FEATURE_COLUMNS)
    y = pd.Series(rng.choice([-1, 0, 1], size=n))
    return X, y


def test_prequential_learns_each_label_only_after_delay(monkeypatch):
    events: list[tuple[str, int]] = []
    original_learn = OnlineSignalModel.learn_one
    original_predict = OnlineSignalModel.predict_one

    def learn(self, feats, label):
        events.append(("learn", int(feats["row"])))
        original_learn(self, feats, label)

    def predict(self, feats):
        events.append(("predict", int(feats["row"])))
        return original_predict(self, feats)

    monkeypatch.setattr(OnlineSignalModel, "learn_one", learn)
    monkeypatch.setattr(OnlineSignalModel, "predict_one", predict)

    n, delay = 30, 5
    X = pd.DataFrame({"row": range(n), "x": np.linspace(0, 1, n)})
    y = pd.Series([1, -1] * (n // 2))
    run_prequential_evaluation(X, y, n_models=2, window_size=10, label_delay=delay)

    for i in range(n):
        predicted_at = events.index(("predict", i))
        learned_before = {r for kind, r in events[:predicted_at] if kind == "learn"}
        # satır i tahmin edilirken yalnızca etiketi gerçekleşmiş (i - delay'den önceki) satırlar öğrenilmiş olmalı
        assert learned_before == set(range(max(0, i - delay)))
    assert sorted(r for kind, r in events if kind == "learn") == list(range(n))


def test_prequential_snapshot_has_not_seen_rows_from_snapshot_point(monkeypatch):
    learned: list[int] = []
    original_learn = OnlineSignalModel.learn_one

    def learn(self, feats, label):
        learned.append(int(feats["row"]))
        original_learn(self, feats, label)

    monkeypatch.setattr(OnlineSignalModel, "learn_one", learn)
    X = pd.DataFrame({"row": range(40), "x": np.linspace(0, 1, 40)})
    y = pd.Series([1, -1] * 20)
    snapshots: list[tuple[OnlineSignalModel, int]] = []
    run_prequential_evaluation(
        X, y, n_models=2, label_delay=3, snapshot_at_row=25, on_snapshot=lambda m: snapshots.append((m, len(learned)))
    )
    assert len(snapshots) == 1
    _, learned_count = snapshots[0]
    assert learned_count == 25 - 3


class _FakePrimary:
    def fit(self, X, y):
        pass

    def predict_batch(self, X):
        preds = np.where(np.arange(len(X)) % 3 == 0, 0, 1)
        return preds, np.full(len(X), 0.6)


def test_meta_dataset_uses_only_out_of_fold_directional_rows():
    X, y = _xy(600)
    time_frac = pd.Series(np.linspace(0, 1, 600))
    meta_X, meta_y = build_meta_dataset_out_of_fold(X, y, time_frac, _FakePrimary, n_splits=5, embargo_frac=0.02)

    first_test_start = 1 / 6
    assert len(meta_X) > 0
    assert (time_frac.loc[meta_X.index] > first_test_start).all()  # ilk eğitim bloğu hiç meta örneği olmaz
    assert set(meta_y.unique()) <= {0, 1}
    assert "primary_confidence" in meta_X.columns


def test_train_meta_label_model_never_sees_holdout(monkeypatch):
    n = 1000
    X, y = _xy(n, seed=1)
    time_frac = pd.Series(np.linspace(0, 1, n))
    stamps = pd.Series(pd.date_range("2025-01-01", periods=n, freq="1h"))
    monkeypatch.setattr(
        train_module,
        "build_training_dataset_with_time",
        lambda *a, **k: (X, y, time_frac, stamps, pd.Series(["BTC/USDT:USDT"] * n)),
    )
    seen: dict = {}

    def fake_oof(X_in, y_in, tf_in, factory, n_splits, embargo_frac):
        seen["rows"] = len(X_in)
        seen["max_tf"] = float(tf_in.max())
        meta_X = X_in.iloc[:200][list(X_in.columns)].copy()
        meta_X["primary_confidence"] = 0.6
        return meta_X, pd.Series([0, 1] * 100)

    monkeypatch.setattr(train_module, "build_meta_dataset_out_of_fold", fake_oof)
    monkeypatch.setattr(train_module.MetaLabelModel, "fit", lambda self, a, b: None)
    monkeypatch.setattr(train_module.MetaLabelModel, "save", lambda self, path=None: None)

    primary = train_module.SignalModel()
    train_module.train_meta_label_model(object(), ["BTC/USDT:USDT"], primary, holdout_frac=0.2)

    assert seen["rows"] == int((time_frac <= 0.8).sum())
    assert seen["max_tf"] <= 1.0 + 1e-9
