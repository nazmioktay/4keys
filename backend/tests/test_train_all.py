from app.ml import train as train_module


class _FakePrimaryResult:
    model = object()
    rows_used = 1000
    accepted = True
    rejection_reason = None

    class out_of_sample:
        balanced_accuracy = 0.42
        true_class_counts = {"1.0": 50, "0.0": 30, "-1.0": 20}
        predicted_class_counts = {"1.0": 45, "0.0": 35, "-1.0": 20}


class _FakeLSTMResult:
    rows_used = 800
    accepted = True
    rejection_reason = None

    class out_of_sample:
        balanced_accuracy = 0.41


class _FakeOnlineReport:
    rows_used = 900
    overall_balanced_accuracy = 0.49
    accepted = True
    rejection_reason = None


class _FakeRegimeResult:
    def __init__(self, regime, error=None):
        self.regime = regime
        self.rows_used = 300
        self.error = error


def test_train_all_models_runs_all_five_steps_in_order(monkeypatch):
    monkeypatch.setattr(train_module, "train_signal_model_validated", lambda *a, **k: _FakePrimaryResult())
    monkeypatch.setattr(train_module, "train_meta_label_model", lambda *a, **k: (object(), 500))
    monkeypatch.setattr(train_module, "train_lstm_signal_model", lambda *a, **k: _FakeLSTMResult())
    monkeypatch.setattr(train_module, "train_online_signal_model", lambda *a, **k: (object(), _FakeOnlineReport()))
    monkeypatch.setattr(
        train_module,
        "train_signal_models_by_regime",
        lambda *a, **k: (object(), [_FakeRegimeResult(0), _FakeRegimeResult(1), _FakeRegimeResult(2)]),
    )

    results = train_module.train_all_models(object(), ["BTC/USDT:USDT"])

    steps = [r.step for r in results]
    assert steps == ["xgboost", "meta_label", "lstm", "online", "regime"]
    assert all(r.ok for r in results)
    assert "1000" in results[0].detail
    assert "500" in results[1].detail
    assert "800" in results[2].detail
    assert "900" in results[3].detail
    assert "rejim 0" in results[4].detail


def test_train_all_models_skips_meta_when_primary_fails(monkeypatch):
    def _boom(*_a, **_k):
        raise ValueError("yetersiz veri")

    monkeypatch.setattr(train_module, "train_signal_model_validated", _boom)
    monkeypatch.setattr(train_module, "train_lstm_signal_model", lambda *a, **k: _FakeLSTMResult())
    monkeypatch.setattr(train_module, "train_online_signal_model", lambda *a, **k: (object(), _FakeOnlineReport()))
    monkeypatch.setattr(
        train_module, "train_signal_models_by_regime", lambda *a, **k: (object(), [_FakeRegimeResult(0)])
    )

    results = train_module.train_all_models(object(), ["BTC/USDT:USDT"])

    by_step = {r.step: r for r in results}
    assert by_step["xgboost"].ok is False
    assert by_step["meta_label"].ok is False
    assert "atlandı" in by_step["meta_label"].detail
    # Bağımsız adımlar birincil model hatasından etkilenmemeli.
    assert by_step["lstm"].ok is True
    assert by_step["online"].ok is True
    assert by_step["regime"].ok is True


def test_train_all_models_one_independent_step_failure_does_not_block_others(monkeypatch):
    def _lstm_boom(*_a, **_k):
        raise ValueError("LSTM için yeterli veri yok")

    monkeypatch.setattr(train_module, "train_signal_model_validated", lambda *a, **k: _FakePrimaryResult())
    monkeypatch.setattr(train_module, "train_meta_label_model", lambda *a, **k: (object(), 500))
    monkeypatch.setattr(train_module, "train_lstm_signal_model", _lstm_boom)
    monkeypatch.setattr(train_module, "train_online_signal_model", lambda *a, **k: (object(), _FakeOnlineReport()))
    monkeypatch.setattr(
        train_module, "train_signal_models_by_regime", lambda *a, **k: (object(), [_FakeRegimeResult(0)])
    )

    results = train_module.train_all_models(object(), ["BTC/USDT:USDT"])
    by_step = {r.step: r for r in results}

    assert by_step["xgboost"].ok is True
    assert by_step["meta_label"].ok is True
    assert by_step["lstm"].ok is False
    assert "yeterli veri yok" in by_step["lstm"].detail
    assert by_step["online"].ok is True
    assert by_step["regime"].ok is True


def test_train_all_models_skips_meta_when_primary_rejected(monkeypatch):
    class _RejectedPrimary:
        model = object()
        rows_used = 1000
        accepted = False
        rejection_reason = "out_of_sample_balanced_accuracy (0.333) eşiğin (0.37) altında"

        class out_of_sample:
            balanced_accuracy = 0.333
            true_class_counts = {"1.0": 40, "0.0": 40, "-1.0": 40}
            predicted_class_counts = {"0.0": 120}

    monkeypatch.setattr(train_module, "train_signal_model_validated", lambda *a, **k: _RejectedPrimary())
    monkeypatch.setattr(train_module, "train_lstm_signal_model", lambda *a, **k: _FakeLSTMResult())
    monkeypatch.setattr(train_module, "train_online_signal_model", lambda *a, **k: (object(), _FakeOnlineReport()))
    monkeypatch.setattr(
        train_module, "train_signal_models_by_regime", lambda *a, **k: (object(), [_FakeRegimeResult(0)])
    )

    results = train_module.train_all_models(object(), ["BTC/USDT:USDT"])
    by_step = {r.step: r for r in results}

    # Reddedilen model de "teknik olarak" başarıyla eğitildi (exception yok),
    # ama detay metninde REDDEDİLDİ olarak açıkça işaretlenmeli.
    assert by_step["xgboost"].ok is True
    assert "REDDEDİLDİ" in by_step["xgboost"].detail
    # Reddedilen (kaydedilmeyen) bir modelin üzerine meta-label eğitmek
    # tutarsız olurdu -> meta-label adımı atlanmalı.
    assert by_step["meta_label"].ok is False
    assert "reddedildi" in by_step["meta_label"].detail
