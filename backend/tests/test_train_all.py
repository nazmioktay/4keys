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


def test_train_all_models_skip_steps_never_calls_the_skipped_function(monkeypatch):
    """Regresyon (bkz. README 'OOM üretim olayı'): `skip_steps` verilen bir
    adım HİÇ ÇAĞRILMAMALI — yalnızca sonucu "atlandı" olarak işaretlenmek
    yetmez, asıl amaç o adımın PAHALI (LSTM için torch) işini hiç
    yapmamak. `train_lstm_signal_model`'i çağrılırsa patlayan bir stub'la
    değiştirip `skip_steps={"lstm"}` ile çağrıldığında bu stub'ın HİÇ
    tetiklenmediğini doğrular."""

    def _lstm_must_not_be_called(*_a, **_k):
        raise AssertionError("skip_steps={'lstm'} verilmişken train_lstm_signal_model ÇAĞRILDI")

    monkeypatch.setattr(train_module, "train_signal_model_validated", lambda *a, **k: _FakePrimaryResult())
    monkeypatch.setattr(train_module, "train_meta_label_model", lambda *a, **k: (object(), 500))
    monkeypatch.setattr(train_module, "train_lstm_signal_model", _lstm_must_not_be_called)
    monkeypatch.setattr(train_module, "train_online_signal_model", lambda *a, **k: (object(), _FakeOnlineReport()))
    monkeypatch.setattr(
        train_module, "train_signal_models_by_regime", lambda *a, **k: (object(), [_FakeRegimeResult(0)])
    )

    results = train_module.train_all_models(object(), ["BTC/USDT:USDT"], skip_steps=frozenset({"lstm"}))
    by_step = {r.step: r for r in results}

    assert by_step["lstm"].ok is True
    assert "atlandı" in by_step["lstm"].detail
    # Bağımsız diğer adımlar normal şekilde çalışmaya devam etmeli.
    assert by_step["xgboost"].ok is True
    assert by_step["meta_label"].ok is True
    assert by_step["online"].ok is True
    assert by_step["regime"].ok is True


def test_train_all_models_skip_steps_can_skip_multiple_independent_steps(monkeypatch):
    def _must_not_be_called(name):
        def _fn(*_a, **_k):
            raise AssertionError(f"skip_steps verilmişken {name} ÇAĞRILDI")

        return _fn

    monkeypatch.setattr(train_module, "train_signal_model_validated", lambda *a, **k: _FakePrimaryResult())
    monkeypatch.setattr(train_module, "train_meta_label_model", lambda *a, **k: (object(), 500))
    monkeypatch.setattr(train_module, "train_lstm_signal_model", _must_not_be_called("train_lstm_signal_model"))
    monkeypatch.setattr(train_module, "train_online_signal_model", _must_not_be_called("train_online_signal_model"))
    monkeypatch.setattr(
        train_module, "train_signal_models_by_regime", lambda *a, **k: (object(), [_FakeRegimeResult(0)])
    )

    results = train_module.train_all_models(object(), ["BTC/USDT:USDT"], skip_steps=frozenset({"lstm", "online"}))
    by_step = {r.step: r for r in results}

    assert "atlandı" in by_step["lstm"].detail
    assert "atlandı" in by_step["online"].detail
    assert by_step["xgboost"].ok is True
    assert by_step["meta_label"].ok is True
    assert by_step["regime"].ok is True


def test_train_all_models_lookback_override_forwarded_to_active_steps(monkeypatch):
    """Bkz. README "temel sadeleşme": `lookback`, kutunun belleğine göre mum
    sayısını `.env` değişikliği olmadan hızlıca denemek için eklendi —
    XGBoost/meta-label/online çağrılarına GERÇEKTEN ulaştığını doğrular."""
    seen: dict[str, int | None] = {}

    def _fake_primary(exchange, symbols, lookback=None, **_k):
        seen["xgboost"] = lookback
        return _FakePrimaryResult()

    def _fake_meta(exchange, symbols, primary_model, lookback=None, **_k):
        seen["meta_label"] = lookback
        return (object(), 500)

    def _fake_online(exchange, symbols, window_size=500, lookback=None, **_k):
        seen["online"] = lookback
        return (object(), _FakeOnlineReport())

    monkeypatch.setattr(train_module, "train_signal_model_validated", _fake_primary)
    monkeypatch.setattr(train_module, "train_meta_label_model", _fake_meta)
    monkeypatch.setattr(train_module, "train_online_signal_model", _fake_online)
    monkeypatch.setattr(train_module, "train_lstm_signal_model", lambda *a, **k: _FakeLSTMResult())
    monkeypatch.setattr(
        train_module, "train_signal_models_by_regime", lambda *a, **k: (object(), [_FakeRegimeResult(0)])
    )

    train_module.train_all_models(object(), ["BTC/USDT:USDT"], lookback=15000)

    assert seen == {"xgboost": 15000, "meta_label": 15000, "online": 15000}
