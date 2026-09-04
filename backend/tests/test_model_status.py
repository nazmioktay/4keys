from app.ml.model_status import is_model_enabled, read_model_status, write_model_status


def test_is_model_enabled_false_when_model_file_missing(tmp_path):
    missing = tmp_path / "nope.joblib"
    assert is_model_enabled(missing) is False


def test_is_model_enabled_false_when_status_file_missing(tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_text("fake")
    assert is_model_enabled(model_path) is False


def test_write_then_read_status_roundtrip(tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_text("fake")

    write_model_status(model_path, enabled=True, balanced_accuracy=0.45)

    assert is_model_enabled(model_path) is True
    status = read_model_status(model_path)
    assert status["enabled"] is True
    assert status["balanced_accuracy"] == 0.45
    assert status["reason"] is None
    assert "updated_at" in status


def test_write_rejected_status_disables_even_with_existing_model_file(tmp_path):
    """Bir model dosyası (eski, iyi bir eğitimden kalma) diskte olsa bile,
    EN SON eğitim eşiğin altında kaldıysa `is_model_enabled` False dönmeli
    — 'düşük puanlı modelin eski verisini kullanma' kuralı."""
    model_path = tmp_path / "model.joblib"
    model_path.write_text("eski iyi model")

    write_model_status(model_path, enabled=True, balanced_accuracy=0.45)
    assert is_model_enabled(model_path) is True

    write_model_status(model_path, enabled=False, balanced_accuracy=0.333, reason="eşiğin altında")
    assert is_model_enabled(model_path) is False

    status = read_model_status(model_path)
    assert status["reason"] == "eşiğin altında"


def test_read_model_status_returns_none_when_missing(tmp_path):
    assert read_model_status(tmp_path / "missing.joblib") is None
