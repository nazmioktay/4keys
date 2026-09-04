import pytest

from app.engine.service import ModelNotTrained
from app.scheduler import jobs, status
from app.scheduler.scheduler import get_scheduler, start_scheduler, stop_scheduler


@pytest.fixture(autouse=True)
def _clean_scheduler_state(monkeypatch):
    # Zamanlayıcı gerçekten başlatıldığında screener taramasını hemen tetikler
    # (bkz. scheduler.py); arka plan thread'i bu testler sırasında gerçek ağ
    # çağrısı yapmasın diye iş fonksiyonları burada mock'lanır. stop_scheduler()
    # bu fixture'ın KENDİ teardown'ında (monkeypatch geri alınmadan ÖNCE)
    # çağrılır, böylece arka plan thread'i monkeypatch hâlâ aktifken durdurulur.
    monkeypatch.setattr(jobs, "refresh_screener", lambda: [])
    monkeypatch.setattr(jobs, "run_cycle_once", lambda: [])
    status.reset()
    stop_scheduler()
    yield
    stop_scheduler()
    status.reset()


def test_job_refresh_screener_records_success(monkeypatch):
    monkeypatch.setattr(jobs, "refresh_screener", lambda: [object(), object(), object()])
    jobs.job_refresh_screener()

    result = status.get_all()[jobs.SCREENER_REFRESH_JOB_ID]
    assert result.ok is True
    assert "3" in result.detail
    assert result.run_count == 1
    assert result.error_count == 0


def test_job_refresh_screener_records_failure(monkeypatch):
    def _boom():
        raise RuntimeError("borsa erişilemedi")

    monkeypatch.setattr(jobs, "refresh_screener", _boom)
    jobs.job_refresh_screener()  # zamanlayıcı thread'i çökmemeli

    result = status.get_all()[jobs.SCREENER_REFRESH_JOB_ID]
    assert result.ok is False
    assert "borsa erişilemedi" in result.detail
    assert result.error_count == 1


def test_job_run_engine_cycle_treats_untrained_model_as_skip_not_error(monkeypatch):
    def _raise_not_trained():
        raise ModelNotTrained("Model henüz eğitilmedi.")

    monkeypatch.setattr(jobs, "run_cycle_once", _raise_not_trained)
    jobs.job_run_engine_cycle()

    result = status.get_all()[jobs.ENGINE_CYCLE_JOB_ID]
    assert result.ok is True  # model eksikliği hata değil, normal bir "henüz değil" durumu
    assert "atlandı" in result.detail
    assert result.error_count == 0


def test_job_run_engine_cycle_records_actions_summary(monkeypatch):
    class FakeAction:
        def __init__(self, symbol, type_):
            self.symbol = symbol
            self.type = type_

    monkeypatch.setattr(jobs, "run_cycle_once", lambda: [FakeAction("BTC/USDT", "open_long")])
    jobs.job_run_engine_cycle()

    result = status.get_all()[jobs.ENGINE_CYCLE_JOB_ID]
    assert result.ok is True
    assert "BTC/USDT:open_long" in result.detail


def test_job_run_engine_cycle_records_unexpected_failure(monkeypatch):
    def _boom():
        raise RuntimeError("beklenmedik hata")

    monkeypatch.setattr(jobs, "run_cycle_once", _boom)
    jobs.job_run_engine_cycle()

    result = status.get_all()[jobs.ENGINE_CYCLE_JOB_ID]
    assert result.ok is False
    assert result.error_count == 1


def test_job_auto_retrain_skips_when_no_symbols(monkeypatch):
    monkeypatch.setattr(jobs, "get_exchange", lambda *_a, **_k: object())
    monkeypatch.setattr(jobs, "refresh_screener", lambda: [])
    jobs.job_auto_retrain()

    result = status.get_all()[jobs.AUTO_RETRAIN_JOB_ID]
    assert result.ok is True
    assert "atlandı" in result.detail


def test_job_auto_retrain_trains_primary_and_records_success(monkeypatch):
    class _FakeResult:
        rows_used = 1000

        class out_of_sample:
            balanced_accuracy = 0.42

    monkeypatch.setattr(jobs, "get_exchange", lambda *_a, **_k: object())
    monkeypatch.setattr(jobs, "refresh_screener", lambda: [object()])
    monkeypatch.setattr(jobs, "top_long", lambda results, n: [type("R", (), {"symbol": "BTC/USDT:USDT"})()])
    monkeypatch.setattr(jobs, "top_short", lambda results, n: [])
    monkeypatch.setattr(jobs, "train_signal_model_validated", lambda *a, **k: _FakeResult())
    monkeypatch.setattr(jobs, "DEFAULT_META_MODEL_PATH", type("P", (), {"exists": staticmethod(lambda: False)})())

    jobs.job_auto_retrain()

    result = status.get_all()[jobs.AUTO_RETRAIN_JOB_ID]
    assert result.ok is True
    assert "1000" in result.detail
    assert "meta-label" not in result.detail  # meta model daha önce eğitilmemiş, otomatik başlatılmamalı


def test_job_auto_retrain_also_retrains_meta_when_it_already_exists(monkeypatch):
    class _FakeResult:
        rows_used = 1000

        class out_of_sample:
            balanced_accuracy = 0.42

    monkeypatch.setattr(jobs, "get_exchange", lambda *_a, **_k: object())
    monkeypatch.setattr(jobs, "refresh_screener", lambda: [object()])
    monkeypatch.setattr(jobs, "top_long", lambda results, n: [type("R", (), {"symbol": "BTC/USDT:USDT"})()])
    monkeypatch.setattr(jobs, "top_short", lambda results, n: [])
    monkeypatch.setattr(jobs, "train_signal_model_validated", lambda *a, **k: _FakeResult())
    monkeypatch.setattr(jobs, "DEFAULT_META_MODEL_PATH", type("P", (), {"exists": staticmethod(lambda: True)})())
    monkeypatch.setattr(jobs.SignalModel, "load_from", classmethod(lambda cls, path=None: object()))
    monkeypatch.setattr(jobs, "train_meta_label_model", lambda *a, **k: (object(), 500))

    jobs.job_auto_retrain()

    result = status.get_all()[jobs.AUTO_RETRAIN_JOB_ID]
    assert result.ok is True
    assert "meta-label: 500" in result.detail


def test_job_auto_retrain_records_failure(monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("borsa erişilemedi")

    monkeypatch.setattr(jobs, "get_exchange", _boom)
    jobs.job_auto_retrain()

    result = status.get_all()[jobs.AUTO_RETRAIN_JOB_ID]
    assert result.ok is False
    assert result.error_count == 1


def test_compute_auto_retrain_interval_seconds_derived_from_lookback(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "ml_auto_retrain_seconds", None)
    monkeypatch.setattr(settings, "ml_train_timeframe", "1h")
    monkeypatch.setattr(settings, "ml_train_lookback", 2000)
    monkeypatch.setattr(settings, "ml_auto_retrain_refresh_fraction", 0.05)
    monkeypatch.setattr(settings, "ml_auto_retrain_max_seconds", 604800)

    # 2000 saat * 3600sn * 0.05 = 360.000sn (tavanın altında, ham hesap geçerli)
    assert jobs.compute_auto_retrain_interval_seconds() == 360_000


def test_compute_auto_retrain_interval_seconds_capped_at_max(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "ml_auto_retrain_seconds", None)
    monkeypatch.setattr(settings, "ml_train_timeframe", "1h")
    monkeypatch.setattr(settings, "ml_train_lookback", 10000)
    monkeypatch.setattr(settings, "ml_auto_retrain_refresh_fraction", 0.05)
    monkeypatch.setattr(settings, "ml_auto_retrain_max_seconds", 604800)

    # Varsayılanlarla ham hesap 1.800.000sn (~20.8 gün) -> 604800sn (7 gün) tavanına çarpar
    assert jobs.compute_auto_retrain_interval_seconds() == 604800


def test_compute_auto_retrain_interval_seconds_respects_explicit_override(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "ml_auto_retrain_seconds", 3600)
    assert jobs.compute_auto_retrain_interval_seconds() == 3600


def test_compute_auto_retrain_interval_seconds_has_a_floor(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "ml_auto_retrain_seconds", None)
    monkeypatch.setattr(settings, "ml_train_timeframe", "1h")
    monkeypatch.setattr(settings, "ml_train_lookback", 10)
    monkeypatch.setattr(settings, "ml_auto_retrain_refresh_fraction", 0.05)
    monkeypatch.setattr(settings, "ml_auto_retrain_max_seconds", 604800)

    assert jobs.compute_auto_retrain_interval_seconds() == 3600  # taban değer


def test_job_auto_retrain_lstm_skips_when_unused(monkeypatch):
    monkeypatch.setattr(jobs, "DEFAULT_LSTM_MODEL_PATH", type("P", (), {"exists": staticmethod(lambda: False)})())

    jobs.job_auto_retrain_lstm()

    result = status.get_all()[jobs.AUTO_RETRAIN_LSTM_JOB_ID]
    assert result.ok is True
    assert "atlandı" in result.detail


def test_job_auto_retrain_lstm_trains_when_already_used(monkeypatch):
    class _FakeResult:
        rows_used = 500

        class out_of_sample:
            balanced_accuracy = 0.4

    monkeypatch.setattr(jobs, "DEFAULT_LSTM_MODEL_PATH", type("P", (), {"exists": staticmethod(lambda: True)})())
    monkeypatch.setattr(jobs, "get_exchange", lambda *_a, **_k: object())
    monkeypatch.setattr(jobs, "refresh_screener", lambda: [object()])
    monkeypatch.setattr(jobs, "top_long", lambda results, n: [type("R", (), {"symbol": "BTC/USDT:USDT"})()])
    monkeypatch.setattr(jobs, "top_short", lambda results, n: [])
    monkeypatch.setattr(jobs, "train_lstm_signal_model", lambda *a, **k: _FakeResult())

    jobs.job_auto_retrain_lstm()

    result = status.get_all()[jobs.AUTO_RETRAIN_LSTM_JOB_ID]
    assert result.ok is True
    assert "500" in result.detail


def test_job_auto_retrain_online_skips_when_unused(monkeypatch):
    monkeypatch.setattr(jobs, "DEFAULT_ONLINE_MODEL_PATH", type("P", (), {"exists": staticmethod(lambda: False)})())

    jobs.job_auto_retrain_online()

    result = status.get_all()[jobs.AUTO_RETRAIN_ONLINE_JOB_ID]
    assert result.ok is True
    assert "atlandı" in result.detail


def test_job_auto_retrain_regime_skips_when_never_trained(monkeypatch):
    monkeypatch.setattr(jobs, "DEFAULT_REGIME_MODEL_PATH", type("P", (), {"exists": staticmethod(lambda: False)})())

    jobs.job_auto_retrain_regime()

    result = status.get_all()[jobs.AUTO_RETRAIN_REGIME_JOB_ID]
    assert result.ok is True
    assert "atlandı" in result.detail


def test_start_scheduler_disabled_returns_none():
    scheduler = start_scheduler(enabled=False)
    assert scheduler is None
    assert get_scheduler() is None


def test_start_scheduler_enabled_registers_both_jobs():
    scheduler = start_scheduler(enabled=True)
    assert scheduler is not None
    assert scheduler.get_job(jobs.SCREENER_REFRESH_JOB_ID) is not None
    assert scheduler.get_job(jobs.ENGINE_CYCLE_JOB_ID) is not None
    assert get_scheduler() is scheduler


def test_start_scheduler_can_disable_auto_retrain(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "ml_auto_retrain_enabled", False)
    scheduler = start_scheduler(enabled=True)
    assert scheduler.get_job(jobs.AUTO_RETRAIN_JOB_ID) is None


def test_start_scheduler_registers_all_four_retrain_jobs_by_default():
    # ml_auto_retrain_enabled artık varsayılan AÇIK (kullanıcı isteği: tüm
    # modeller otomatik yenilensin) — XGBoost + LSTM + online + regime job'ları
    # hepsi kaydedilmeli (kendi içlerinde dosya-var-mı kontrolüyle gereksiz
    # eğitimi atlıyorlar, ama job'un KENDİSİ her zaman kayıtlıdır).
    scheduler = start_scheduler(enabled=True)
    assert scheduler.get_job(jobs.AUTO_RETRAIN_JOB_ID) is not None
    assert scheduler.get_job(jobs.AUTO_RETRAIN_LSTM_JOB_ID) is not None
    assert scheduler.get_job(jobs.AUTO_RETRAIN_ONLINE_JOB_ID) is not None
    assert scheduler.get_job(jobs.AUTO_RETRAIN_REGIME_JOB_ID) is not None


def test_start_scheduler_is_idempotent():
    first = start_scheduler(enabled=True)
    second = start_scheduler(enabled=True)
    assert first is second


def test_stop_scheduler_clears_instance():
    start_scheduler(enabled=True)
    assert get_scheduler() is not None
    stop_scheduler()
    assert get_scheduler() is None
