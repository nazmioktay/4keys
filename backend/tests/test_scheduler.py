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


def test_start_scheduler_is_idempotent():
    first = start_scheduler(enabled=True)
    second = start_scheduler(enabled=True)
    assert first is second


def test_stop_scheduler_clears_instance():
    start_scheduler(enabled=True)
    assert get_scheduler() is not None
    stop_scheduler()
    assert get_scheduler() is None
