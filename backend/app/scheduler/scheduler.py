import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

from app.core.config import settings

from .jobs import (
    AUTO_RETRAIN_JOB_ID,
    AUTO_RETRAIN_LSTM_JOB_ID,
    AUTO_RETRAIN_ONLINE_JOB_ID,
    AUTO_RETRAIN_REGIME_JOB_ID,
    ENGINE_CYCLE_JOB_ID,
    MACRO_REFRESH_JOB_ID,
    ORDERBOOK_REFRESH_JOB_ID,
    SCREENER_REFRESH_JOB_ID,
    compute_auto_retrain_interval_seconds,
    job_auto_retrain,
    job_auto_retrain_lstm,
    job_auto_retrain_online,
    job_auto_retrain_regime,
    job_refresh_macro,
    job_refresh_orderbook,
    job_refresh_screener,
    job_run_engine_cycle,
)

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def start_scheduler(enabled: bool | None = None) -> BackgroundScheduler | None:
    """Screener ve ML karar motorunu periyodik olarak çalıştıran arka plan
    zamanlayıcısını başlatır.

    Uygulama başına tek bir zamanlayıcı olur (idempotent — zaten çalışıyorsa
    tekrar başlatmaz). `enabled=False` verilirse (veya
    `FOURKEYS_SCHEDULER_ENABLED=false`) hiçbir iş planlanmaz — testlerde ve
    tek seferlik komut satırı kullanımında arka plan thread'i istenmez.
    """
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    if enabled if enabled is not None else settings.scheduler_enabled:
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            job_refresh_screener,
            "interval",
            seconds=settings.screener_refresh_seconds,
            id=SCREENER_REFRESH_JOB_ID,
            next_run_time=None,
            max_instances=1,
            coalesce=True,
        )
        scheduler.add_job(
            job_run_engine_cycle,
            "interval",
            seconds=settings.engine_cycle_seconds,
            id=ENGINE_CYCLE_JOB_ID,
            max_instances=1,
            coalesce=True,
        )
        scheduler.add_job(
            job_refresh_macro,
            "interval",
            seconds=settings.macro_refresh_seconds,
            id=MACRO_REFRESH_JOB_ID,
            max_instances=1,
            coalesce=True,
        )
        scheduler.add_job(
            job_refresh_orderbook,
            "interval",
            seconds=settings.orderbook_refresh_seconds,
            id=ORDERBOOK_REFRESH_JOB_ID,
            max_instances=1,
            coalesce=True,
        )
        if settings.ml_auto_retrain_enabled:
            # Aynı hesaplanmış aralık (bkz. `compute_auto_retrain_interval_seconds`)
            # 4 model ailesi için de kullanılır. `next_run_time` KASITLI olarak
            # verilmez (APScheduler varsayılanı: ilk çalıştırma bir TAM aralık
            # sonra) — uygulama her açılışta/redeploy'da (bkz. recreate-backend.sh
            # sık restart deseni) ağır eğitimleri hemen tetiklemesin diye (bkz.
            # önceki `job_auto_retrain`'deki aynı gerekçe). Job'lar zaten kendi
            # içlerinde dosya-var-mı/ensemble-bayrağı kontrolüyle gereksiz
            # eğitimleri atlıyor (bkz. app.scheduler.jobs); yine de her biri
            # ayrı bir `id` ile farklı thread zamanlamasına düşer, aynı anda
            # çakışmaları APScheduler'ın kendi thread havuzu yönetir.
            interval_seconds = compute_auto_retrain_interval_seconds()
            for job_id, job_func in (
                (AUTO_RETRAIN_JOB_ID, job_auto_retrain),
                (AUTO_RETRAIN_LSTM_JOB_ID, job_auto_retrain_lstm),
                (AUTO_RETRAIN_ONLINE_JOB_ID, job_auto_retrain_online),
                (AUTO_RETRAIN_REGIME_JOB_ID, job_auto_retrain_regime),
            ):
                scheduler.add_job(
                    job_func,
                    "interval",
                    seconds=interval_seconds,
                    id=job_id,
                    max_instances=1,
                    coalesce=True,
                )
        scheduler.start()
        # İlk taramayı hemen tetikle ki motor döngüsü boş önbekleğe düşmesin.
        scheduler.modify_job(SCREENER_REFRESH_JOB_ID, next_run_time=datetime.now())
        scheduler.modify_job(MACRO_REFRESH_JOB_ID, next_run_time=datetime.now())
        scheduler.modify_job(ORDERBOOK_REFRESH_JOB_ID, next_run_time=datetime.now())
        # auto_retrain'e İLK çalıştırmada hemen tetiklenmez — yüzlerce
        # sembolde ağır bir eğitim, uygulama başlarken ilk isteklerin
        # gecikmesine yol açmasın; yalnızca normal interval'ında çalışır.
        _scheduler = scheduler
        logger.info(
            "scheduler started: screener every %ss, engine cycle every %ss, macro refresh every %ss",
            settings.screener_refresh_seconds,
            settings.engine_cycle_seconds,
            settings.macro_refresh_seconds,
        )
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def get_scheduler() -> BackgroundScheduler | None:
    return _scheduler
