import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from app.core.config import settings

from . import persistence
from .jobs import (
    AUTO_RETRAIN_JOB_ID,
    AUTO_RETRAIN_LSTM_JOB_ID,
    AUTO_RETRAIN_ONLINE_JOB_ID,
    AUTO_RETRAIN_REGIME_JOB_ID,
    ENGINE_CYCLE_JOB_ID,
    MACRO_REFRESH_JOB_ID,
    OPEN_INTEREST_REFRESH_JOB_ID,
    ORDERBOOK_REFRESH_JOB_ID,
    PERIODIC_OPTIMIZATION_JOB_ID,
    SCREENER_REFRESH_JOB_ID,
    compute_auto_retrain_interval_seconds,
    job_auto_retrain,
    job_auto_retrain_lstm,
    job_auto_retrain_online,
    job_auto_retrain_regime,
    job_periodic_optimization,
    job_refresh_macro,
    job_refresh_open_interest,
    job_refresh_orderbook,
    job_refresh_screener,
    job_run_engine_cycle,
)

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _persisted(job_id: str, job_func):
    """`job_func`u sarar, çalışmayı (başarılı/başarısız fark etmeksizin,
    tamamlandığı sürece) `persistence.record_run` ile kalıcı diske
    yazar — bkz. `persistence.py` modül docstring'i: bu, süreç yeniden
    başlasa da "son ne zaman denendi" bilgisinin kaybolmamasını sağlar."""

    def wrapper() -> None:
        try:
            job_func()
        finally:
            persistence.record_run(job_id)

    wrapper.__name__ = getattr(job_func, "__name__", job_id)
    return wrapper


def _catch_up_time(job_id: str, interval_seconds: int, stagger_minutes: int) -> datetime | None:
    """Son (denenen) çalışmadan bu yana `interval_seconds`i AŞAN bir süre
    geçtiyse (ör. bilgisayar günlerce/haftalarca kapalı kaldıysa), job'u
    normal aralığı beklemek yerine açılıştan `stagger_minutes` sonra
    tetikler — birden fazla ağır job'un (auto_retrain + LSTM + online +
    regime + periodic_optimization) TAM AÇILIŞ ANINDA aynı anda
    çakışmaması için her biri farklı `stagger_minutes` alır. Hiç
    çalışmamışsa (ilk kurulum) `None` döner — APScheduler'ın kendi
    varsayılanı (bir tam interval sonra) korunur, ilk açılışta ağır
    eğitim tetiklenmez."""
    last_run = persistence.read_last_run(job_id)
    if last_run is None:
        return None
    elapsed = (datetime.now(timezone.utc) - last_run).total_seconds()
    if elapsed < interval_seconds:
        return None
    return datetime.now(timezone.utc) + timedelta(minutes=stagger_minutes)


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
        scheduler.add_job(
            job_refresh_open_interest,
            "interval",
            seconds=settings.open_interest_refresh_seconds,
            id=OPEN_INTEREST_REFRESH_JOB_ID,
            max_instances=1,
            coalesce=True,
        )
        _catch_up_jobs: list[tuple[str, datetime]] = []
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
            #
            # YAKALAMA (catch-up): yukarıdaki "hemen tetiklenmesin" kararı,
            # SÜREKLİ AÇIK bir sunucuda doğrudur — ama bu proje sık kapanıp
            # açılan bir makinede de geliştiriliyor, burada HER yeniden
            # başlatma sayacı süreç başlangıcından sıfırlardı (7 günlük
            # aralık, süreç 7 günden sık yeniden başladığı sürece HİÇ
            # dolmazdı, iş asla çalışmazdı). `_catch_up_time`, kalıcı diskte
            # tutulan "son deneme" zamanına bakıp süresi gerçekten dolmuşsa
            # işi açılıştan birkaç dakika sonraya (job'lar arası kademeli
            # gecikmeyle, hepsi aynı anda çakışmasın diye) planlar.
            interval_seconds = compute_auto_retrain_interval_seconds()
            for stagger_minutes, (job_id, job_func) in enumerate(
                (
                    (AUTO_RETRAIN_JOB_ID, job_auto_retrain),
                    (AUTO_RETRAIN_LSTM_JOB_ID, job_auto_retrain_lstm),
                    (AUTO_RETRAIN_ONLINE_JOB_ID, job_auto_retrain_online),
                    (AUTO_RETRAIN_REGIME_JOB_ID, job_auto_retrain_regime),
                ),
                start=2,
            ):
                scheduler.add_job(
                    _persisted(job_id, job_func),
                    "interval",
                    seconds=interval_seconds,
                    id=job_id,
                    max_instances=1,
                    coalesce=True,
                )
                catch_up = _catch_up_time(job_id, interval_seconds, stagger_minutes * 5)
                if catch_up is not None:
                    _catch_up_jobs.append((job_id, catch_up))
        if settings.ml_periodic_optimization_enabled:
            # Haftalık walk-forward parametre optimizasyonu (bkz. README
            # "karlılık", app.scheduler.jobs.job_periodic_optimization
            # docstring'i) — CANLI ayarları DEĞİŞTİRMEZ, yalnızca öneriyi
            # kaydeder. `next_run_time` KASITLI verilmez — auto_retrain
            # ile AYNI gerekçe, uygulama her açılışta/redeploy'da hemen
            # tetiklenmesin. Aynı yakalama (catch-up) mantığı burada da
            # uygulanır (bkz. yukarıdaki uzun açıklama).
            scheduler.add_job(
                _persisted(PERIODIC_OPTIMIZATION_JOB_ID, job_periodic_optimization),
                "interval",
                seconds=settings.ml_periodic_optimization_seconds,
                id=PERIODIC_OPTIMIZATION_JOB_ID,
                max_instances=1,
                coalesce=True,
            )
            catch_up = _catch_up_time(PERIODIC_OPTIMIZATION_JOB_ID, settings.ml_periodic_optimization_seconds, 30)
            if catch_up is not None:
                _catch_up_jobs.append((PERIODIC_OPTIMIZATION_JOB_ID, catch_up))
        scheduler.start()
        # İlk taramayı hemen tetikle ki motor döngüsü boş önbekleğe düşmesin.
        scheduler.modify_job(SCREENER_REFRESH_JOB_ID, next_run_time=datetime.now())
        scheduler.modify_job(MACRO_REFRESH_JOB_ID, next_run_time=datetime.now())
        scheduler.modify_job(ORDERBOOK_REFRESH_JOB_ID, next_run_time=datetime.now())
        scheduler.modify_job(OPEN_INTEREST_REFRESH_JOB_ID, next_run_time=datetime.now())
        # auto_retrain'e İLK çalıştırmada hemen tetiklenmez — yüzlerce
        # sembolde ağır bir eğitim, uygulama başlarken ilk isteklerin
        # gecikmesine yol açmasın; yalnızca normal interval'ında çalışır.
        # İSTİSNA: yukarıda toplanan `_catch_up_jobs` — süresi gerçekten
        # dolmuş (bilgisayar uzun süre kapalıydı) job'lar için kademeli
        # gecikmeli "yakalama" zamanı burada, diğer `modify_job`larla
        # AYNI (scheduler.start() SONRASI) noktada uygulanır.
        for job_id, catch_up in _catch_up_jobs:
            logger.warning("scheduler: %s süresi dolmuş (muhtemelen süreç uzun süre kapalıydı) — %s itibariyle yakalanacak", job_id, catch_up)
            scheduler.modify_job(job_id, next_run_time=catch_up)
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
