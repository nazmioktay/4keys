from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.scheduler import status
from app.scheduler.jobs import ENGINE_CYCLE_JOB_ID, SCREENER_REFRESH_JOB_ID, job_refresh_screener, job_run_engine_cycle
from app.scheduler.scheduler import get_scheduler

router = APIRouter(prefix="/scheduler", tags=["scheduler"])


class JobInfo(BaseModel):
    job_id: str
    interval_seconds: int
    next_run_at: str | None
    last_run_at: str | None
    ok: bool | None
    detail: str
    run_count: int
    error_count: int


class SchedulerStatusResponse(BaseModel):
    enabled: bool
    running: bool
    jobs: list[JobInfo]


_JOB_RUNNERS = {
    SCREENER_REFRESH_JOB_ID: job_refresh_screener,
    ENGINE_CYCLE_JOB_ID: job_run_engine_cycle,
}


@router.get("/status", response_model=SchedulerStatusResponse)
def scheduler_status() -> SchedulerStatusResponse:
    """Zamanlayıcıda GERÇEKTEN kayıtlı olan TÜM işleri döner (önceden yalnızca
    screener_refresh/engine_cycle'ı gösteren sabit bir listeydi — auto_retrain
    (x4) ve periodic_optimization job'ları `ml_auto_retrain_enabled`/
    `ml_periodic_optimization_enabled` açıkken zamanlayıcıya EKLENİYORDU ama
    bu uç nokta onları hiç RAPORLAMIYORDU, "haftalık otomatik eğitim çalışıyor
    mu?" gibi bir soruya API üzerinden yanıt verilemiyordu). Artık
    `scheduler.get_jobs()`in döndürdüğü GERÇEK listeye dayanır."""
    scheduler = get_scheduler()
    run_status = status.get_all()

    jobs: list[JobInfo] = []
    if scheduler is not None:
        for job in scheduler.get_jobs():
            job_status = run_status.get(job.id)
            interval = getattr(job.trigger, "interval", None)
            jobs.append(
                JobInfo(
                    job_id=job.id,
                    interval_seconds=int(interval.total_seconds()) if interval is not None else 0,
                    next_run_at=job.next_run_time.isoformat() if job.next_run_time else None,
                    last_run_at=job_status.last_run_at if job_status else None,
                    ok=job_status.ok if job_status else None,
                    detail=job_status.detail if job_status else "",
                    run_count=job_status.run_count if job_status else 0,
                    error_count=job_status.error_count if job_status else 0,
                )
            )

    return SchedulerStatusResponse(enabled=settings.scheduler_enabled, running=scheduler is not None, jobs=jobs)


@router.post("/trigger/{job_id}")
def trigger_job(job_id: Literal["screener_refresh", "engine_cycle"]) -> dict:
    """Bir zamanlanmış işi beklemeden hemen, senkron olarak çalıştırır."""
    runner = _JOB_RUNNERS.get(job_id)
    if runner is None:
        raise HTTPException(status_code=404, detail=f"Bilinmeyen job: {job_id}")
    runner()
    job_status = status.get_all().get(job_id)
    return {
        "job_id": job_id,
        "ok": job_status.ok if job_status else None,
        "detail": job_status.detail if job_status else "",
    }
