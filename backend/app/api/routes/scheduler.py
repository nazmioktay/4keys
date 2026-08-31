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


_JOB_INTERVALS = {
    SCREENER_REFRESH_JOB_ID: lambda: settings.screener_refresh_seconds,
    ENGINE_CYCLE_JOB_ID: lambda: settings.engine_cycle_seconds,
}
_JOB_RUNNERS = {
    SCREENER_REFRESH_JOB_ID: job_refresh_screener,
    ENGINE_CYCLE_JOB_ID: job_run_engine_cycle,
}


@router.get("/status", response_model=SchedulerStatusResponse)
def scheduler_status() -> SchedulerStatusResponse:
    scheduler = get_scheduler()
    run_status = status.get_all()

    jobs: list[JobInfo] = []
    for job_id, interval_fn in _JOB_INTERVALS.items():
        next_run_at = None
        if scheduler is not None:
            job = scheduler.get_job(job_id)
            if job is not None and job.next_run_time is not None:
                next_run_at = job.next_run_time.isoformat()
        job_status = run_status.get(job_id)
        jobs.append(
            JobInfo(
                job_id=job_id,
                interval_seconds=interval_fn(),
                next_run_at=next_run_at,
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
