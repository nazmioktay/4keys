from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class JobRunStatus:
    last_run_at: str | None = None
    ok: bool | None = None
    detail: str = ""
    run_count: int = 0
    error_count: int = 0


_status: dict[str, JobRunStatus] = {}


def record(job_id: str, ok: bool, detail: str = "") -> None:
    current = _status.setdefault(job_id, JobRunStatus())
    current.last_run_at = datetime.now(timezone.utc).isoformat()
    current.ok = ok
    current.detail = detail
    current.run_count += 1
    if not ok:
        current.error_count += 1


def get_all() -> dict[str, JobRunStatus]:
    return dict(_status)


def reset() -> None:
    _status.clear()
