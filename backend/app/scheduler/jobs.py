import logging

from app.engine.service import ModelNotTrained, run_cycle_once
from app.screener.service import refresh as refresh_screener
from app.security.kill_switch import KillSwitchActive

from . import status

logger = logging.getLogger(__name__)

SCREENER_REFRESH_JOB_ID = "screener_refresh"
ENGINE_CYCLE_JOB_ID = "engine_cycle"


def job_refresh_screener() -> None:
    """Periyodik iş: screener önbelleğini tazeler.

    API isteği gelen kullanıcı taramanın bitmesini beklemesin diye bu iş
    düzenli aralıklarla arka planda çalışır.
    """
    try:
        results = refresh_screener()
        status.record(SCREENER_REFRESH_JOB_ID, ok=True, detail=f"{len(results)} sembol tarandı")
    except Exception as exc:  # noqa: BLE001 - zamanlayıcı thread'i asla çökmemeli
        logger.exception("screener refresh job failed")
        status.record(SCREENER_REFRESH_JOB_ID, ok=False, detail=str(exc))


def job_run_engine_cycle() -> None:
    """Periyodik iş: ML karar motorunun bir döngüsünü çalıştırır.

    Model henüz eğitilmemişse bu normal bir durumdur (kullanıcı henüz
    `/ml/train` çağırmamış olabilir) — hata olarak değil, "atlandı" olarak
    kaydedilir; zamanlayıcı bir sonraki turda tekrar dener.
    """
    try:
        actions = run_cycle_once()
        summary = ", ".join(f"{a.symbol}:{a.type}" for a in actions) or "aksiyon yok"
        status.record(ENGINE_CYCLE_JOB_ID, ok=True, detail=summary)
    except ModelNotTrained as exc:
        status.record(ENGINE_CYCLE_JOB_ID, ok=True, detail=f"atlandı: {exc}")
    except KillSwitchActive as exc:
        status.record(ENGINE_CYCLE_JOB_ID, ok=True, detail=f"atlandı: {exc}")
    except Exception as exc:  # noqa: BLE001 - zamanlayıcı thread'i asla çökmemeli
        logger.exception("engine cycle job failed")
        status.record(ENGINE_CYCLE_JOB_ID, ok=False, detail=str(exc))
