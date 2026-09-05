import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _status_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".status.json")


def write_model_status(
    model_path: Path,
    enabled: bool,
    balanced_accuracy: float,
    reason: str | None = None,
    holdout_start_time: str | None = None,
) -> None:
    """Bir modelin (LSTM/online) canlı karar motorunda KULLANILIP
    KULLANILMAYACAĞINI, eğitimin SONUNDA otomatik olarak belirler ve
    kalıcı olarak yazar — önceden bu, kullanıcının elle açması gereken
    statik bir ayar bayrağıydı (`FOURKEYS_ENSEMBLE_LSTM_ENABLED` vb.).

    `enabled=False` (kalite eşiğinin altında kalan bir eğitim) yazıldığında,
    modelin ESKİ (daha önce kaydedilmiş, iyi) dosyası diskte kalsa bile
    canlı karar motoru onu KULLANMAZ — "düşük puanlı modelin eski verisini
    kullanma" kuralı: bir model yalnızca EN SON eğitiminde eşiği geçtiği
    sürece aktif kalır, geçemediği an (yeni bir eğitim eşiği geçene kadar)
    devre dışı kalır.

    `holdout_start_time` (ISO zaman damgası, opsiyonel): bu modelin son
    `fit()` çağrısında kronolojik holdout diliminin BAŞLANGICI — yani
    modelin HİÇ GÖRMEDİĞİ verinin başladığı an (bkz. `split_out_of_sample`).
    `app.backtest.system_runner` bunu okuyup backtest'i varsayılan olarak
    yalnızca bu tarihten SONRAKİ barlarla sınırlar (aksi halde backtest,
    modelin kendi eğitim verisiyle çakışan, "ezber" içeren bir pencereyi
    test eder). `None` verilirse (ör. model REDDEDİLDİ, disk'teki dosya
    DEĞİŞMEDİ) daha önce kaydedilmiş değer OLDUĞU GİBİ korunur — reddedilen
    bir denemenin holdout'u, diskte hâlâ duran ESKİ modelin holdout'unun
    üzerine yanlışlıkla yazılmaz."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    previous = read_model_status(model_path) or {}
    status = {
        "enabled": enabled,
        "balanced_accuracy": balanced_accuracy,
        "reason": reason,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "holdout_start_time": holdout_start_time if holdout_start_time is not None else previous.get("holdout_start_time"),
    }
    try:
        _status_path(model_path).write_text(json.dumps(status))
    except OSError:
        logger.exception("model status yazılamadı: %s", model_path)


def is_model_enabled(model_path: Path) -> bool:
    """Bir modelin canlı karar motorunda kullanılıp kullanılmayacağını
    döner — dosya YOKSA veya status dosyası YOKSA/okunamazsa (ör. hiç
    eğitilmemiş) `False`. `write_model_status`'un tersidir."""
    if not model_path.exists():
        return False
    status_file = _status_path(model_path)
    if not status_file.exists():
        return False
    try:
        status = json.loads(status_file.read_text())
    except (OSError, json.JSONDecodeError):
        logger.exception("model status okunamadı: %s", model_path)
        return False
    return bool(status.get("enabled", False))


def get_holdout_start_time(model_path: Path) -> str | None:
    """Modelin en son kabul edilen eğitiminde holdout'un başladığı ISO
    zaman damgasını döner (bkz. `write_model_status`) — kayıt yoksa
    (ör. bu alan henüz hiç yazılmamış eski bir model) `None`."""
    status = read_model_status(model_path)
    if not status:
        return None
    return status.get("holdout_start_time")


def read_model_status(model_path: Path) -> dict | None:
    status_file = _status_path(model_path)
    if not status_file.exists():
        return None
    try:
        return json.loads(status_file.read_text())
    except (OSError, json.JSONDecodeError):
        return None
