"""Haftalık optimizasyonun canlıya uyguladığı eşiklerin kalıcılığı.

`settings.live_*` değerleri süreç belleğinde tutulur; otomatik optimizasyon
bunları değiştirdiğinde önceden bir restart her şeyi varsayılana döndürüyordu.
Uygulanan değerler `app_state` tablosuna yazılır, açılışta geri yüklenir.
(Kelly ayarları portföy kurallarının parçasıdır, portföy durumuyla birlikte
kalıcıdır — bkz. `app.portfolio.shared`.)
"""

import logging

from app.core.config import settings
from app.db import repository as db

logger = logging.getLogger(__name__)

_STATE_KEY = "live_overrides"
LIVE_OVERRIDE_FIELDS = ("live_open_confidence", "live_close_confidence", "live_meta_label_act_threshold")


def persist_live_overrides() -> None:
    db.save_app_state(_STATE_KEY, {field: getattr(settings, field) for field in LIVE_OVERRIDE_FIELDS})


def apply_persisted_live_overrides() -> dict:
    state = db.load_app_state(_STATE_KEY) or {}
    applied = {}
    for field in LIVE_OVERRIDE_FIELDS:
        if field in state and state[field] is not None:
            setattr(settings, field, float(state[field]))
            applied[field] = float(state[field])
    if applied:
        logger.info("kalıcı canlı ayarlar geri yüklendi: %s", applied)
    return applied
