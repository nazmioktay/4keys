"""Nihai test penceresi kilidi (KURALLAR.md §1) — ağır bağımlılık yok, her katmandan içe aktarılabilir."""

from __future__ import annotations

import pandas as pd

from .config import EXPERIMENT_LOG, FINAL_TEST_MARKER, FINAL_TEST_START


class FinalTestError(RuntimeError):
    """Nihai test penceresine (>= 2025-10-01) izinsiz erişim."""


def _naive(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_convert(None) if ts.tzinfo is not None else ts


def final_test_is_open(experiment_id: str | None, log_path=EXPERIMENT_LOG) -> bool:
    """`deneyler.md` içinde `FINAL-TEST-ACILDI <id>` satırı var mı?"""
    if not experiment_id:
        return False
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError:
        return False
    needle = f"{FINAL_TEST_MARKER} {experiment_id}"
    return any(line.strip().startswith(needle) for line in text.splitlines())


def ensure_final_test_open(allow_final_test: bool, experiment_id: str | None, log_path=EXPERIMENT_LOG) -> bool:
    """True: pencere açık (açık izin + kayıt var). False: kapalı (kes). Kayıtsız açma denemesi -> FinalTestError."""
    if not allow_final_test:
        return False
    if not final_test_is_open(experiment_id, log_path):
        raise FinalTestError(
            "Nihai test penceresini açmak için docs/research/deneyler.md içinde "
            f"'{FINAL_TEST_MARKER} <deney_id>' kaydı ve final_test_experiment_id gerekir."
        )
    return True


def cut_final_test(obj, allow_final_test: bool = False, experiment_id: str | None = None, log_path=EXPERIMENT_LOG):
    """DatetimeIndex'li DataFrame/Series'i varsayılan olarak `FINAL_TEST_START` öncesine keser."""
    if ensure_final_test_open(allow_final_test, experiment_id, log_path):
        return obj
    idx = obj.index
    idx_naive = idx.tz_convert(None) if getattr(idx, "tz", None) is not None else idx
    return obj[idx_naive < FINAL_TEST_START]


def assert_no_final_test(index, allow_final_test: bool = False, experiment_id: str | None = None, log_path=EXPERIMENT_LOG) -> None:
    """Savunma derinliği: motor/rapor girişinde nihai pencere tarihi var mı?"""
    if ensure_final_test_open(allow_final_test, experiment_id, log_path):
        return
    idx = pd.DatetimeIndex(index)
    if len(idx) and _naive(idx.max()) >= FINAL_TEST_START:
        raise FinalTestError(
            f"Veri nihai test penceresine ({FINAL_TEST_START.date()} ve sonrası) uzanıyor: son tarih {idx.max()}."
        )
