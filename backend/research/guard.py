"""Kilitli pencere koruması (KURALLAR.md §1 ve §10) — ağır bağımlılık yok, her katmandan içe aktarılabilir.

İki bölge vardır; her deney tam olarak birinde koşar (`zone(...)` bağlamı, varsayılan "main"):
- **main** (ana araştırma): veri < FINAL_TEST_START. Nihai test açılınca (Aşama 5) [.., FORWARD_EXPLORE_START).
- **forward** (yalnızca forward_only kaynaklı deneyler): veri [FORWARD_EXPLORE_START, FORWARD_TEST_START). Forward test
  açılınca (`FORWARD-TEST-ACILDI <id>` + en erken FORWARD_TEST_MIN_OPEN) [FORWARD_EXPLORE_START, ..).
Hiçbir bölge diğerinin kilitli penceresini göremez: nihai pencere [FINAL_TEST_START, FORWARD_EXPLORE_START) forward
bölgesinden de görünmez. Yükleyiciler, motor, rapor ve kaynaklar ETKİN bölgeyi buradan okur."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .config import (
    EXPERIMENT_LOG,
    FINAL_TEST_MARKER,
    FINAL_TEST_START,
    FORWARD_EXPLORE_START,
    FORWARD_TEST_MARKER,
    FORWARD_TEST_MIN_OPEN,
    FORWARD_TEST_START,
    ZONES,
)

_ZONE: contextvars.ContextVar[str] = contextvars.ContextVar("research_zone", default="main")


class FinalTestError(RuntimeError):
    """Kilitli pencereye (nihai test veya forward test) izinsiz erişim."""


@contextmanager
def zone(name: str):
    """Bu blok içindeki tüm yükleme/motor/rapor çağrıları `name` bölgesinin sınırlarına uyar."""
    if name not in ZONES:
        raise ValueError(f"bilinmeyen bölge '{name}' (geçerli: {', '.join(ZONES)})")
    token = _ZONE.set(name)
    try:
        yield name
    finally:
        _ZONE.reset(token)


def current_zone() -> str:
    return _ZONE.get()


def _naive(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_convert(None) if ts.tzinfo is not None else ts


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()


def _marker_present(marker: str, experiment_id: str | None, log_path) -> bool:
    if not experiment_id:
        return False
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError:
        return False
    for line in text.splitlines():
        tokens = line.strip().split()
        if len(tokens) >= 2 and tokens[0] == marker and tokens[1] == experiment_id:  # tam eşleşme ('e1' 'e10'u AÇMAZ)
            return True
    return False


def final_test_is_open(experiment_id: str | None, log_path=EXPERIMENT_LOG) -> bool:
    """`deneyler.md` içinde `FINAL-TEST-ACILDI <id>` satırı var mı?"""
    return _marker_present(FINAL_TEST_MARKER, experiment_id, log_path)


def forward_test_is_open(experiment_id: str | None, log_path=EXPERIMENT_LOG) -> bool:
    """`deneyler.md` içinde `FORWARD-TEST-ACILDI <id>` satırı var mı?"""
    return _marker_present(FORWARD_TEST_MARKER, experiment_id, log_path)


def ensure_final_test_open(allow_final_test: bool, experiment_id: str | None, log_path=EXPERIMENT_LOG) -> bool:
    """ETKİN bölgenin kilitli test penceresi açık mı? True: açık (açık izin + kayıt var). False: kapalı (kes).
    Kayıtsız (veya forward testte erken) açma denemesi -> FinalTestError."""
    if not allow_final_test:
        return False
    if current_zone() == "forward":
        if not forward_test_is_open(experiment_id, log_path):
            raise FinalTestError(
                "Forward test penceresini açmak için docs/research/deneyler.md içinde "
                f"'{FORWARD_TEST_MARKER} <deney_id>' kaydı ve final_test_experiment_id gerekir."
            )
        if _today() < FORWARD_TEST_MIN_OPEN:
            raise FinalTestError(
                f"Forward test en erken {FORWARD_TEST_MIN_OPEN.date()} tarihinde açılabilir (>= 6 ay test verisi; KURALLAR.md §10)."
            )
        return True
    if not final_test_is_open(experiment_id, log_path):
        raise FinalTestError(
            "Nihai test penceresini açmak için docs/research/deneyler.md içinde "
            f"'{FINAL_TEST_MARKER} <deney_id>' kaydı ve final_test_experiment_id gerekir."
        )
    return True


def zone_bounds(allow_final_test: bool = False, experiment_id: str | None = None, log_path=EXPERIMENT_LOG):
    """ETKİN bölgede izin verilen [alt, üst) tarih aralığı; None = sınırsız."""
    opened = ensure_final_test_open(allow_final_test, experiment_id, log_path)
    if current_zone() == "forward":
        return FORWARD_EXPLORE_START, (None if opened else FORWARD_TEST_START)
    return None, (FORWARD_EXPLORE_START if opened else FINAL_TEST_START)


def _describe(lo, hi) -> str:
    return f"[{'-∞' if lo is None else lo.date()}, {'+∞' if hi is None else hi.date()})"


def cut_final_test(obj, allow_final_test: bool = False, experiment_id: str | None = None, log_path=EXPERIMENT_LOG):
    """DatetimeIndex'li DataFrame/Series'i ETKİN bölgenin izinli aralığına keser (varsayılan main: < FINAL_TEST_START)."""
    lo, hi = zone_bounds(allow_final_test, experiment_id, log_path)
    idx = obj.index
    idx_naive = idx.tz_convert(None) if getattr(idx, "tz", None) is not None else idx
    mask = np.ones(len(idx_naive), dtype=bool)
    if lo is not None:
        mask &= idx_naive >= lo
    if hi is not None:
        mask &= idx_naive < hi
    return obj[mask]


def assert_no_final_test(index, allow_final_test: bool = False, experiment_id: str | None = None, log_path=EXPERIMENT_LOG) -> None:
    """Savunma derinliği: motor/rapor girişindeki tarihler ETKİN bölgenin izinli aralığında mı?"""
    lo, hi = zone_bounds(allow_final_test, experiment_id, log_path)
    idx = pd.DatetimeIndex(index)
    if not len(idx):
        return
    first, last = _naive(idx.min()), _naive(idx.max())
    if hi is not None and last >= hi:
        window = "forward test penceresine" if current_zone() == "forward" else "nihai test penceresine"
        raise FinalTestError(
            f"Veri kilitli pencereye ({window}) uzanıyor ({current_zone()} bölgesi izinli aralık {_describe(lo, hi)}): son tarih {idx.max()}."
        )
    if lo is not None and first < lo:
        raise FinalTestError(
            f"Veri {current_zone()} bölgesinin izinli aralığından önce başlıyor ({_describe(lo, hi)}): ilk tarih {idx.min()}."
        )
