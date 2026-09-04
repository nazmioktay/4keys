from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.core.config import settings
from app.db import repository as db
from app.db.session import init_db, reset_for_tests, session_scope
from app.db.models import OHLCVRaw
from app.ml.features import build_features


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "sqlite:///:memory:")
    reset_for_tests()
    init_db()
    yield
    reset_for_tests()


def test_get_ohlcv_returns_tz_naive_timestamps_even_when_stored_tz_aware():
    """`OHLCVRaw.time` gerçek Postgres'te (`DateTime(timezone=True)`) tz-aware
    bir datetime döner — borsadan gelen zaman damgaları HER ZAMAN tz-naive.
    Bu test, satırı KASITLI OLARAK tz-aware bir datetime ile yazıp
    (Postgres'in davranışını taklit ederek — SQLite bu ayrımı testlerde
    doğal olarak yakalamıyordu, bkz. session notu) `get_ohlcv`'in yine de
    tz-naive döndüğünü doğrular."""
    aware_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    with session_scope() as session:
        session.add(
            OHLCVRaw(
                time=aware_time,
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
            )
        )

    result = db.get_ohlcv("BTC/USDT:USDT", "1h", limit=10)

    assert len(result) == 1
    ts = result["timestamp"].iloc[0]
    assert pd.Timestamp(ts).tzinfo is None
    assert pd.Timestamp(ts) == pd.Timestamp("2024-01-01 12:00:00")


def test_get_ohlcv_output_is_compatible_with_build_features():
    """Gerçek regresyon: tz-aware bir zaman damgası `build_features()`'ın
    `.astype("datetime64[ns]")` dönüşümünü patlatıyordu — bu da
    `_build_symbol_frames`'in HER sembolü sessizce atlamasına (0 satır)
    yol açıyordu (bkz. train-all sonucunda XGBoost'tan SONRA çalışan
    LSTM/online/regime'in hepsinin "yeterli veri yok" vermesi)."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(0)
    with session_scope() as session:
        for i in range(120):
            close = 100 + rng.normal(0, 1)
            session.add(
                OHLCVRaw(
                    time=base + timedelta(hours=i),
                    symbol="BTC/USDT:USDT",
                    timeframe="1h",
                    open=close,
                    high=close + 1,
                    low=close - 1,
                    close=close,
                    volume=100.0,
                )
            )

    ohlcv = db.get_ohlcv("BTC/USDT:USDT", "1h", limit=120)
    assert len(ohlcv) == 120

    # Regresyondan önce bu satır TypeError fırlatırdı.
    features = build_features(ohlcv)
    assert not features.empty
