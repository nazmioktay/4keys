import pandas as pd
import pytest

from app.core.config import settings
from app.db import repository as db
from app.db.session import init_db, reset_for_tests
from app.ml.openinterest_features import latest_open_interest_feature_row, load_open_interest_history, merge_open_interest_features
from app.openinterest.service import fetch_and_record_open_interest_snapshot, refresh_all_configured_symbols


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "sqlite:///:memory:")
    reset_for_tests()
    init_db()
    yield
    reset_for_tests()


class _FakeExchange:
    def __init__(self, metrics=None):
        self._metrics = metrics

    def fetch_open_interest(self, symbol):
        return self._metrics


def test_fetch_and_record_open_interest_snapshot_persists_to_db():
    metrics = {"open_interest": 15000.0, "open_interest_value": 950_000_000.0}
    result = fetch_and_record_open_interest_snapshot(_FakeExchange(metrics), "BTC/USDT:USDT")
    assert result == metrics

    latest = db.get_latest_open_interest_snapshot("BTC/USDT:USDT")
    assert latest is not None
    assert latest["open_interest"] == pytest.approx(15000.0)
    assert latest["open_interest_value"] == pytest.approx(950_000_000.0)


def test_fetch_and_record_open_interest_snapshot_returns_none_when_exchange_fails():
    result = fetch_and_record_open_interest_snapshot(_FakeExchange(None), "BTC/USDT:USDT")
    assert result is None
    assert db.get_latest_open_interest_snapshot("BTC/USDT:USDT") is None


def test_refresh_all_configured_symbols_handles_per_symbol_failure_independently():
    class _MixedExchange:
        def fetch_open_interest(self, symbol):
            if symbol == "BTC/USDT:USDT":
                return {"open_interest": 100.0, "open_interest_value": 6_000_000.0}
            raise RuntimeError("boom")

    results = {}
    try:
        results = refresh_all_configured_symbols(_MixedExchange(), ["BTC/USDT:USDT"])
    except RuntimeError:
        pytest.fail("bir sembolün hatası diğerlerini etkilememeli")
    assert results["BTC/USDT:USDT"] is not None


def test_merge_open_interest_features_backward_fills_without_lookahead():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="1h").astype("datetime64[ns]").astype("int64"),
            "close": [100.0, 102.0, 101.0, 105.0],
        }
    )
    history = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01T00:30:00Z", "2024-01-01T02:30:00Z"]),
            "open_interest": [1000.0, 1100.0],  # ikinci kayıtta OI %10 arttı
        }
    )

    merged = merge_open_interest_features(frame, history)

    # bar 0 (00:00): henüz hiçbir OI kaydı YOK -> NaN
    assert pd.isna(merged["oi_change_pct"].iloc[0])
    # bar 1 (01:00): yalnızca İLK kayıt bilinir (tek nokta -> pct_change NaN)
    assert pd.isna(merged["oi_change_pct"].iloc[1])
    # bar 3 (03:00): 02:30'daki İKİNCİ kayıt artık bilinir -> %10 artış
    assert merged["oi_change_pct"].iloc[3] == pytest.approx(10.0)
    # fiyat da AYNI yönde (100->102->101->105, net yukarı) yükseldi -> pozitif uyum
    assert merged["oi_price_divergence"].iloc[3] == pytest.approx(1.0)


def test_merge_open_interest_features_all_nan_when_history_empty():
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="1h").astype("int64"), "close": [1.0, 2.0, 3.0]})
    merged = merge_open_interest_features(frame, pd.DataFrame())
    assert merged["oi_change_pct"].isna().all()


def test_load_open_interest_history_empty_when_db_disabled(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "")
    reset_for_tests()
    assert load_open_interest_history("BTC/USDT:USDT").empty
    reset_for_tests()


def test_latest_open_interest_feature_row_returns_neutral_when_no_history():
    row = latest_open_interest_feature_row("BTC/USDT:USDT")
    assert row == {"oi_change_pct": 0.0, "oi_price_divergence": 0.0}


def test_latest_open_interest_feature_row_reflects_recent_change():
    db.record_open_interest_snapshot("BTC/USDT:USDT", {"open_interest": 1000.0, "open_interest_value": 60_000_000.0})
    db.record_open_interest_snapshot("BTC/USDT:USDT", {"open_interest": 1050.0, "open_interest_value": 63_000_000.0})
    row = latest_open_interest_feature_row("BTC/USDT:USDT")
    assert row["oi_change_pct"] == pytest.approx(5.0)
