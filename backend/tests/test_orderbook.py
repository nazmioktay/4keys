import pandas as pd
import pytest

from app.core.config import settings
from app.db import repository as db
from app.db.session import init_db, reset_for_tests
from app.ml.orderbook_features import latest_orderbook_feature_row, load_orderbook_history, merge_orderbook_features
from app.orderbook.service import fetch_and_record_orderbook_snapshot, refresh_all_configured_symbols


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

    def fetch_order_book_metrics(self, symbol, depth=20):
        return self._metrics


def test_fetch_and_record_orderbook_snapshot_persists_to_db():
    metrics = {"bid_volume": 10.0, "ask_volume": 8.0, "imbalance": 0.111, "spread_pct": 0.02}
    result = fetch_and_record_orderbook_snapshot(_FakeExchange(metrics), "BTC/USDT:USDT")
    assert result == metrics

    latest = db.get_latest_orderbook_snapshot("BTC/USDT:USDT")
    assert latest is not None
    assert latest["bid_volume"] == pytest.approx(10.0)
    assert latest["imbalance"] == pytest.approx(0.111)


def test_fetch_and_record_orderbook_snapshot_returns_none_when_exchange_fails():
    result = fetch_and_record_orderbook_snapshot(_FakeExchange(None), "BTC/USDT:USDT")
    assert result is None
    assert db.get_latest_orderbook_snapshot("BTC/USDT:USDT") is None


def test_refresh_all_configured_symbols_handles_per_symbol_failure_independently():
    class _MixedExchange:
        def fetch_order_book_metrics(self, symbol, depth=20):
            if symbol == "BTC/USDT:USDT":
                return {"bid_volume": 5.0, "ask_volume": 5.0, "imbalance": 0.0, "spread_pct": 0.01}
            raise RuntimeError("boom")  # gerçek borsalarda fetch_order_book_metrics zaten None döner, ama savunmacı olalım

    results = {}
    try:
        results = refresh_all_configured_symbols(_MixedExchange(), ["BTC/USDT:USDT"])
    except RuntimeError:
        pytest.fail("bir sembolün hatası diğerlerini etkilememeli")
    assert results["BTC/USDT:USDT"] is not None


def test_merge_orderbook_features_backward_fills_without_lookahead():
    frame = pd.DataFrame(
        {
            # build_features her zaman ns epoch üretir (bkz. app.ml.features) —
            # test fixture'ı bunu birebir taklit etmeli, aksi halde
            # pandas'ın varsayılan çözünürlüğü (us) ile yanlışlıkla eşleşebilir.
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="1h").astype("datetime64[ns]").astype("int64"),
        }
    )
    history = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01T00:30:00Z", "2024-01-01T02:30:00Z"]),
            "bid_volume": [10.0, 20.0],
            "ask_volume": [10.0, 10.0],
            "imbalance": [0.0, 0.25],
            "spread_pct": [0.01, 0.02],
        }
    )

    merged = merge_orderbook_features(frame, history)

    # bar 0 (00:00): henüz hiçbir emir defteri kaydı YOK (00:30 sonra geliyor) -> NaN
    assert pd.isna(merged["orderbook_imbalance"].iloc[0])
    # bar 1 (01:00): 00:30'daki kayıt bilinir -> imbalance 0.0
    assert merged["orderbook_imbalance"].iloc[1] == pytest.approx(0.0)
    # bar 2 (02:00): hâlâ 00:30'daki kayıt geçerli (02:30 henüz gelmedi)
    assert merged["orderbook_imbalance"].iloc[2] == pytest.approx(0.0)
    # bar 3 (03:00): 02:30'daki kayıt artık bilinir -> imbalance 0.25
    assert merged["orderbook_imbalance"].iloc[3] == pytest.approx(0.25)


def test_merge_orderbook_features_all_nan_when_history_empty():
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="1h").astype("int64")})
    merged = merge_orderbook_features(frame, pd.DataFrame())
    assert merged["orderbook_imbalance"].isna().all()


def test_load_orderbook_history_empty_when_db_disabled(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "")
    reset_for_tests()
    assert load_orderbook_history("BTC/USDT:USDT").empty
    reset_for_tests()


def test_latest_orderbook_feature_row_returns_neutral_when_no_history():
    row = latest_orderbook_feature_row("BTC/USDT:USDT")
    assert row == {"orderbook_imbalance": 0.0, "orderbook_spread_norm": 0.0, "orderbook_depth_norm": 0.0}


def test_latest_orderbook_feature_row_reflects_latest_snapshot():
    db.record_orderbook_snapshot("BTC/USDT:USDT", {"bid_volume": 10.0, "ask_volume": 8.0, "imbalance": 0.111, "spread_pct": 0.05})
    row = latest_orderbook_feature_row("BTC/USDT:USDT")
    assert row["orderbook_imbalance"] == pytest.approx(0.111)
    assert row["orderbook_spread_norm"] == pytest.approx(0.05)
