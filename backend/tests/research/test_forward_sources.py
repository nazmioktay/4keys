"""forward_only kaynaklar (liquidations, depth_bands, oi_detail): DB'den okuma, günlük özet, 12 ay kuralı."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.core.config import settings
from app.db import repository as db
from app.db.session import init_db, reset_for_tests
from research.panel import ForwardOnlyTooShortError, check_forward_only
from research.sources import pit
from research.sources.base import validate_panel
from research.sources.registry import get_source


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "sqlite:///:memory:")
    reset_for_tests()
    init_db()
    yield
    reset_for_tests()


def _ts(day: str, hour: int = 12) -> datetime:
    return datetime.fromisoformat(f"{day}T{hour:02d}:00:00").replace(tzinfo=timezone.utc)


def test_liquidations_daily_panel_zero_fills_only_inside_the_collection_window():
    rows = [
        {"time": _ts("2024-01-02"), "symbol": "BTCUSDT", "side": "SELL", "price": 40000.0, "avg_price": 40000.0, "quantity": 1.0, "quote_value": 40000.0, "status": "FILLED"},
        {"time": _ts("2024-01-02", 15), "symbol": "BTCUSDT", "side": "BUY", "price": 40100.0, "avg_price": 40100.0, "quantity": 0.5, "quote_value": 20050.0, "status": "FILLED"},
        {"time": _ts("2024-01-05"), "symbol": "BTCUSDT", "side": "SELL", "price": 39000.0, "avg_price": 39000.0, "quantity": 2.0, "quote_value": 78000.0, "status": "FILLED"},
    ]
    db.record_liquidation_events(rows)
    src = get_source("liquidations")
    dates = pd.date_range("2023-12-25", "2024-01-10")
    panel = src.to_panel(["BTCUSDT"], dates).set_index("date")
    validate_panel(src, panel.reset_index())
    assert panel.loc["2024-01-02", "liquidations__liq_long_usd"] == 40000.0 and panel.loc["2024-01-02", "liquidations__liq_short_usd"] == 20050.0
    assert panel.loc["2024-01-02", "liquidations__liq_net_usd"] == pytest.approx(20050.0 - 40000.0)
    assert panel.loc["2024-01-03", "liquidations__liq_count"] == 0  # toplama penceresi İÇİNDE olaysız gün = 0
    assert "2023-12-30" not in panel.index.strftime("%Y-%m-%d") and "2024-01-08" not in panel.index.strftime("%Y-%m-%d")  # pencere DIŞI: satır yok (NaN kalır)
    assert (panel["available_at"] == panel.index + pd.Timedelta(days=1)).all()
    assert src.forward_only is True


def test_depth_bands_daily_imbalance_and_missing_bands_stay_nan():
    for h, bid, ask in [(1, 300.0, 100.0), (13, 100.0, 100.0)]:
        db.record_depth_band_snapshot("BTCUSDT", {"time": _ts("2024-02-01", h), "mid_price": 100.0, "spread_bps": 2.0 + h, "depth_coverage_bp": 120.0,
                                                  "bid_10bp": bid, "ask_10bp": ask, "bid_50bp": bid * 2, "ask_50bp": ask * 2,
                                                  "bid_100bp": bid * 3, "ask_100bp": ask * 3, "bid_200bp": None, "ask_200bp": None})
    # _record_snapshot `time`'ı kolonlar arasında almaz (default utcnow): satır zamanlarını doğrudan güncelle
    from app.db.models import DepthBandSnapshot
    from app.db.session import session_scope

    with session_scope() as s:
        for row, h in zip(s.query(DepthBandSnapshot).order_by(DepthBandSnapshot.id).all(), (1, 13)):
            row.time = _ts("2024-02-01", h)
    src = get_source("depth_bands")
    panel = src.to_panel(["BTCUSDT"], pd.date_range("2024-01-28", "2024-02-05"))
    row = panel[panel["date"] == "2024-02-01"].iloc[0]
    assert row["depth_bands__imb_10"] == pytest.approx(((300 - 100) / 400 + 0.0) / 2)  # iki anlık görüntünün ortalaması
    assert row["depth_bands__n_snapshots"] == 2 and row["depth_bands__spread_bps"] == pytest.approx((3 + 15) / 2)


def test_oi_detail_daily_last_and_change():
    from app.db.models import OIDetailSnapshot
    from app.db.session import session_scope

    for day, val in [("2024-03-01", 1e9), ("2024-03-02", 1.1e9), ("2024-03-03", 1.21e9)]:
        db.record_oi_detail_snapshot("BTCUSDT", {"open_interest": 100.0, "open_interest_value": val, "top_ls_account": 1.5, "global_ls_account": 1.2, "taker_buy_sell_ratio": 1.05})
    with session_scope() as s:
        for row, day in zip(s.query(OIDetailSnapshot).order_by(OIDetailSnapshot.id).all(), ("2024-03-01", "2024-03-02", "2024-03-03")):
            row.time = _ts(day)
    panel = get_source("oi_detail").to_panel(["BTCUSDT"], pd.date_range("2024-02-28", "2024-03-05")).set_index("date")
    assert panel.loc["2024-03-02", "oi_detail__oi_chg_1d"] == pytest.approx(np.log(1.1))
    assert panel.loc["2024-03-03", "oi_detail__oi_chg_1d"] == pytest.approx(np.log(1.1))
    assert panel.loc["2024-03-01", "oi_detail__oi_value_last"] == 1e9


def test_twelve_month_rule_rejects_short_history_and_accepts_long_history():
    src = get_source("oi_detail")
    assert src.accumulated_days() is None and src.history_start is None  # hiç veri yok
    with pytest.raises(ForwardOnlyTooShortError, match="forward_only"):
        check_forward_only([src])
    from app.db.models import OIDetailSnapshot
    from app.db.session import session_scope

    for _ in range(2):
        db.record_oi_detail_snapshot("BTCUSDT", {"open_interest": 1.0})
    with session_scope() as s:
        rows = s.query(OIDetailSnapshot).order_by(OIDetailSnapshot.id).all()
        rows[0].time, rows[1].time = _ts("2023-01-01"), _ts("2023-06-01")
    assert src.accumulated_days() == pytest.approx(151.0, abs=1)
    with pytest.raises(ForwardOnlyTooShortError, match="151|150"):
        check_forward_only([src])  # 5 ay < 12 ay
    with session_scope() as s:
        rows = s.query(OIDetailSnapshot).order_by(OIDetailSnapshot.id).all()
        rows[1].time = _ts("2024-02-01")
    assert src.accumulated_days() > 365
    check_forward_only([src])  # artık geçer
    assert src.history_start == pd.Timestamp("2023-01-01 12:00")
    non_forward = get_source("ohlcv_core")
    check_forward_only([non_forward])  # forward_only olmayanlar etkilenmez


def test_forward_sources_merge_point_in_time_with_missing_flag_when_no_data():
    from research.sources.pit import decision_frame, merge_sources

    src = get_source("liquidations")
    dates = pd.date_range("2024-01-01", periods=3)
    merged = merge_sources(decision_frame(dates, symbols=["BTCUSDT"]), [(src, src.to_panel(["BTCUSDT"], dates))])
    assert (merged["liquidations__missing"] == 1).all()  # veri yok -> NaN + eksik bayrağı, sıfır doldurma yok
    assert merged.filter(like="liquidations__liq_").isna().all().all()
