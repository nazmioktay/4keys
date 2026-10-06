"""İleriye dönük toplayıcılar (app/forwardcollect): ağsız; sahte WebSocket/HTTP ve bellek içi SQLite."""

import asyncio
import json

import pandas as pd
import pytest

from app.core.config import settings
from app.db import repository as db
from app.db.session import init_db, reset_for_tests
from app.forwardcollect import depth, liquidations, oi_detail
from app.scheduler import jobs, scheduler as scheduler_module, status
from app.scheduler.scheduler import get_scheduler, start_scheduler, stop_scheduler


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "sqlite:///:memory:")
    reset_for_tests()
    init_db()
    yield
    reset_for_tests()


def _msg(symbol="BTCUSDT", side="SELL", price="60000", ap="59990", z="0.5", q="0.5", t=1_700_000_000_000, status_="FILLED"):
    return json.dumps({"e": "forceOrder", "E": t, "o": {"s": symbol, "S": side, "o": "LIMIT", "f": "IOC", "q": q, "p": price, "ap": ap, "X": status_, "l": z, "z": z, "T": t}})


# ---------------------------------------------------------------- tasfiye akışı
def test_parse_force_order_extracts_fields_and_rejects_garbage():
    row = liquidations.parse_force_order(_msg())
    assert row["symbol"] == "BTCUSDT" and row["side"] == "SELL" and row["quantity"] == 0.5
    assert row["quote_value"] == pytest.approx(0.5 * 59990) and row["time"].tzinfo is not None
    assert liquidations.parse_force_order("{bozuk") is None
    assert liquidations.parse_force_order(json.dumps({"e": "aggTrade"})) is None
    unfilled = liquidations.parse_force_order(_msg(z="0", ap="0", q="2.0"))
    assert unfilled["quantity"] == 2.0 and unfilled["avg_price"] is None and unfilled["quote_value"] == pytest.approx(2.0 * 60000)


def test_collector_filters_symbols_buffers_and_writes_deduplicated_rows():
    c = liquidations.LiquidationCollector(symbols=["BTCUSDT"], flush_seconds=999)
    assert c.ingest(_msg()) and c.ingest(_msg()) and not c.ingest(_msg(symbol="ETHUSDT")) and not c.ingest("{x")
    assert c.flush() == 2
    stored = db.get_liquidation_events()
    assert len(stored) == 1  # aynı olay tekrar geldi: benzersizlik kısıtı yok sayar
    assert stored["symbol"].tolist() == ["BTCUSDT"] and stored["side"].iloc[0] == "SELL"
    assert c.stats["messages"] == 4 and c.stats["parsed"] == 2 and c.stats["flushed"] == 2


def test_flush_failure_keeps_rows_for_retry():
    calls = []

    def flaky(rows):
        calls.append(len(rows))
        if len(calls) == 1:
            raise RuntimeError("DB yok")
        return len(rows)

    c = liquidations.LiquidationCollector(flush_seconds=999, flush_fn=flaky)
    c.ingest(_msg())
    assert c.flush() == 0 and len(c._buffer) == 1  # satır kaybolmadı
    c.ingest(_msg(t=1_700_000_001_000))
    assert c.flush() == 2 and calls == [1, 2]


def test_process_stream_flushes_on_interval_and_at_end():
    async def stream():
        for i in range(5):
            yield _msg(t=1_700_000_000_000 + i * 1000, price=str(60000 + i))

    c = liquidations.LiquidationCollector(flush_seconds=999)
    asyncio.run(c.process_stream(stream()))
    assert len(db.get_liquidation_events()) == 5  # akış bitince son tampon yazıldı


def test_run_forever_reconnects_with_exponential_backoff_then_succeeds():
    sleeps, attempts = [], []

    class _WS:
        def __init__(self, ok): self.ok = ok
        async def __aenter__(self):
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("koptu")
            return self
        async def __aexit__(self, *a): return False
        def __aiter__(self): return self._gen()
        async def _gen(self):
            yield _msg()
            c._stop.set()  # bağlantı kurulup mesaj alınınca durdur

    async def fake_sleep(s):
        sleeps.append(s)

    c = liquidations.LiquidationCollector(connect=lambda: _WS(True), flush_seconds=999, sleep=fake_sleep)
    asyncio.run(c.run_forever())
    assert sleeps == [1.0, 2.0]  # iki başarısız bağlantı: 1 sn, 2 sn
    assert c.stats["reconnects"] == 2 and len(db.get_liquidation_events()) == 1


# ---------------------------------------------------------------- derinlik bantları
def test_depth_bands_sum_notional_within_band_and_mark_uncovered_bands_missing():
    bids = [[100.0 - 0.05 * i, 2.0] for i in range(1, 31)]  # 99.95 .. 98.5 (≈150 bp aşağı)
    asks = [[100.0 + 0.05 * i, 1.0] for i in range(1, 31)]  # 100.05 .. 101.5
    out = depth.compute_depth_bands([[99.95, 2.0]] + bids[1:], [[100.05, 1.0]] + asks[1:])
    assert out["mid_price"] == pytest.approx(100.0) and out["spread_bps"] == pytest.approx(10.0, abs=0.01)
    assert out["bid_10bp"] == pytest.approx(sum(p * q for p, q in bids if p >= 99.9))  # yalnızca ≥ mid×(1-10bp)
    assert out["ask_10bp"] == pytest.approx(sum(p * q for p, q in asks if p <= 100.1))
    assert 140 < out["depth_coverage_bp"] <= 150 and out["bid_100bp"] is not None
    assert out["bid_200bp"] is None and out["ask_500bp"] is None  # kapsanmayan bant: EKSİK, sıfır değil
    assert depth.compute_depth_bands([], []) is None


def test_collect_depth_persists_and_tolerates_failures():
    def get_json(url, params=None):
        if params["symbol"] == "FAILUSDT":
            raise RuntimeError("HTTP 500")
        return {"bids": [["99.95", "2"], ["99.5", "1"]], "asks": [["100.05", "1"], ["100.5", "3"]]}

    res = depth.collect_depth(["BTCUSDT", "FAILUSDT"], get_json)
    assert res["BTCUSDT"] is not None and res["FAILUSDT"] is None
    assert db.get_depth_band_snapshots()["symbol"].tolist() == ["BTCUSDT"]


# ---------------------------------------------------------------- ayrıntılı OI
def _oi_http(fail=()):
    def get_json(url, params=None):
        for f in fail:
            if f in url:
                raise RuntimeError("HTTP")
        if url.endswith("/fapi/v1/openInterest"):
            return {"openInterest": "1000.5"}
        if "openInterestHist" in url:
            return [{"sumOpenInterestValue": "6e7"}]
        if "takerlongshortRatio" in url:
            return [{"buySellRatio": "1.1", "buyVol": "55", "sellVol": "50"}]
        return [{"longShortRatio": "1.7"}]

    return get_json


def test_oi_detail_collects_all_fields_and_allows_partial_rows():
    full = oi_detail.fetch_oi_detail("BTCUSDT", _oi_http())
    assert full["open_interest"] == 1000.5 and full["open_interest_value"] == 6e7 and full["top_ls_account"] == 1.7
    assert full["taker_buy_sell_ratio"] == 1.1 and full["taker_buy_vol"] == 55.0
    partial = oi_detail.fetch_oi_detail("BTCUSDT", _oi_http(fail=("topLongShortAccountRatio", "takerlongshortRatio")))
    assert "top_ls_account" not in partial and partial["open_interest"] == 1000.5  # başarısız uçlar yok, kalanlar var
    assert oi_detail.fetch_oi_detail("BTCUSDT", _oi_http(fail=("/fapi/", "/futures/"))) is None
    oi_detail.collect_oi_detail(["BTCUSDT"], _oi_http())
    stored = db.get_oi_detail_snapshots()
    assert len(stored) == 1 and stored["global_ls_account"].iloc[0] == 1.7 and pd.isna(stored["taker_sell_vol"].iloc[0]) is False


# ---------------------------------------------------------------- job'lar ve zamanlayıcı kaydı
def test_forward_jobs_record_status_and_never_raise(monkeypatch):
    monkeypatch.setattr(settings, "forward_collector_symbols", "BTCUSDT,ETHUSDT")
    monkeypatch.setattr(jobs, "collect_depth", lambda symbols: {"BTCUSDT": {"x": 1}, "ETHUSDT": None})
    jobs.job_collect_depth_bands()
    st = status.get_all()[jobs.FORWARD_DEPTH_JOB_ID]
    assert st.ok and "ETHUSDT" in st.detail
    monkeypatch.setattr(jobs, "collect_oi_detail", lambda symbols: (_ for _ in ()).throw(RuntimeError("boom")))
    jobs.job_collect_oi_detail()
    assert status.get_all()[jobs.FORWARD_OI_DETAIL_JOB_ID].ok is False


@pytest.fixture
def _quiet_scheduler(monkeypatch):
    monkeypatch.setattr(jobs, "refresh_screener", lambda: [])
    monkeypatch.setattr(jobs, "run_cycle_once", lambda: [])
    monkeypatch.setattr(jobs, "job_refresh_macro", lambda: None, raising=False)
    started = []
    monkeypatch.setattr(scheduler_module, "start_liquidation_collector", lambda symbols, flush: started.append((symbols, flush)))
    monkeypatch.setattr(jobs, "collect_depth", lambda symbols: {})
    monkeypatch.setattr(jobs, "collect_oi_detail", lambda symbols: {})
    status.reset()
    stop_scheduler()
    yield started
    stop_scheduler()


def test_forward_collectors_are_off_by_default(_quiet_scheduler):
    assert settings.forward_collectors_enabled is False
    sched = start_scheduler(enabled=True)
    assert sched.get_job(jobs.FORWARD_DEPTH_JOB_ID) is None and sched.get_job(jobs.FORWARD_OI_DETAIL_JOB_ID) is None
    assert _quiet_scheduler == []  # tasfiye akışı da başlatılmadı


def test_forward_collectors_register_jobs_and_start_the_stream_when_enabled(_quiet_scheduler, monkeypatch):
    monkeypatch.setattr(settings, "forward_collectors_enabled", True)
    monkeypatch.setattr(settings, "forward_collector_symbols", "BTCUSDT, ethusdt")
    sched = start_scheduler(enabled=True)
    assert sched.get_job(jobs.FORWARD_DEPTH_JOB_ID) is not None and sched.get_job(jobs.FORWARD_OI_DETAIL_JOB_ID) is not None
    assert _quiet_scheduler == [(["BTCUSDT", "ETHUSDT"], settings.forward_liquidation_flush_seconds)]
    assert get_scheduler() is sched
