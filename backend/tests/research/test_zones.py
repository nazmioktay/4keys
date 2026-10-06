"""İleriye dönük doğrulama bölgesi (KURALLAR.md §10): bölge sınırları, kilit açma koşulları, kaynak <-> bölge tutarlılığı,
forward bölgesinde uçtan uca koşu (yükleyici/motor/rapor bölge sınırına uyar)."""

import numpy as np
import pandas as pd
import pytest

from research import config, guard, runner
from research.data import store
from research.panel import ForwardOnlyNoOverlapError, ZoneMismatchError, build_panel
from research.sources.base import Source
from research.sources.registry import SOURCE_REGISTRY, register_source
from tests.research.synth import make_market, write_market

T0, FTS = config.FORWARD_EXPLORE_START, config.FORWARD_TEST_START


def _daily(start, end):
    idx = pd.date_range(start, end, freq="D", name="open_time")
    return pd.DataFrame({"close": np.arange(len(idx), dtype=float)}, index=idx)


@pytest.fixture
def log(tmp_path):
    p = tmp_path / "deneyler.md"
    p.write_text("# günlük\nFINAL-TEST-ACILDI e_final\nFORWARD-TEST-ACILDI e_fwd\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------- guard: bölge sınırları
def test_calendar_constants_are_ordered_and_match_the_rules():
    assert config.FINAL_TEST_START < T0 < FTS < config.FORWARD_TEST_MIN_OPEN
    assert (T0, FTS, config.FORWARD_TEST_MIN_OPEN) == (pd.Timestamp("2026-11-01"), pd.Timestamp("2027-11-01"), pd.Timestamp("2028-05-01"))
    text = (config.DOCS_DIR / "KURALLAR.md").read_text(encoding="utf-8")
    for d in ("2026-11-01", "2027-11-01", "2028-05-01"):
        assert d in text  # kurallar belgesi ve kod aynı takvimi taşır


def test_default_zone_is_main_and_the_context_is_restored():
    assert guard.current_zone() == "main"
    with guard.zone("forward"):
        assert guard.current_zone() == "forward"
    assert guard.current_zone() == "main"
    with pytest.raises(ValueError, match="bilinmeyen bölge"):
        with guard.zone("test"):
            pass


def test_main_zone_cuts_at_final_test_and_opened_final_test_stops_at_t0(log):
    df = _daily("2025-09-01", "2027-12-31")
    assert guard.cut_final_test(df).index.max() == pd.Timestamp("2025-09-30")
    opened = guard.cut_final_test(df, True, "e_final", log_path=log)
    assert opened.index.max() == T0 - pd.Timedelta(days=1)  # nihai pencere açılınca bile forward bölgesi GÖRÜNMEZ
    with pytest.raises(guard.FinalTestError):
        guard.cut_final_test(df, True, "e_fwd", log_path=log)  # forward kaydı ana bölgenin kilidini AÇMAZ


def test_forward_zone_sees_only_the_exploration_window(log):
    df = _daily("2025-09-01", "2028-12-31")
    with guard.zone("forward"):
        cut = guard.cut_final_test(df)
        assert cut.index.min() == T0 and cut.index.max() == FTS - pd.Timedelta(days=1)  # nihai pencere ve forward test KESİLİR
        guard.assert_no_final_test(pd.date_range(T0, FTS - pd.Timedelta(days=1)))
        with pytest.raises(guard.FinalTestError, match="önce başlıyor"):
            guard.assert_no_final_test(pd.date_range(T0 - pd.Timedelta(days=3), periods=10))  # nihai pencereye uzanan ısınma
        with pytest.raises(guard.FinalTestError, match="kilitli pencereye"):
            guard.assert_no_final_test(pd.date_range(FTS - pd.Timedelta(days=3), periods=10))


def test_forward_test_opens_only_with_its_own_marker_and_not_before_the_minimum_date(log, monkeypatch):
    df = _daily("2025-09-01", "2028-12-31")
    with guard.zone("forward"):
        with pytest.raises(guard.FinalTestError, match="FORWARD-TEST-ACILDI"):
            guard.cut_final_test(df, True, "e_final", log_path=log)  # nihai test kaydı forward testi AÇMAZ
        monkeypatch.setattr(guard, "_today", lambda: pd.Timestamp("2028-04-30"))
        with pytest.raises(guard.FinalTestError, match="en erken 2028-05-01"):
            guard.cut_final_test(df, True, "e_fwd", log_path=log)
        monkeypatch.setattr(guard, "_today", lambda: pd.Timestamp("2028-05-01"))
        opened = guard.cut_final_test(df, True, "e_fwd", log_path=log)
        assert opened.index.min() == T0 and opened.index.max() == pd.Timestamp("2028-12-31")


def test_store_loaders_follow_the_active_zone(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    frame = _daily("2025-06-01", "2028-01-31").assign(open=1.0, high=1.0, low=1.0, quote_volume=1.0)
    store.write_frame("um_1d", "BTCUSDT", frame)
    assert store.load_klines("BTCUSDT").index.max() < config.FINAL_TEST_START
    with guard.zone("forward"):
        k = store.load_klines("BTCUSDT")
        assert k.index.min() == T0 and k.index.max() < FTS
        assert store.load_panel("close", ["BTCUSDT"]).index.min() == T0


# ---------------------------------------------------------------- kaynak <-> bölge tutarlılığı ve uçtan uca koşu
@pytest.fixture
def fwd_source():
    """Sahte forward_only kaynak: istenen her (gün, sembol) için deterministik bir özellik üretir; 400 gün birikmiş sayılır."""

    @register_source
    class FwdFeat(Source):
        name = "fwd_feat"
        forward_only = True
        feature_names = ("fwd_feat__x",)
        empty = False

        def accumulated_days(self):
            return 400.0

        def data_signature(self):
            return f"empty={FwdFeat.empty}"  # gerçek forward kaynaklar gibi: veri değişince önbellek geçersiz

        def fetch(self, start, end, symbols=None):
            return None

        def to_panel(self, universe, dates):
            if FwdFeat.empty:
                return pd.DataFrame(columns=["date", "symbol", "available_at"])
            rows = pd.MultiIndex.from_product([dates, universe], names=["date", "symbol"]).to_frame(index=False)
            rows["x"] = np.random.default_rng(0).normal(size=len(rows))
            rows["available_at"] = rows["date"] + pd.Timedelta(days=1)
            return self.prefix(rows)

    yield FwdFeat
    SOURCE_REGISTRY.pop("fwd_feat", None)


@pytest.fixture
def fwd_market(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    write_market(make_market(n_symbols=8, n_days=700, start="2026-03-01"))  # nihai pencere + keşif + forward test verisi
    return tmp_path


def _fwd_cfg(**over):
    cfg = {
        "id": "fwd_001", "question": "fwd_deneme", "zone": "forward", "sources": ["ohlcv_core", "fwd_feat"], "universe": {"n": 6},
        "period": {"start": str(T0.date()), "end": str((FTS - pd.Timedelta(days=1)).date())}, "target": {"name": "next_rv", "h": 1},
        "models": ["ridge"], "cv": {"n_splits": 3, "embargo_days": 1, "min_train_days": 150},
        "signal": {"adapter": "vol_target", "params": {"target_vol_annual": 0.4, "cap": 1.0}}, "seed": 0,
        "checks": {"shuffled_target": False, "extra_lag": False, "reproducibility": False},
    }
    cfg.update(over)
    return cfg


def test_zone_must_match_the_presence_of_forward_only_sources(fwd_market, fwd_source):
    with pytest.raises(ZoneMismatchError, match="yalnızca `zone: forward`"):
        build_panel(runner.load_config(_fwd_cfg(zone="main", period={"start": "2025-01-01", "end": "2025-06-30"})))
    with pytest.raises(ZoneMismatchError, match="ana araştırma forward bölgesine giremez"):
        build_panel(runner.load_config(_fwd_cfg(sources=["ohlcv_core"])))
    with pytest.raises(runner.ConfigError, match="zone 'test' geçersiz"):
        runner.load_config(_fwd_cfg(zone="test"))
    assert runner.load_config({k: v for k, v in _fwd_cfg().items() if k != "zone"})["zone"] == "main"


def test_forward_period_must_lie_inside_the_exploration_window(fwd_market, fwd_source):
    with pytest.raises(guard.FinalTestError, match="önce başlıyor"):
        build_panel(runner.load_config(_fwd_cfg(period={"start": "2026-09-01", "end": "2027-06-30"})))  # nihai pencereden başlıyor
    with pytest.raises(guard.FinalTestError, match="kilitli pencereye"):
        build_panel(runner.load_config(_fwd_cfg(period={"start": "2026-11-01", "end": "2027-12-31"})))  # forward teste uzanıyor


def test_forward_panel_contains_only_exploration_data_and_empty_forward_data_fails_loudly(fwd_market, fwd_source):
    panel = build_panel(runner.load_config(_fwd_cfg()))
    assert panel.zone == "forward"
    for frame in (panel.qv_all, panel.prices["open"]):
        assert frame.index.min() >= T0 and frame.index.max() < FTS  # ne nihai pencere ne forward test yüklendi
    dates = panel.X.index.get_level_values("date")
    assert dates.min() >= T0 and dates.max() < FTS
    assert panel.X["fwd_feat__x"].notna().any()
    fwd_source.empty = True
    with pytest.raises(ForwardOnlyNoOverlapError, match="forward keşif"):
        build_panel(runner.load_config(_fwd_cfg(id="fwd_002")))


def test_forward_experiment_runs_end_to_end_inside_its_zone_and_panel_zone_is_checked(fwd_market, fwd_source):
    paths = dict(results_dir=fwd_market / "results", log_path=fwd_market / "deneyler.md", registry_file=fwd_market / "results" / "_registry.json")
    r = runner.run_experiment(_fwd_cfg(), smoke=True, **paths)  # motor/rapor girişindeki tarih kontrolleri forward bölgesinde geçer
    assert r.metrics["zone"] == "forward" and r.smoke and not r.registered
    assert guard.current_zone() == "main"  # bağlam koşudan sonra geri alınır
    fwd_panel = build_panel(runner.load_config(_fwd_cfg()))
    with pytest.raises(runner.ConfigError, match="bölgesinde kurulmuş"):
        runner.run_experiment(_fwd_cfg(zone="main"), smoke=True, panel=fwd_panel, **paths)  # forward paneli ana bölgede koşturulamaz


def test_forward_db_sources_are_cut_to_the_zone_so_pre_t0_rows_do_not_leak_into_diffs(monkeypatch):
    """Denetim bulgusu: OIDetail.oi_chg_1d satır diff()'i ile hesaplanır; T0 öncesi (nihai pencere) satırı T0 değerine sızmamalı."""
    from datetime import datetime, timezone

    from app.core.config import settings
    from app.db import repository as dbrepo
    from app.db.models import OIDetailSnapshot
    from app.db.session import init_db, reset_for_tests, session_scope
    from research.sources.forward import OIDetail

    monkeypatch.setattr(settings, "database_url", "sqlite:///:memory:")
    reset_for_tests()
    init_db()
    try:
        times = [datetime(2026, 10, 31, 12, tzinfo=timezone.utc), datetime(2026, 11, 1, 12, tzinfo=timezone.utc),
                 datetime(2026, 11, 2, 12, tzinfo=timezone.utc), datetime(2027, 11, 5, 12, tzinfo=timezone.utc)]
        for v in (1e6, 2e6, 4e6, 8e6):
            dbrepo.record_oi_detail_snapshot("BTCUSDT", {"open_interest": 1.0, "open_interest_value": v})
        with session_scope() as sess:
            for row, t in zip(sess.query(OIDetailSnapshot).order_by(OIDetailSnapshot.id).all(), times):
                row.time = t
        src = OIDetail()
        with guard.zone("forward"):
            fetched = src.fetch(None, None, ["BTCUSDT"])
            assert len(fetched) == 2  # 2026-10-31 (nihai pencere) ve 2027-11-05 (forward test) KESİLDİ
            panel = src.to_panel(["BTCUSDT"], pd.date_range(T0, periods=2)).set_index("date")
            assert np.isnan(panel.loc[T0, "oi_detail__oi_chg_1d"])  # nihai penceredeki 1e6 kullanılmadı
            assert panel.loc[T0 + pd.Timedelta(days=1), "oi_detail__oi_chg_1d"] == pytest.approx(np.log(2.0))
        assert src.fetch(None, None, ["BTCUSDT"]).empty  # ana bölge: hepsi >= 2025-10-01 -> görünmez
    finally:
        reset_for_tests()


def test_min_train_days_longer_than_the_period_is_rejected_with_a_forward_hint():
    with pytest.raises(runner.ConfigError, match="etkin keşif"):
        runner.load_config(_fwd_cfg(cv={"n_splits": 3, "embargo_days": 1, "min_train_days": 300}))  # 300 < 365 ama > 365 - 90
    with pytest.raises(runner.ConfigError, match="dönem uzunluğundan"):
        runner.load_config(_fwd_cfg(zone="main", period={"start": "2024-01-01", "end": "2024-06-30"}, cv={"min_train_days": 365}))
