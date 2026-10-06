import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from research import config
from research.data import download_metrics, store
from research.sources import cache, coverage, pit
from research.sources.base import MARKET, Source, validate_panel
from research.sources.registry import SOURCE_REGISTRY, available_sources, get_source, register_source

from tests.research.test_data import FakeHttp, _listing_xml, _zip


# ---------------------------------------------------------------- yardımcılar
@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    return tmp_path


def _klines(n=200, start="2024-01-01", seed=0, price=100.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="D", name="open_time")
    close = price * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = np.r_[price, close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    qv = rng.uniform(1e6, 2e6, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": qv / close, "quote_volume": qv,
                         "trades": 100.0, "taker_buy_base": qv / close * 0.5, "taker_buy_quote": qv * 0.55}, index=idx)


class _Toy(Source):
    """Test kaynağı: değer = gün numarası; available_at gecikmesi parametreli."""

    name = "toy"
    version = "1"
    max_staleness = pd.Timedelta(days=3)

    def __init__(self, lag_days=0.0, scope="symbol", **params):
        super().__init__(lag_days=lag_days, **params)
        self.lag = pd.Timedelta(days=lag_days)
        self.scope = scope

    def fetch(self, start, end, symbols=None):
        return None

    def to_panel(self, universe, dates):
        syms = [MARKET] if self.scope == "market" else universe
        rows = [{"date": d, "symbol": s, "toy__x": float(d.dayofyear), "available_at": d + pd.Timedelta(days=1) + self.lag}
                for d in dates for s in syms]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------- registry / arayüz
def test_registry_lists_core_sources_and_rejects_unknown_names():
    names = available_sources()
    for expected in ("ohlcv_core", "funding", "taker_flow", "derivatives_metrics", "macro", "sentiment_fng"):
        assert expected in names
    with pytest.raises(KeyError, match="Bilinmeyen kaynak"):
        get_source("yok_boyle_kaynak")
    assert get_source("ohlcv_core").describe()["forward_only"] is False


def test_register_conflict_and_prefix_contract():
    class Dup(Source):
        name = "ohlcv_core"

        def fetch(self, start, end, symbols=None): ...
        def to_panel(self, universe, dates): ...

    with pytest.raises(ValueError, match="zaten kayıtlı"):
        register_source(Dup)
    bad = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "symbol": ["A"], "x": [1.0], "available_at": [pd.Timestamp("2024-01-02")]})
    with pytest.raises(ValueError, match="önekli"):
        validate_panel(_Toy(), bad)
    market_bad = _Toy(scope="market").to_panel(["A"], pd.date_range("2024-01-01", periods=2))
    market_bad["symbol"] = "A"
    with pytest.raises(ValueError, match="piyasa-geneli"):
        validate_panel(_Toy(scope="market"), market_bad)


# ---------------------------------------------------------------- PIT birleştirme
def test_pit_merge_uses_latest_value_available_at_decision_time_without_zero_fill():
    dates = pd.date_range("2024-01-10", periods=5)
    toy = _Toy(lag_days=1.0)  # d gününün değeri d+2 gün 00:00'da kullanılabilir
    panel = toy.to_panel(["A"], dates)
    decisions = pit.decision_frame(dates, symbols=["A"])
    merged = pit.merge_sources(decisions, [(toy, panel)])
    # karar d kapanışında (d+1): d günü değeri henüz yok (d+2'de çıkar), d-1 günü değeri VAR
    for i, d in enumerate(dates):
        value = merged.loc[i, "toy__x"]
        if i == 0:
            assert np.isnan(value)  # önceki gün paneli yok
        else:
            assert value == float((d - pd.Timedelta(days=1)).dayofyear)
    assert merged.loc[0, "toy__missing"] == 1 and merged.loc[1, "toy__missing"] == 0
    assert not (merged["toy__x"] == 0).any()  # NaN, sıfır DEĞİL
    assert pit.audit_available_at(merged, [toy]) == {"toy": 0}


def test_pit_merge_respects_staleness_tolerance_and_per_symbol_grouping():
    dates = pd.date_range("2024-01-01", periods=12)
    toy = _Toy()
    full = toy.to_panel(["A", "B"], dates)
    # B'nin 4-10. günleri yok -> 3 günden eski değer kullanılmaz (NaN)
    panel = full[~((full["symbol"] == "B") & (full["date"] >= "2024-01-04") & (full["date"] <= "2024-01-10"))]
    merged = pit.merge_sources(pit.decision_frame(dates, symbols=["A", "B"]), [(toy, panel)])
    b = merged[merged["symbol"] == "B"].set_index("date")
    assert b.loc["2024-01-05", "toy__missing"] == 0  # son değer 2 gün eski: tolerans içinde
    assert b.loc["2024-01-07", "toy__missing"] == 1  # son değer 4 gün eski: tolerans (3 gün) aşıldı -> NaN
    assert b.loc["2024-01-09", "toy__missing"] == 1
    a = merged[merged["symbol"] == "A"]
    assert (a["toy__missing"] == 0).all()


def test_pit_market_scope_broadcasts_to_all_symbols():
    dates = pd.date_range("2024-01-01", periods=4)
    toy = _Toy(scope="market")
    merged = pit.merge_sources(pit.decision_frame(dates, symbols=["A", "B", "C"]), [(toy, toy.to_panel(["A", "B", "C"], dates))])
    assert merged.groupby("date")["toy__x"].nunique().max() == 1
    assert len(merged) == 12 and merged["toy__x"].notna().all()


def test_audit_detects_available_at_after_decision_time():
    toy = _Toy()
    decisions = pit.decision_frame(pd.date_range("2024-01-01", periods=3), symbols=["A"])
    merged = decisions.copy()
    merged["toy__available_at"] = merged["decision_time"] + pd.Timedelta(hours=1)  # ihlal
    assert pit.audit_available_at(merged, [toy]) == {"toy": 3}
    assert "toy__available_at" not in pit.feature_matrix_columns(merged)


# ---------------------------------------------------------------- önbellek
def test_cache_builds_once_and_key_changes_with_params_version_and_data(tmp_cache):
    dates = pd.date_range("2024-01-01", periods=3)
    calls = []

    class Counting(_Toy):
        def to_panel(self, universe, dates):
            calls.append(1)
            return super().to_panel(universe, dates)

    a = Counting(lag_days=0.0)
    cache.load_or_build(a, ["A"], dates, "hashA")
    cache.load_or_build(a, ["A"], dates, "hashA")
    assert len(calls) == 1
    cache.load_or_build(a, ["A"], dates, "hashB")  # veri değişti
    cache.load_or_build(Counting(lag_days=1.0), ["A"], dates, "hashA")  # parametre değişti
    assert len(calls) == 3
    keys = {cache.cache_key(a, "hashA", ["A"], dates), cache.cache_key(a, "hashB", ["A"], dates), cache.cache_key(Counting(lag_days=1.0), "hashA", ["A"], dates)}
    assert len(keys) == 3


# ---------------------------------------------------------------- ohlcv_core / taker_flow
def test_ohlcv_core_features_match_hand_computation_and_are_point_in_time(tmp_cache):
    df = _klines(200, seed=1)
    store.write_frame("um_1d", "AAAUSDT", df)
    dates = pd.date_range("2024-05-01", "2024-06-30")
    src = get_source("ohlcv_core")
    panel = src.to_panel(["AAAUSDT"], dates)
    validate_panel(src, panel)
    row = panel[panel["date"] == pd.Timestamp("2024-06-01")].iloc[0]
    close = df["close"]
    assert row["ohlcv_core__ret_1"] == pytest.approx(np.log(close["2024-06-01"] / close["2024-05-31"]))
    assert row["ohlcv_core__ret_7"] == pytest.approx(np.log(close["2024-06-01"] / close["2024-05-25"]))
    pk = np.sqrt(np.log(df.loc["2024-06-01", "high"] / df.loc["2024-06-01", "low"]) ** 2 / (4 * np.log(2)))
    assert row["ohlcv_core__rv_pk_1"] == pytest.approx(pk)
    assert row["available_at"] == pd.Timestamp("2024-06-02")  # bar kapanışı
    assert set(panel["date"]) <= set(dates)
    # gelecek bozulursa geçmiş özellikler değişmez
    broken = df.copy()
    broken.loc["2024-06-10":, ["open", "high", "low", "close"]] *= 7.0
    store.write_frame("um_1d", "AAAUSDT", broken)
    panel2 = src.to_panel(["AAAUSDT"], dates)
    cols = [c for c in panel.columns if c.startswith("ohlcv_core__")]
    early = panel["date"] < pd.Timestamp("2024-06-10")
    pd.testing.assert_frame_equal(panel.loc[early, cols].reset_index(drop=True), panel2.loc[panel2["date"] < pd.Timestamp("2024-06-10"), cols].reset_index(drop=True))


def test_taker_flow_ratio_and_availability(tmp_cache):
    df = _klines(60, seed=2)
    store.write_frame("um_1d", "AAAUSDT", df)
    dates = pd.date_range("2024-02-01", periods=20)
    panel = get_source("taker_flow").to_panel(["AAAUSDT"], dates)
    row = panel.iloc[0]
    d = row["date"]
    assert row["taker_flow__ratio"] == pytest.approx(df.loc[d, "taker_buy_quote"] / df.loc[d, "quote_volume"])
    assert (panel["available_at"] == panel["date"] + pd.Timedelta(days=1)).all()


# ---------------------------------------------------------------- funding
def test_funding_daily_bucket_matches_engine_convention(tmp_cache):
    times = pd.to_datetime(["2024-03-01 00:00", "2024-03-01 08:00", "2024-03-01 16:00", "2024-03-02 00:00", "2024-03-02 08:00"])
    rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
    store.write_frame("um_funding", "AAAUSDT", pd.DataFrame({"rate": rates, "interval_hours": 8.0}, index=pd.DatetimeIndex(times, name="time")))
    panel = get_source("funding").to_panel(["AAAUSDT"], pd.date_range("2024-03-01", periods=2))
    d1 = panel[panel["date"] == "2024-03-01"].iloc[0]
    # (03-01 00:00, 03-02 00:00]: 08:00, 16:00, 00:00(03-02)
    assert d1["funding__sum"] == pytest.approx(0.0002 + 0.0003 + 0.0004)
    assert d1["available_at"] == pd.Timestamp("2024-03-02 00:00")
    assert (panel["available_at"] <= panel["date"] + pd.Timedelta(days=1)).all()


# ---------------------------------------------------------------- derivatives_metrics
def _metrics_zip(day="2024-03-01", n=288):
    header = "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio"
    t0 = pd.Timestamp(day)
    rows = [f"{t0 + pd.Timedelta(minutes=5 * i)},BTCUSDT,{1000 + i},{(1000 + i) * 50},1.{i % 10},0.9,1.1,{0.8 + i / 1000}" for i in range(n)]
    return _zip("\n".join([header] + rows))


def test_metrics_parse_daily_aggregate_last_and_mean():
    df = download_metrics.parse_metrics_zip(_metrics_zip())
    assert len(df) == 288 and df["oi"].iloc[-1] == 1287
    daily = download_metrics.daily_aggregate(df)
    assert len(daily) == 1
    assert daily["oi_value_last"].iloc[0] == 1287 * 50
    assert daily["taker_ls_vol_mean"].iloc[0] == pytest.approx(df["taker_ls_vol"].mean())


def test_metrics_coverage_lists_files_without_downloading():
    prefix = "data/futures/um/daily/metrics/BTCUSDT/"
    keys = [f"{prefix}BTCUSDT-metrics-2024-03-0{i}.zip" for i in (1, 2, 3)]
    http = FakeHttp({(prefix, None): _listing_xml(keys=keys), ("data/futures/um/daily/metrics/NEWUSDT/", None): _listing_xml(keys=[])}, {})
    table = download_metrics.metrics_coverage(http, ["BTCUSDT", "NEWUSDT"], workers=2)
    assert table.loc["BTCUSDT", "n_files"] == 3 and table.loc["BTCUSDT", "first"] == "2024-03-01" and table.loc["BTCUSDT", "last"] == "2024-03-03"
    assert table.loc["NEWUSDT", "n_files"] == 0
    assert not any(c[0].startswith(download_metrics.BASE) for c in http.calls)  # hiç dosya indirilmedi


def test_derivatives_metrics_source_end_to_end(tmp_cache):
    http_files = {}
    prefix = "data/futures/um/daily/metrics/BTCUSDT/"
    keys = []
    for i in range(10):
        day = (pd.Timestamp("2024-03-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        key = f"{prefix}BTCUSDT-metrics-{day}.zip"
        keys.append(key)
        http_files[f"{download_metrics.BASE}/{key}"] = _metrics_zip(day)
    http = FakeHttp({(prefix, None): _listing_xml(keys=keys)}, http_files)
    out = download_metrics.sync_metrics(http, "BTCUSDT")
    assert out["status"] == "ok" and out["rows"] == 10
    assert download_metrics.sync_metrics(http, "BTCUSDT")["status"] == "cached"
    src = get_source("derivatives_metrics")
    panel = src.to_panel(["BTCUSDT"], pd.date_range("2024-03-02", periods=8))
    validate_panel(src, panel)
    assert (panel["available_at"] == panel["date"] + pd.Timedelta(days=1)).all()
    assert panel["derivatives_metrics__oi_chg_1d"].notna().any()
    assert src.history_start == pd.Timestamp("2024-03-01")


# ---------------------------------------------------------------- sentiment_fng
def test_fng_value_visible_to_close_of_same_day_but_not_the_previous_one():
    ts = lambda d: int(pd.Timestamp(d).timestamp())  # noqa: E731
    data = [{"value": str(v), "timestamp": str(ts(d)), "value_classification": "x"} for v, d in [(20, "2024-03-01"), (60, "2024-03-02"), (80, "2024-03-03")]]
    src = get_source("sentiment_fng", fetcher=lambda: data)
    dates = pd.date_range("2024-03-01", periods=3)
    panel = src.to_panel(["A", "B"], dates)
    validate_panel(src, panel)
    assert (panel["symbol"] == MARKET).all() and panel["sentiment_fng__fng"].tolist() == [0.2, 0.6, 0.8]
    merged = pit.merge_sources(pit.decision_frame(dates, symbols=["A"]), [(src, panel)])
    # karar d kapanışı (d+1 00:00) >= d 00:00 + 6s -> d değeri kullanılabilir
    assert merged["sentiment_fng__fng"].tolist() == [0.2, 0.6, 0.8]
    # ama d gününün İÇİNDE (d 00:00 + 3s) henüz yok
    early = pd.DataFrame({"date": [pd.Timestamp("2024-03-02")], "symbol": ["A"], "decision_time": [pd.Timestamp("2024-03-02 03:00")]})
    assert pit.merge_sources(early, [(src, panel)]).loc[0, "sentiment_fng__fng"] == 0.2  # 03-02 değeri (06:00'da çıkar) YOK; önceki gün (0,2) var


# ---------------------------------------------------------------- macro
def _macro_fetcher(n=120, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return lambda start, end: {
        "vix": pd.Series(15 + np.cumsum(rng.normal(0, 0.3, n)), index=idx),
        "sp500": pd.Series(4000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx),
        "nikkei": pd.Series(30000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx),
    }


def test_macro_availability_uses_conservative_close_hours_and_leaves_weekends_nan():
    src = get_source("macro", fetcher=_macro_fetcher())
    dates = pd.date_range("2024-03-01", "2024-04-15")
    panel = src.to_panel(["A"], dates)
    validate_panel(src, panel)
    row = panel[panel["date"] == pd.Timestamp("2024-03-04")].iloc[0]  # Pazartesi
    assert row["available_at"] == pd.Timestamp("2024-03-04 23:00")  # vix 22:00 + 1s (en geç seri)
    assert (panel["available_at"] <= panel["date"] + pd.Timedelta(days=1)).all()
    assert pd.Timestamp("2024-03-09") not in set(panel["date"])  # Cumartesi satırı yok
    merged = pit.merge_sources(pit.decision_frame(dates, symbols=["A", "B"]), [(src, panel)])
    sat = merged[merged["date"] == pd.Timestamp("2024-03-09")]
    # hafta sonu kararı Cuma'nın değerini taşır (geçmiş, bayatlık toleransı içinde); sıfır doldurma yok
    assert (sat["macro__vix_z"].notna()).all() and (sat["macro__missing"] == 0).all()


def test_macro_expanding_zscore_ignores_the_future():
    dates = pd.date_range("2024-03-01", "2024-04-15")
    base = get_source("macro", fetcher=_macro_fetcher(seed=5)).to_panel(["A"], dates)
    rng = np.random.default_rng(5)

    def shifted(start, end):
        out = _macro_fetcher(seed=5)(start, end)
        for k, s in out.items():
            s = s.copy()
            s[s.index >= "2024-04-01"] *= 3.0  # gelecek bozulur
            out[k] = s
        return out

    changed = get_source("macro", fetcher=shifted).to_panel(["A"], dates)
    cols = [c for c in base.columns if c.startswith("macro__")]
    early = base["date"] < pd.Timestamp("2024-04-01")
    pd.testing.assert_frame_equal(base.loc[early, cols].reset_index(drop=True), changed.loc[changed["date"] < pd.Timestamp("2024-04-01"), cols].reset_index(drop=True))
    _ = rng


# ---------------------------------------------------------------- kapsam raporu
def test_coverage_report_counts_symbols_and_missing_rate(tmp_cache):
    store.write_frame("um_1d", "AAAUSDT", _klines(150, seed=1))
    store.write_frame("um_1d", "BBBUSDT", _klines(60, start="2024-03-01", seed=2))
    dates = pd.date_range("2024-04-01", periods=40)
    src = get_source("ohlcv_core")
    entry = coverage.source_coverage(src, src.to_panel(["AAAUSDT", "BBBUSDT", "CCCUSDT"], dates), ["AAAUSDT", "BBBUSDT", "CCCUSDT"], dates)
    assert entry["symbols_with_data"] == 2 and entry["symbol_coverage"] == pytest.approx(2 / 3)
    assert 0.38 < entry["decision_rows_missing"] < 0.42  # CCC hiç yok (40/120) + BBB 29 Nisan sonrası, bayatlık toleransını aşan ~8 gün
    md = coverage.coverage_markdown([entry])
    assert "ohlcv_core" in md and "67%" in md
