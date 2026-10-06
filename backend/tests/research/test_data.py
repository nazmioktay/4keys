import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from research import config, guard
from research.data import download, limits, quality, store, universe


# ---------------------------------------------------------------- yardımcılar
def _zip(csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data.csv", csv_text)
    return buf.getvalue()


def _kline_csv(days, start="2021-01-01", header=False, micro=False, price=100.0):
    rows = []
    if header:
        rows.append("open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore")
    t0 = pd.Timestamp(start)
    for i in range(days):
        ts = int((t0 + pd.Timedelta(days=i)).timestamp() * (1_000_000 if micro else 1000))
        p = price + i
        rows.append(f"{ts},{p},{p+1},{p-1},{p+0.5},10,{ts+86399999},{p*10},5,4,{p*4},0")
    return "\n".join(rows)


def _listing_xml(prefixes=(), keys=(), truncated=False, next_marker=None):
    ns = "http://s3.amazonaws.com/doc/2006-03-01/"
    parts = [f'<?xml version="1.0"?><ListBucketResult xmlns="{ns}">', f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>"]
    if next_marker:
        parts.append(f"<NextMarker>{next_marker}</NextMarker>")
    parts += [f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in prefixes]
    parts += [f"<Contents><Key>{k}</Key></Contents>" for k in keys]
    parts.append("</ListBucketResult>")
    return "".join(parts).encode()


class FakeHttp:
    """S3 listeleme + zip + REST için sahte HTTP; çağrıları kaydeder."""

    def __init__(self, listings: dict, files: dict):
        self.listings = listings  # (prefix, marker) -> xml bytes
        self.files = files  # url -> bytes
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        if url == download.S3_LIST:
            return self.listings.get((params["prefix"], params.get("marker")))
        return self.files.get(url)

    def get_json(self, url, params=None):
        raw = self.get(url, params)
        return None if raw is None else json.loads(raw)


# ---------------------------------------------------------------- ayrıştırma
def test_parse_listing_returns_next_marker_only_when_truncated():
    p, k, m = download.parse_listing(_listing_xml(prefixes=["a/", "b/"], truncated=True, next_marker="b/"))
    assert (p, k, m) == (["a/", "b/"], [], "b/")
    _, keys, marker = download.parse_listing(_listing_xml(keys=["x.zip"], truncated=False))
    assert keys == ["x.zip"] and marker is None


def test_list_s3_follows_pagination():
    http = FakeHttp(
        {
            ("data/p/", None): _listing_xml(keys=["data/p/1.zip"], truncated=True, next_marker="data/p/1.zip"),
            ("data/p/", "data/p/1.zip"): _listing_xml(keys=["data/p/2.zip"]),
        },
        {},
    )
    assert [v for _, v in download.list_s3(http, "data/p/")] == ["data/p/1.zip", "data/p/2.zip"]


def test_parse_kline_zip_handles_header_and_millisecond_timestamps():
    df = download.parse_kline_zip(_zip(_kline_csv(3, header=True)))
    assert list(df.index) == list(pd.date_range("2021-01-01", periods=3, freq="D"))
    assert df["close"].iloc[0] == 100.5 and df["quote_volume"].iloc[0] == 1000.0
    assert df.index.name == "open_time" and df.index.tz is None


def test_parse_kline_zip_handles_microsecond_timestamps_without_header():
    df = download.parse_kline_zip(_zip(_kline_csv(2, micro=True)))
    assert df.index[0] == pd.Timestamp("2021-01-01")


def test_timestamp_unit_is_detected_per_value_and_impossible_dates_are_rejected():
    sec = int(pd.Timestamp("2023-01-02").timestamp())
    ms = sec * 1000
    us = ms * 1000
    idx = download._to_naive_utc(pd.Series([str(sec), str(ms), str(us)]))
    assert list(idx) == [pd.Timestamp("2023-01-02")] * 3  # saniye / ms / µs karışık dosya (KLAYUSDT spot)
    with pytest.raises(ValueError):
        download._to_naive_utc(pd.Series(["1000"]))  # 1970 -> olanaksız


def test_parse_funding_zip_reads_rate_and_interval():
    csv = "calc_time,funding_interval_hours,last_funding_rate\n1609459200000,8,0.0001\n1609488000000,8,-0.00005"
    df = download.parse_funding_zip(_zip(csv))
    assert list(df["rate"]) == [0.0001, -0.00005]
    assert df["interval_hours"].tolist() == [8, 8]
    assert df.index[0] == pd.Timestamp("2021-01-01 00:00")


def test_download_klines_uses_monthly_then_only_newer_daily_files():
    mprefix = "data/futures/um/monthly/klines/BTCUSDT/1d/"
    dprefix = "data/futures/um/daily/klines/BTCUSDT/1d/"
    m1, m2 = mprefix + "BTCUSDT-1d-2021-01.zip", mprefix + "BTCUSDT-1d-2021-02.zip"
    d_old = dprefix + "BTCUSDT-1d-2021-02-10.zip"  # aylık dosyada zaten var -> indirilmez
    d_new = dprefix + "BTCUSDT-1d-2021-03-02.zip"
    http = FakeHttp(
        {(mprefix, None): _listing_xml(keys=[m1, m2]), (dprefix, None): _listing_xml(keys=[d_old, d_new])},
        {
            f"{download.BASE}/{m1}": _zip(_kline_csv(31, "2021-01-01")),
            f"{download.BASE}/{m2}": _zip(_kline_csv(28, "2021-02-01")),
            f"{download.BASE}/{d_new}": _zip(_kline_csv(1, "2021-03-02")),
        },
    )
    df = download.download_klines(http, "futures/um", "BTCUSDT", "1d")
    assert len(df) == 31 + 28 + 1
    assert df.index.max() == pd.Timestamp("2021-03-02")
    assert not any(d_old in c[0] for c in http.calls)  # tekrar indirilmedi


# ---------------------------------------------------------------- önbellek + nihai pencere kilidi
@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    return tmp_path


def _frame(start, days):
    idx = pd.date_range(start, periods=days, freq="D", name="open_time")
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "quote_volume": 1.0,
                         "trades": 1.0, "taker_buy_base": 0.5, "taker_buy_quote": 0.5}, index=idx)


def test_store_roundtrip_and_snapshot_hash_changes_with_content(tmp_cache):
    store.write_frame("um_1d", "BTCUSDT", _frame("2025-01-01", 10))
    h1 = store.snapshot_hash()
    assert store.read_frame("um_1d", "BTCUSDT").shape[0] == 10
    store.write_frame("um_1d", "BTCUSDT", _frame("2025-01-01", 11))
    assert store.snapshot_hash() != h1
    assert store.list_symbols("um_1d") == ["BTCUSDT"]
    manifest = json.loads((tmp_cache / "cache" / "manifest.json").read_text())
    assert "um_1d/BTCUSDT.parquet" in manifest


def test_loaders_cut_final_test_window_by_default(tmp_cache):
    store.write_frame("um_1d", "BTCUSDT", _frame("2025-09-20", 30))  # 09-20 .. 10-19
    df = store.load_klines("BTCUSDT")
    assert df.index.max() == pd.Timestamp("2025-09-30")
    assert store.load_panel("close", ["BTCUSDT"]).index.max() == pd.Timestamp("2025-09-30")
    funding = pd.DataFrame({"rate": 0.0001, "interval_hours": 8.0}, index=pd.date_range("2025-09-29", periods=6, freq="8h", name="time"))
    store.write_frame("um_funding", "BTCUSDT", funding)
    assert store.load_funding("BTCUSDT").index.max() < pd.Timestamp("2025-10-01")


def test_final_test_cannot_be_opened_without_registered_experiment(tmp_cache, tmp_path):
    store.write_frame("um_1d", "BTCUSDT", _frame("2025-09-20", 30))
    with pytest.raises(guard.FinalTestError):
        store.load_klines("BTCUSDT", allow_final_test=True)  # id yok
    with pytest.raises(guard.FinalTestError):
        store.load_klines("BTCUSDT", allow_final_test=True, final_test_experiment_id="e1")  # deneyler.md'de kayıt yok


def test_final_test_opening_requires_marker_line_in_experiment_log(tmp_path):
    log = tmp_path / "deneyler.md"
    log.write_text("# günlük\nFINAL-TEST-ACILDI e1\n", encoding="utf-8")
    df = _frame("2025-09-20", 30)
    assert guard.cut_final_test(df, True, "e1", log_path=log).index.max() == pd.Timestamp("2025-10-19")
    with pytest.raises(guard.FinalTestError):
        guard.cut_final_test(df, True, "baska-id", log_path=log)
    assert guard.cut_final_test(df, False, "e1", log_path=log).index.max() == pd.Timestamp("2025-09-30")
    guard.assert_no_final_test(pd.date_range("2025-01-01", periods=5))
    with pytest.raises(guard.FinalTestError):
        guard.assert_no_final_test(pd.date_range("2025-09-28", periods=5))


# ---------------------------------------------------------------- evren (noktasal-zamanlı)
def _volume_panel(seed=0, days=600):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=days, freq="D")
    cols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "USDCUSDT", "BTCUPUSDT", "BTCDOMUSDT", "LATEUSDT", "BTCUSDT_230331"]
    base = {"BTCUSDT": 1e9, "ETHUSDT": 6e8, "SOLUSDT": 3e8, "DOGEUSDT": 1e8, "USDCUSDT": 9e9, "BTCUPUSDT": 8e8, "BTCDOMUSDT": 7e8,
            "LATEUSDT": 5e8, "BTCUSDT_230331": 2e8}
    panel = pd.DataFrame({c: base[c] * rng.uniform(0.8, 1.2, days) for c in cols}, index=idx)
    panel.loc[panel.index < "2023-01-01", "LATEUSDT"] = np.nan  # geç listelenen
    return panel


def test_universe_excludes_stable_leveraged_index_delivery_and_ranks_by_volume():
    panel = _volume_panel()
    u = universe.universe_at("2023-01-20", 10, panel)
    assert u[:4] == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
    for banned in ("USDCUSDT", "BTCUPUSDT", "BTCDOMUSDT", "BTCUSDT_230331"):
        assert banned not in u


def test_universe_requires_minimum_history():
    panel = _volume_panel()
    assert "LATEUSDT" not in universe.universe_at("2023-02-15", 10, panel)  # listelenmesinden ~45 gün sonra
    assert "LATEUSDT" in universe.universe_at("2023-05-01", 10, panel)  # >= 90 gün


def test_universe_at_date_ignores_all_data_on_or_after_the_date():
    """Gelecek bilgisi yok: date ve sonrasını rastgele bozmak/silmek sonucu DEĞİŞTİRMEZ."""
    panel = _volume_panel()
    date = pd.Timestamp("2023-03-01")
    baseline = universe.universe_at(date, 5, panel)
    rng = np.random.default_rng(99)
    corrupted = panel.copy()
    future = corrupted.index >= date
    corrupted.loc[future] = rng.uniform(1e3, 1e12, size=(int(future.sum()), corrupted.shape[1]))
    assert universe.universe_at(date, 5, corrupted) == baseline
    assert universe.universe_at(date, 5, panel[panel.index < date]) == baseline  # gelecek tamamen silinse de aynı
    # kontrol: bir gün SONRASI bozulursa sonuç gerçekten farklılaşabilir (test boş değil)
    shifted = panel.copy()
    shifted.loc[shifted.index < date, "DOGEUSDT"] *= 1e4
    assert universe.universe_at(date, 5, shifted) != baseline


def test_ever_in_top_n_collects_symbols_that_ranked_at_any_time():
    panel = _volume_panel()
    assert "LATEUSDT" in universe.ever_in_top_n(panel, 5, step_days=30)
    assert "USDCUSDT" not in universe.ever_in_top_n(panel, 5, step_days=30)


# ---------------------------------------------------------------- kalite raporu
def test_quality_flags_gaps_zero_volume_and_jumps_without_filling():
    df = _frame("2021-01-01", 60)
    df["close"] = 100.0 + np.sin(np.arange(60)) * 0.5
    df = df.drop(df.index[[10, 11, 30]])  # 3 eksik gün
    df.loc[df.index[5], "volume"] = 0.0
    df.loc[df.index[40], "close"] = 400.0  # sıçrama
    q = quality.symbol_quality(df)
    assert q["missing_days"] == 3 and q["rows"] == 57
    assert q["zero_volume_days"] == 1
    assert q["n_jumps"] >= 1
    assert len(df) == 57  # doldurulmadı
    md = quality.to_markdown(quality.quality_report({"AAAUSDT": df, "CLEANUSDT": _frame("2021-01-01", 60)}))
    assert "AAAUSDT" in md and "DOLDURULMAZ" in md
    assert "| CLEANUSDT |" not in md  # temiz sembol sorun tablosunda yok


# ---------------------------------------------------------------- limitler
def test_limits_are_cached_and_unknown_symbols_reported(tmp_cache):
    calls = []

    def fetcher(sym):
        calls.append(sym)
        if sym.startswith("DEAD"):
            raise RuntimeError("bilinmeyen")
        return {"amount_step": 0.001, "amount_min": 0.001, "price_tick": 0.1, "cost_min": 100.0}

    out = limits.get_limits(["BTCUSDT", "DEADUSDT"], fetcher=fetcher)
    assert out["BTCUSDT"]["cost_min"] == 100.0 and out["DEADUSDT"] is None
    assert calls == ["BTC/USDT:USDT", "DEAD/USDT:USDT"]
    limits.get_limits(["BTCUSDT", "DEADUSDT"], fetcher=fetcher)  # önbellekten
    assert len(calls) == 2
    min_notional, step, unknown = limits.engine_limits(out)
    assert min_notional["BTCUSDT"] == 100.0 and unknown == ["DEADUSDT"]
