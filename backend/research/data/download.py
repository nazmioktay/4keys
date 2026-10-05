"""data.binance.vision toplu arşivinden (delist semboller DAHİL) parquet önbelleğine indirme.

Kullanım:
    python -m research.data.download --symbols BTCUSDT ETHUSDT          # hızlı (smoke için)
    python -m research.data.download --all --hourly-top 60              # tam evren
    python -m research.data.download --all --update                     # yalnızca yeni ay/gün + REST kuyruğu

Tam aylar aylık zip'ten, devam eden ay günlük zip'ten, kalan kuyruk (arşiv ~1 gün gecikmeli)
REST'ten (yalnızca halen listeli semboller) tamamlanır. Eksik/sıçrama DÜZELTİLMEZ (bkz. quality.py).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Iterator
from xml.etree import ElementTree

import pandas as pd
import requests

from . import store

S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BASE = "https://data.binance.vision"
FAPI = "https://fapi.binance.com"
SPOT_API = "https://api.binance.com"
_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


class Http:
    """Yeniden denemeli ince HTTP sarmalayıcı; 404 -> None."""

    def __init__(self, retries: int = 4, timeout: float = 60.0) -> None:
        self.session = requests.Session()
        self.retries = retries
        self.timeout = timeout

    def get(self, url: str, params: dict | None = None) -> bytes | None:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 404:
                    return None
                if r.status_code in (418, 429) or r.status_code >= 500:
                    time.sleep(2 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r.content
            except requests.RequestException as exc:
                last = exc
                time.sleep(1 + attempt)
        if last:
            raise last
        return None

    def get_json(self, url: str, params: dict | None = None):
        raw = self.get(url, params)
        return None if raw is None else json.loads(raw)


# ------------------------------------------------------------------ S3 listeleme
def parse_listing(xml_bytes: bytes) -> tuple[list[str], list[str], str | None]:
    """(alt önekler, anahtarlar, sonraki marker | None) — S3 ListBucketResult."""
    root = ElementTree.fromstring(xml_bytes)
    prefixes = [e.text for e in root.findall("s3:CommonPrefixes/s3:Prefix", _NS)]
    keys = [e.text for e in root.findall("s3:Contents/s3:Key", _NS)]
    truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=_NS) or "false").lower() == "true"
    next_marker = root.findtext("s3:NextMarker", namespaces=_NS)
    if truncated and not next_marker:
        next_marker = (keys[-1] if keys else (prefixes[-1] if prefixes else None))
    return prefixes, keys, (next_marker if truncated else None)


def list_s3(http: Http, prefix: str, delimiter: str | None = None) -> Iterator[tuple[str, str]]:
    """('prefix'|'key', değer) üretir; sayfalamayı yönetir."""
    marker = None
    while True:
        params = {"prefix": prefix, "max-keys": 1000}
        if delimiter:
            params["delimiter"] = delimiter
        if marker:
            params["marker"] = marker
        raw = http.get(S3_LIST, params)
        if raw is None:
            return
        prefixes, keys, marker = parse_listing(raw)
        for p in prefixes:
            yield "prefix", p
        for k in keys:
            yield "key", k
        if not marker:
            return


def list_symbol_dirs(http: Http, market_path: str) -> list[str]:
    """`data/<market_path>/monthly/klines/` altındaki tüm sembol dizinleri (delist dahil)."""
    base = f"data/{market_path}/monthly/klines/"
    return sorted(p[len(base):].strip("/") for kind, p in list_s3(http, base, "/") if kind == "prefix")


# ------------------------------------------------------------------ ayrıştırma
def _to_naive_utc(series: pd.Series) -> pd.DatetimeIndex:
    vals = pd.to_numeric(series, errors="coerce")
    unit = "us" if vals.dropna().abs().max() > 1e14 else "ms"  # 2025 sonrası spot dosyaları mikrosaniye
    return pd.DatetimeIndex(pd.to_datetime(vals, unit=unit, utc=True).dt.tz_convert(None))


def _read_zip_csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        raw = pd.read_csv(z.open(name), header=None, dtype=str)
    if len(raw) and not re.fullmatch(r"-?\d+(\.\d+)?", str(raw.iloc[0, 0]).strip()):
        raw = raw.iloc[1:]  # başlık satırı
    return raw.reset_index(drop=True)


def parse_kline_zip(content: bytes) -> pd.DataFrame:
    raw = _read_zip_csv(content)
    raw.columns = _KLINE_COLS[: raw.shape[1]]
    out = pd.DataFrame(index=_to_naive_utc(raw["open_time"]))
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"):
        out[col] = pd.to_numeric(raw[col], errors="coerce").to_numpy()
    out.index.name = "open_time"
    return out


def parse_funding_zip(content: bytes) -> pd.DataFrame:
    raw = _read_zip_csv(content)
    time_col = _to_naive_utc(raw.iloc[:, 0])
    interval = pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy() if raw.shape[1] >= 3 else float("nan")
    out = pd.DataFrame({"rate": pd.to_numeric(raw.iloc[:, -1], errors="coerce").to_numpy(), "interval_hours": interval}, index=time_col)
    out.index.name = "time"
    return out


# ------------------------------------------------------------------ indirme (arşiv)
def _month_of(key: str) -> str | None:
    m = re.search(r"-(\d{4}-\d{2})\.zip$", key)
    return m.group(1) if m else None


def _day_of(key: str) -> str | None:
    m = re.search(r"-(\d{4}-\d{2}-\d{2})\.zip$", key)
    return m.group(1) if m else None


def _fetch_zip_frames(http: Http, keys: list[str], parser) -> list[pd.DataFrame]:
    frames = []
    for key in keys:
        content = http.get(f"{BASE}/{key}")
        if content:
            try:
                frames.append(parser(content))
            except Exception:  # noqa: BLE001 - bozuk zip: atlanır, kalite raporunda eksik gün olarak görünür
                continue
    return frames


def download_klines(http: Http, market_path: str, symbol: str, interval: str, since: pd.Timestamp | None = None) -> pd.DataFrame | None:
    """Arşivden klines: tam aylar aylık zip'ten, son yayımlanmış aydan sonraki günler günlük zip'ten.
    `since` verilirse yalnızca o tarihin ayından itibaren dosyalar indirilir (artımlı güncelleme)."""
    mprefix = f"data/{market_path}/monthly/klines/{symbol}/{interval}/"
    dprefix = f"data/{market_path}/daily/klines/{symbol}/{interval}/"
    monthly = sorted(k for kind, k in list_s3(http, mprefix) if kind == "key" and k.endswith(".zip"))
    if since is not None:
        floor = since.strftime("%Y-%m")
        monthly = [k for k in monthly if (_month_of(k) or "") >= floor]
    last_month = max((_month_of(k) for k in monthly), default=None)
    daily = sorted(k for kind, k in list_s3(http, dprefix) if kind == "key" and k.endswith(".zip"))
    daily_keys = []
    for k in daily:
        d = _day_of(k)
        if not d:
            continue
        if last_month is not None and d[:7] <= last_month:
            continue  # o ay zaten aylık dosyada
        if since is not None and pd.Timestamp(d) < since.normalize():
            continue
        daily_keys.append(k)
    frames = _fetch_zip_frames(http, monthly + daily_keys, parse_kline_zip)
    if not frames:
        return None
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


def download_funding(http: Http, symbol: str, since: pd.Timestamp | None = None) -> pd.DataFrame | None:
    prefix = f"data/futures/um/monthly/fundingRate/{symbol}/"
    keys = sorted(k for kind, k in list_s3(http, prefix) if kind == "key" and k.endswith(".zip"))
    if since is not None:
        floor = since.strftime("%Y-%m")
        keys = [k for k in keys if (_month_of(k) or "") >= floor]
    frames = _fetch_zip_frames(http, keys, parse_funding_zip)
    if not frames:
        return None
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


# ------------------------------------------------------------------ REST kuyruğu (yalnızca listeli semboller)
def rest_klines_tail(http: Http, symbol: str, interval: str, start: pd.Timestamp, spot: bool = False) -> pd.DataFrame | None:
    """`start`tan itibaren KAPANMIŞ barlar (devam eden bar atılır)."""
    url = f"{SPOT_API}/api/v3/klines" if spot else f"{FAPI}/fapi/v1/klines"
    now = pd.Timestamp.now(tz="UTC").tz_convert(None)
    delta = pd.Timedelta(interval.replace("1d", "1D").replace("1h", "1h"))
    rows: list = []
    cursor = int(start.tz_localize("UTC").timestamp() * 1000)
    while True:
        data = http.get_json(url, {"symbol": symbol, "interval": interval, "startTime": cursor, "limit": 1000})
        if not data:
            break
        rows.extend(data)
        if len(data) < 1000:
            break
        cursor = int(data[-1][0]) + 1
    if not rows:
        return None
    raw = pd.DataFrame(rows).iloc[:, :11]
    raw.columns = _KLINE_COLS[:11]
    out = pd.DataFrame(index=_to_naive_utc(raw["open_time"]))
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"):
        out[col] = pd.to_numeric(raw[col], errors="coerce").to_numpy()
    out.index.name = "open_time"
    return out[out.index + delta <= now]  # yalnızca kapanmış barlar


def rest_funding_tail(http: Http, symbol: str, start: pd.Timestamp) -> pd.DataFrame | None:
    rows: list = []
    cursor = int(start.tz_localize("UTC").timestamp() * 1000)
    while True:
        data = http.get_json(f"{FAPI}/fapi/v1/fundingRate", {"symbol": symbol, "startTime": cursor, "limit": 1000})
        if not data:
            break
        rows.extend(data)
        if len(data) < 1000:
            break
        cursor = int(data[-1]["fundingTime"]) + 1
    if not rows:
        return None
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {"rate": pd.to_numeric(df["fundingRate"]).to_numpy(), "interval_hours": float("nan")},
        index=pd.DatetimeIndex(pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True).dt.tz_convert(None)),
    )
    # Binance fundingTime milisaniye gürültüsü taşır (ör. 00:00:00.003) -> saniyeye yuvarla
    out.index = out.index.round("s")
    out.index.name = "time"
    return out


# ------------------------------------------------------------------ orkestrasyon
def _merge(old: pd.DataFrame | None, *new: pd.DataFrame | None) -> pd.DataFrame | None:
    frames = [f for f in (old, *new) if f is not None and len(f)]
    if not frames:
        return None
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


def _since_for(existing: pd.DataFrame | None, update: bool) -> pd.Timestamp | None:
    if existing is None or not update or existing.empty:
        return None
    return existing.index.max().normalize() - pd.Timedelta(days=2)  # küçük bindirme


def sync_klines(http: Http, kind: str, market_path: str, symbol: str, interval: str, listed: bool, update: bool, spot: bool = False) -> dict:
    existing = store.read_frame(kind, symbol)
    if existing is not None and not update:
        return {"symbol": symbol, "kind": kind, "status": "cached", "rows": len(existing)}
    since = _since_for(existing, update)
    archive = download_klines(http, market_path, symbol, interval, since)
    merged = _merge(existing, archive)
    if listed and merged is not None:
        tail_start = merged.index.max() + pd.Timedelta(interval.replace("1d", "1D"))
        if tail_start < pd.Timestamp.now(tz="UTC").tz_convert(None):
            merged = _merge(merged, rest_klines_tail(http, symbol, interval, tail_start, spot))
    if merged is None:
        return {"symbol": symbol, "kind": kind, "status": "empty", "rows": 0}
    store.write_frame(kind, symbol, merged)
    return {"symbol": symbol, "kind": kind, "status": "ok", "rows": len(merged), "first": str(merged.index.min()), "last": str(merged.index.max())}


def sync_funding(http: Http, symbol: str, listed: bool, update: bool) -> dict:
    existing = store.read_frame("um_funding", symbol)
    if existing is not None and not update:
        return {"symbol": symbol, "kind": "um_funding", "status": "cached", "rows": len(existing)}
    since = _since_for(existing, update)
    merged = _merge(existing, download_funding(http, symbol, since))
    if listed and merged is not None:
        merged = _merge(merged, rest_funding_tail(http, symbol, merged.index.max() + pd.Timedelta(seconds=1)))
    if merged is None:
        return {"symbol": symbol, "kind": "um_funding", "status": "empty", "rows": 0}
    store.write_frame("um_funding", symbol, merged)
    return {"symbol": symbol, "kind": "um_funding", "status": "ok", "rows": len(merged), "first": str(merged.index.min()), "last": str(merged.index.max())}


def listed_perpetuals(http: Http) -> set[str]:
    info = http.get_json(f"{FAPI}/fapi/v1/exchangeInfo") or {}
    return {s["symbol"] for s in info.get("symbols", []) if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING"}


def _run_parallel(tasks: list, workers: int, label: str) -> list[dict]:
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *args): str(args[1:4]) for fn, *args in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - tek sembol hatası tümünü durdurmaz; raporlanır
                results.append({"symbol": futures[fut], "status": "error", "error": str(exc)[:200]})
            if i % 25 == 0 or i == len(futures):
                print(f"[{label}] {i}/{len(futures)}", flush=True)
    return results


def run(args: argparse.Namespace, http: Http | None = None) -> dict:
    http = http or Http()
    started = datetime.now(timezone.utc)
    all_um = [s for s in list_symbol_dirs(http, "futures/um") if s.endswith("USDT") and "_" not in s]
    delivery_um = [s for s in list_symbol_dirs(http, "futures/um") if re.fullmatch(r"(BTC|ETH)USDT_\d{6}", s)]
    delivery_cm = [s for s in list_symbol_dirs(http, "futures/cm") if re.fullmatch(r"(BTC|ETH)USD_\d{6}", s)]
    symbols = sorted(set(args.symbols)) if args.symbols else all_um
    listed = listed_perpetuals(http)
    meta: dict = {"started": started.isoformat(), "archive_perp_symbols": len(all_um), "selected": len(symbols)}

    daily_tasks = [(sync_klines, http, "um_1d", "futures/um", s, "1d", s in listed, args.update) for s in symbols]
    daily_tasks += [(sync_funding, http, s, s in listed, args.update) for s in symbols]
    daily_tasks += [(sync_klines, http, "spot_1d", "spot", s, "1d", s in listed, args.update, True) for s in symbols]
    if not args.no_delivery:
        daily_tasks += [(sync_klines, http, "delivery_um_1d", "futures/um", s, "1d", False, args.update) for s in delivery_um]
        daily_tasks += [(sync_klines, http, "delivery_cm_1d", "futures/cm", s, "1d", False, args.update) for s in delivery_cm]
    results = _run_parallel(daily_tasks, args.workers, "günlük/funding/spot/vadeli")

    hourly = []
    if args.hourly_top and not args.symbols:
        from .universe import ever_in_top_n

        qv = store.load_panel("quote_volume", symbols, "um_1d", allow_final_test=False)
        hourly = ever_in_top_n(qv, args.hourly_top) if not qv.empty else []
    elif args.symbols and args.hourly:
        hourly = symbols
    if hourly:
        hourly_tasks = [(sync_klines, http, "um_1h", "futures/um", s, "1h", s in listed, args.update) for s in hourly]
        results += _run_parallel(hourly_tasks, max(2, args.workers // 2), "1h")

    meta.update(
        {
            "finished": datetime.now(timezone.utc).isoformat(),
            "delisted_in_selection": sorted(s for s in symbols if s not in listed),
            "delivery_um": delivery_um,
            "delivery_cm": delivery_cm,
            "hourly_symbols": hourly,
            "results": results,
            "snapshot_hash": store.snapshot_hash(),
        }
    )
    out = store.cache_dir() / "download_meta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Binance toplu arşivinden araştırma verisi indir")
    ap.add_argument("--symbols", nargs="*", help="Yalnızca bu USDT-M semboller (ör. BTCUSDT ETHUSDT)")
    ap.add_argument("--all", action="store_true", help="Arşivdeki TÜM USDT-M perpetual semboller (delist dahil)")
    ap.add_argument("--hourly-top", type=int, default=0, help="Hiç ilk-N hacme girmiş semboller için 1h (varsayılan 0 = yok)")
    ap.add_argument("--hourly", action="store_true", help="--symbols ile birlikte 1h de indir")
    ap.add_argument("--update", action="store_true", help="Var olan parquet'leri artımlı güncelle")
    ap.add_argument("--no-delivery", action="store_true", help="BTC/ETH vadeli kontratları indirme")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    if not args.all and not args.symbols:
        ap.error("--all veya --symbols gerekli")
    meta = run(args)
    bad = [r for r in meta["results"] if r.get("status") == "error"]
    print(f"bitti: {len(meta['results'])} görev, {len(bad)} hata, snapshot={meta['snapshot_hash']}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
