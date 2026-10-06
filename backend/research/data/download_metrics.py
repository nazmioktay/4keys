"""data.binance.vision futures 'metrics' dosyaları (açık pozisyon, top-trader / global long-short, taker L/S hacim oranı).

Yalnızca GÜNLÜK dosya vardır (aylık yok), 5 dakikalık gözlemler; sembol başına başlangıç tarihi değişir.
Kullanım:
    python -m research.data.download_metrics --coverage                 # tüm semboller için S3 listelemesiyle kapsam (indirmez)
    python -m research.data.download_metrics --symbols BTCUSDT ETHUSDT  # seçili semboller (günlük özet parquet: um_metrics_1d)
    python -m research.data.download_metrics --top 20                   # hacme göre ilk-N (BTC/ETH her zaman dahil)
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import store
from .download import BASE, Http, _read_zip_csv, list_s3, list_symbol_dirs

_RAW_COLS = ["create_time", "symbol", "oi", "oi_value", "toptrader_ls_acc", "toptrader_ls_pos", "global_ls", "taker_ls_vol"]
_VALUE_COLS = _RAW_COLS[2:]
_DAY_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.zip$")


def parse_metrics_zip(content: bytes) -> pd.DataFrame:
    raw = _read_zip_csv(content)
    raw = raw.iloc[:, : len(_RAW_COLS)]
    raw.columns = _RAW_COLS[: raw.shape[1]]
    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(raw["create_time"])))
    for c in _VALUE_COLS:
        out[c] = pd.to_numeric(raw[c], errors="coerce").to_numpy() if c in raw else float("nan")
    out.index.name = "time"
    return out


def daily_aggregate(df5: pd.DataFrame) -> pd.DataFrame:
    """5 dk gözlemler -> günlük: her kolon için gün SONU (last) ve ortalama (mean). İndeks = gün (00:00)."""
    day = df5.index.floor("D")
    last = df5.groupby(day).last().add_suffix("_last")
    mean = df5.groupby(day).mean().add_suffix("_mean")
    out = pd.concat([last, mean], axis=1)
    out.index.name = "date"
    return out


def list_metrics_files(http: Http, symbol: str) -> list[str]:
    prefix = f"data/futures/um/daily/metrics/{symbol}/"
    return sorted(k for kind, k in list_s3(http, prefix) if kind == "key" and k.endswith(".zip"))


def metrics_coverage(http: Http, symbols: list[str], workers: int = 16) -> pd.DataFrame:
    """Sembol başına ilk/son gün ve dosya sayısı (yalnızca S3 listelemesi — dosya indirmez)."""

    def one(sym: str) -> dict:
        days = [m.group(1) for k in list_metrics_files(http, sym) if (m := _DAY_RE.search(k))]
        return {"symbol": sym, "first": min(days) if days else None, "last": max(days) if days else None, "n_files": len(days)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, symbols))
    return pd.DataFrame(rows).set_index("symbol").sort_index()


def download_metrics(http: Http, symbol: str, since: pd.Timestamp | None = None) -> pd.DataFrame | None:
    keys = list_metrics_files(http, symbol)
    if since is not None:
        floor = since.strftime("%Y-%m-%d")
        keys = [k for k in keys if (m := _DAY_RE.search(k)) and m.group(1) >= floor]
    frames = []
    for k in keys:
        content = http.get(f"{BASE}/{k}")
        if content:
            try:
                frames.append(daily_aggregate(parse_metrics_zip(content)))
            except Exception:  # noqa: BLE001 - bozuk dosya: eksik gün olarak kalır (kalite raporu)
                continue
    if not frames:
        return None
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


def sync_metrics(http: Http, symbol: str, update: bool = False) -> dict:
    existing = store.read_frame("um_metrics_1d", symbol)
    if existing is not None and not update:
        return {"symbol": symbol, "status": "cached", "rows": len(existing)}
    since = existing.index.max() - pd.Timedelta(days=2) if (existing is not None and len(existing)) else None
    new = download_metrics(http, symbol, since)
    frames = [f for f in (existing, new) if f is not None and len(f)]
    if not frames:
        return {"symbol": symbol, "status": "empty", "rows": 0}
    merged = pd.concat(frames)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    store.write_frame("um_metrics_1d", symbol, merged)
    return {"symbol": symbol, "status": "ok", "rows": len(merged), "first": str(merged.index.min()), "last": str(merged.index.max())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Binance futures 'metrics' arşivi")
    ap.add_argument("--coverage", action="store_true", help="Tüm USDT-M semboller için kapsam (indirmez)")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--top", type=int, default=0, help="Hacme göre ilk-N (um_1d önbelleğinden)")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    http = Http()
    if args.coverage:
        syms = [s for s in list_symbol_dirs(http, "futures/um") if s.endswith("USDT") and "_" not in s]
        table = metrics_coverage(http, syms, args.workers)
        out = store.cache_dir() / "metrics_coverage.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out)
        has = table[table["n_files"] > 0]
        print(f"{len(table)} sembol, metrics dosyası olan: {len(has)}; ilk gün aralığı {has['first'].min()} .. {has['first'].max()}; -> {out}")
        return 0
    symbols = list(args.symbols or [])
    if args.top:
        from .universe import universe_at

        qv = store.load_panel("quote_volume")
        symbols = sorted(set(symbols) | {"BTCUSDT", "ETHUSDT"} | set(universe_at(qv.index.max(), args.top, qv)))
    if not symbols:
        ap.error("--coverage, --symbols veya --top gerekli")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda s: sync_metrics(http, s, args.update), symbols))
    bad = [r for r in results if r["status"] == "empty"]
    print(f"{len(results)} sembol işlendi, boş: {len(bad)} {[r['symbol'] for r in bad][:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
