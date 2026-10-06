"""forward_only kaynaklar: geçmişi OLMAYAN veriler (zorunlu tasfiye akışı, emir defteri derinliği, ayrıntılı açık pozisyon),
`app.forwardcollect` ile toplamaya başlandığı andan itibaren DB'de birikir. Koşucu, `accumulated_days() < 365` ise bu
kaynakları kullanan deneyleri REDDEDER (`research.panel.ForwardOnlyTooShortError`; KURALLAR.md §9).

Karar zamanı: gün d'nin kapanışı (d+1 00:00); `available_at = d + 1 gün`. Gün başına özet (UTC).
Tasfiye: toplayıcının çalıştığı varsayılan [ilk olay günü, son olay günü] aralığında olaysız gün = 0 tasfiye (toplayıcı
kesintisi bilinemez — bu varsayım kapsam raporunda belirtilir); aralık dışı NaN kalır."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..guard import cut_final_test
from .base import Source
from .registry import register_source


def _repo():
    from app.db import (
        repository,  # DB opsiyoneldir; yalnızca bu kaynaklar kullanılırken içe aktarılır
    )

    return repository


def _naive_day(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True)
    return t.dt.tz_convert(None).dt.floor("D")


class _ForwardSource(Source):
    forward_only = True
    max_staleness = pd.Timedelta(days=2)
    _reader = ""

    def _read(self) -> pd.DataFrame:
        return getattr(_repo(), self._reader)()

    def data_signature(self) -> str:
        df = self._read()
        return f"{len(df)}:{df['time'].max() if len(df) else ''}"  # yeni satır gelince önbellek geçersiz

    def accumulated_days(self) -> float | None:
        df = self._read()
        if df.empty:
            return None
        t = pd.to_datetime(df["time"], utc=True)
        return float((t.max() - t.min()).total_seconds() / 86400.0)

    @property
    def history_start(self):
        df = self._read()
        return None if df.empty else pd.to_datetime(df["time"], utc=True).min().tz_convert(None)

    def fetch(self, start, end, symbols=None):
        df = self._read()
        if symbols is not None and len(df):
            df = df[df["symbol"].isin(symbols)]
        if len(df):  # kilitli pencereler (etkin bölge) KESİLİR: T0 öncesi satır diff()/ısınma için bile kullanılmaz (KURALLAR §10)
            t = pd.DatetimeIndex(pd.to_datetime(df["time"], utc=True)).tz_convert(None)
            df = cut_final_test(df.set_index(t)).reset_index(drop=True)
        return df


@register_source
class Liquidations(_ForwardSource):
    name = "liquidations"
    version = "1"
    feature_names = ("liquidations__liq_long_usd", "liquidations__liq_short_usd", "liquidations__liq_net_usd", "liquidations__liq_count")
    _reader = "get_liquidation_events"

    def to_panel(self, universe, dates):
        df = self.fetch(dates.min(), dates.max(), universe)
        if df.empty:
            return pd.DataFrame(columns=["date", "symbol", "available_at"])
        df = df.assign(day=_naive_day(df["time"]))
        df["long_usd"] = np.where(df["side"] == "SELL", df["quote_value"], 0.0)  # SELL = uzun tasfiye
        df["short_usd"] = np.where(df["side"] == "BUY", df["quote_value"], 0.0)
        daily = df.groupby(["day", "symbol"]).agg(liq_long_usd=("long_usd", "sum"), liq_short_usd=("short_usd", "sum"), liq_count=("quote_value", "size")).reset_index()
        rows = []
        for sym, g in daily.groupby("symbol"):
            full = pd.date_range(g["day"].min(), g["day"].max(), freq="D")
            g = g.set_index("day").reindex(full).fillna(0.0)  # toplayıcı çalışırken olaysız gün = 0 (belgelenmiş varsayım)
            g["liq_net_usd"] = g["liq_short_usd"] - g["liq_long_usd"]
            g["symbol"] = sym
            rows.append(g.rename_axis("date").reset_index())
        panel = pd.concat(rows, ignore_index=True)
        panel = panel[panel["date"].isin(dates)]
        panel["available_at"] = panel["date"] + pd.Timedelta(days=1)
        return self.prefix(panel[["date", "symbol", "liq_long_usd", "liq_short_usd", "liq_net_usd", "liq_count", "available_at"]]).reset_index(drop=True)


@register_source
class DepthBands(_ForwardSource):
    name = "depth_bands"
    version = "1"
    feature_names = tuple(f"depth_bands__{c}" for c in ("spread_bps", "imb_10", "imb_50", "imb_100", "depth_50_log", "n_snapshots"))
    _reader = "get_depth_band_snapshots"

    def to_panel(self, universe, dates):
        df = self.fetch(dates.min(), dates.max(), universe)
        if df.empty:
            return pd.DataFrame(columns=["date", "symbol", "available_at"])
        df = df.assign(day=_naive_day(df["time"]))
        for bp in (10, 50, 100):
            b, a = df[f"bid_{bp}bp"], df[f"ask_{bp}bp"]
            df[f"imb_{bp}"] = (b - a) / (b + a).where((b + a) > 0)
        df["depth_50_log"] = np.log((df["bid_50bp"] + df["ask_50bp"]).where(lambda s: s > 0))
        g = df.groupby(["day", "symbol"]).agg(
            spread_bps=("spread_bps", "mean"), imb_10=("imb_10", "mean"), imb_50=("imb_50", "mean"), imb_100=("imb_100", "mean"),
            depth_50_log=("depth_50_log", "mean"), n_snapshots=("spread_bps", "size"),
        ).reset_index().rename(columns={"day": "date"})
        g = g[g["date"].isin(dates)]
        g["available_at"] = g["date"] + pd.Timedelta(days=1)
        return self.prefix(g).reset_index(drop=True)


@register_source
class OIDetail(_ForwardSource):
    name = "oi_detail"
    version = "1"
    feature_names = tuple(f"oi_detail__{c}" for c in ("oi_value_last", "top_ls_account", "top_ls_position", "global_ls_account", "taker_buy_sell_ratio", "n_snapshots", "oi_chg_1d"))
    _reader = "get_oi_detail_snapshots"

    def to_panel(self, universe, dates):
        df = self.fetch(dates.min(), dates.max(), universe)
        if df.empty:
            return pd.DataFrame(columns=["date", "symbol", "available_at"])
        df = df.assign(day=_naive_day(df["time"]))
        g = df.groupby(["day", "symbol"]).agg(
            oi_value_last=("open_interest_value", "last"), top_ls_account=("top_ls_account", "last"), top_ls_position=("top_ls_position", "last"),
            global_ls_account=("global_ls_account", "last"), taker_buy_sell_ratio=("taker_buy_sell_ratio", "mean"), n_snapshots=("open_interest", "size"),
        ).reset_index().rename(columns={"day": "date"})
        g["oi_chg_1d"] = g.groupby("symbol")["oi_value_last"].transform(lambda s: np.log(s.where(s > 0)).diff())
        g = g[g["date"].isin(dates)]
        g["available_at"] = g["date"] + pd.Timedelta(days=1)
        return self.prefix(g).reset_index(drop=True)
