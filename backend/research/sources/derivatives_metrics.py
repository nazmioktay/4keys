"""derivatives_metrics: data.binance.vision futures 'metrics' (açık pozisyon, top-trader/global long-short oranı,
taker L/S hacim oranı). Günlük özetler `um_metrics_1d` parquet'inden (bkz. data/download_metrics.py).

Kapsam sembol bazında DEĞİŞİR (BTC 2020-09'dan; çoğu sembol daha geç) — kapsam raporuna bakın. Eksik günler NaN kalır.
`available_at = date + 1 gün` (gün sonu gözlemi canlıda API'den anlık okunabilir; arşiv gecikmesi canlı kullanımı etkilemez)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import store
from ..guard import cut_final_test
from .base import Source
from .ohlcv_core import stack_features
from .registry import register_source


@register_source
class DerivativesMetrics(Source):
    name = "derivatives_metrics"
    version = "1"
    max_staleness = pd.Timedelta(days=2)

    @property
    def history_start(self):
        btc = store.read_frame("um_metrics_1d", "BTCUSDT")
        return None if btc is None or btc.empty else btc.index.min()

    def fetch(self, start, end, symbols=None):
        symbols = symbols if symbols is not None else store.list_symbols("um_metrics_1d")
        out = {}
        for s in symbols:
            df = store.read_frame("um_metrics_1d", s)
            if df is not None and len(df):
                df = cut_final_test(df)  # nihai pencere (>= 2025-10-01) varsayılan KESİLİR (doğrudan fetch() çağrısı da)
                out[s] = df[(df.index >= start) & (df.index <= end)]
        return out

    def to_panel(self, universe, dates):
        raw = self.fetch(dates.min() - pd.Timedelta(days=10), dates.max(), universe)
        if not raw:
            return pd.DataFrame(columns=["date", "symbol", "available_at"])
        cols = ["oi_value_last", "toptrader_ls_acc_last", "toptrader_ls_pos_last", "global_ls_last", "taker_ls_vol_last", "taker_ls_vol_mean"]
        wide = {c: pd.DataFrame({s: df[c] for s, df in raw.items() if c in df}) for c in cols}
        oi = wide["oi_value_last"]
        log_oi = np.log(oi.where(oi > 0))
        feats = {
            "oi_value_log": log_oi,
            "oi_chg_1d": log_oi.diff(),
            "oi_chg_7d": log_oi.diff(7),
            "toptrader_ls_acc": wide["toptrader_ls_acc_last"],
            "toptrader_ls_pos": wide["toptrader_ls_pos_last"],
            "global_ls": wide["global_ls_last"],
            "taker_ls_vol": wide["taker_ls_vol_last"],
            "taker_ls_vol_mean": wide["taker_ls_vol_mean"],
        }
        panel = stack_features(feats)
        panel = panel[panel["date"].isin(dates)].dropna(subset=list(feats), how="all")
        panel["available_at"] = panel["date"] + pd.Timedelta(days=1)
        return self.prefix(panel).reset_index(drop=True)
