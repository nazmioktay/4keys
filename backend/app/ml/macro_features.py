"""Makro/piyasa bağlamı verilerini (bkz. `app.macro`) OHLCV tabanlı ML
özellik çerçevesine zaman bazlı en-yakın-geçmiş (as-of, backward)
eşleştirmeyle ekler.

`macro_snapshots` tablosu periyodik (varsayılan 6 saatte bir) anlık
görüntüler tutar — OHLCV mumlarıyla birebir aynı zaman damgasına sahip
değildir. Bu yüzden her bar için "o ana kadar bilinen EN SON makro
değeri" kullanılır (`pd.merge_asof(..., direction="backward")`) — bu,
geleceğe bakma (look-ahead bias) YARATMAZ, çünkü yalnızca o bar anında
zaten bilinen/geçmiş bir değeri taşır.

Not: makro veri toplama yakın zamanda başladığı için geçmiş DB'de henüz
kısa bir tarihçe var; bu modül DB boşsa veya erişilemezse tüm makro
kolonlarını NaN bırakır (eğitim akışını bozmaz, sadece dropna ile o
satırlar elenir) — makro geçmişi biriktikçe kullanılabilir satır sayısı
otomatik olarak artar.
"""

import logging

import pandas as pd

from .features import MACRO_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

_RAW_TO_NORM = {
    "total_market_cap": "macro_total_market_cap_norm",
    "btc_dominance": "macro_btc_dominance_norm",
    "funding_rate_btc": "macro_funding_rate_btc",
    "vix": "macro_vix_norm",
    "gold_price": "macro_gold_norm",
    "sp500": "macro_sp500_norm",
    "nasdaq": "macro_nasdaq_norm",
    "nikkei": "macro_nikkei_norm",
    "dax": "macro_dax_norm",
    "fed_funds_rate": "macro_fed_funds_rate_norm",
    "ecb_deposit_rate": "macro_ecb_deposit_rate_norm",
}


def load_macro_history() -> pd.DataFrame:
    """DB'deki tüm makro geçmişini zaman sıralı DataFrame olarak döner.
    DB kapalı/erişilemezse (ör. testler, henüz DB kurulmamış ortamlar)
    boş DataFrame döner."""
    try:
        from app.db.repository import get_macro_snapshots  # local import: döngüsel bağımlılığı önler

        history = get_macro_snapshots(limit=200_000)
    except Exception:  # noqa: BLE001 - makro geçmişi opsiyoneldir, ana akışı bozmamalı
        logger.exception("makro geçmişi yüklenemedi")
        return pd.DataFrame()

    if history.empty or "time" not in history.columns:
        return pd.DataFrame()

    history = history.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    history["time"] = pd.to_datetime(history["time"], utc=True)
    return history


# Genişleyen (expanding) z-score'un anlamlı olması için gereken en az
# anlık görüntü sayısı — bundan önce değer NaN (modelde 0.0, nötr) kalır.
_MIN_NORMALIZE_PERIODS = 20


def _expanding_zscore(series: pd.Series) -> pd.Series:
    """Her nokta YALNIZCA kendisine kadarki geçmişin ortalama/std'siyle
    normalize edilir. Önceden tüm geçmişin ortalama/std'si kullanılıyordu —
    eski bir barın değeri, o an henüz bilinmeyen gelecekteki değerlere göre
    ölçekleniyordu (eğitim ve backtest için geleceğe bakma)."""
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.expanding(min_periods=_MIN_NORMALIZE_PERIODS).mean()
    std = numeric.expanding(min_periods=_MIN_NORMALIZE_PERIODS).std()
    z = (numeric - mean) / std.where(std > 0)
    return z.clip(-5, 5)


def normalized_macro_history(history: pd.DataFrame) -> pd.DataFrame:
    """Ham makro anlık görüntülerini ("time" + ham kolonlar) model
    özelliklerine dönüştürür: "time" + `MACRO_FEATURE_COLUMNS`. Eğitim,
    backtest ve canlı AYNI dönüşümü kullanır — canlıdaki değer bu tablonun
    son satırıdır."""
    result = pd.DataFrame({"time": history["time"]}) if "time" in history.columns else pd.DataFrame(index=history.index)
    for raw_col, norm_col in _RAW_TO_NORM.items():
        if raw_col not in history.columns:
            result[norm_col] = float("nan")
        elif raw_col == "funding_rate_btc":
            result[norm_col] = pd.to_numeric(history[raw_col], errors="coerce").clip(-0.01, 0.01) * 100
        else:
            result[norm_col] = _expanding_zscore(history[raw_col])
    return result


def latest_macro_feature_row() -> dict[str, float]:
    """Canlı (tekil bar) tahmin için en son bilinen makro değerleri,
    eğitimdekiyle AYNI dönüşümle (`normalized_macro_history`'nin son satırı).
    Makro geçmişi yoksa tüm değerler 0.0 (nötr) döner."""
    history = load_macro_history()
    result = {col: 0.0 for col in MACRO_FEATURE_COLUMNS}
    if history.empty:
        return result

    latest = normalized_macro_history(history).iloc[-1]
    for col in MACRO_FEATURE_COLUMNS:
        if col in latest.index and not pd.isna(latest[col]):
            result[col] = float(latest[col])
    return result


def merge_macro_features(frame: pd.DataFrame, macro_history: pd.DataFrame) -> pd.DataFrame:
    """`frame`'e ("timestamp" kolonu zorunlu) makro özellikleri ekler.
    `macro_history` boşsa tüm makro kolonları NaN olarak eklenir."""
    result = frame.copy()
    for col in MACRO_FEATURE_COLUMNS:
        result[col] = float("nan")

    if macro_history.empty or "timestamp" not in frame.columns:
        return result

    left = pd.DataFrame({"timestamp": pd.to_datetime(frame["timestamp"], unit="ns", utc=True)})
    left["_order"] = range(len(left))
    left = left.sort_values("timestamp")

    right = normalized_macro_history(macro_history).rename(columns={"time": "timestamp"})
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True).astype("datetime64[ns, UTC]")

    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    merged = merged.sort_values("_order").reset_index(drop=True)

    for col in MACRO_FEATURE_COLUMNS:
        if col in merged.columns:
            result[col] = merged[col].to_numpy()

    return result
