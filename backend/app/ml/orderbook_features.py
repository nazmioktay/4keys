"""Emir defteri (order book) verilerini (bkz. `app.orderbook`) OHLCV
tabanlı ML özellik çerçevesine, sembol bazında zaman bazlı en-yakın-geçmiş
(as-of, backward) eşleştirmeyle ekler.

`app.ml.macro_features` ile aynı mantık — geleceğe bakma (look-ahead
bias) YARATMAZ — tek fark: emir defteri sembole özgüdür (makro gibi tüm
piyasa için ortak değil), bu yüzden geçmiş DataFrame'i sembole göre
filtrelenir.

Not: Borsalar geçmişe dönük emir defteri saklamaz — bu tablo yalnızca
toplamaya başladığımız andan itibaren birikir. DB boşsa/erişilemezse
veya henüz o sembol için hiç toplanmamışsa tüm kolonlar NaN kalır
(eğitim akışını bozmaz, XGBoost NaN'ı doğal olarak ele alır)."""

import logging

import pandas as pd

from .features import ORDERBOOK_FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def load_orderbook_history(symbol: str) -> pd.DataFrame:
    """Bir sembolün emir defteri geçmişini zaman sıralı DataFrame olarak
    döner. DB kapalı/erişilemezse veya o sembol için hiç veri yoksa boş
    DataFrame döner."""
    try:
        from app.db.repository import get_orderbook_snapshots  # local import: döngüsel bağımlılığı önler

        history = get_orderbook_snapshots(symbol, limit=200_000)
    except Exception:  # noqa: BLE001 - order book geçmişi opsiyoneldir, ana akışı bozmamalı
        logger.exception("emir defteri geçmişi yüklenemedi: %s", symbol)
        return pd.DataFrame()

    if history.empty or "time" not in history.columns:
        return pd.DataFrame()

    history = history.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    history["time"] = pd.to_datetime(history["time"], utc=True)
    return history


def latest_orderbook_feature_row(symbol: str) -> dict[str, float]:
    """Canlı (tekil bar) tahmin için, o sembolün en son bilinen emir
    defteri özelliklerini döner. Geçmiş yoksa tüm değerler 0.0 (nötr)."""
    history = load_orderbook_history(symbol)
    result = {col: 0.0 for col in ORDERBOOK_FEATURE_COLUMNS}
    if history.empty:
        return result

    latest = history.iloc[-1]
    if "imbalance" in history.columns and not pd.isna(latest["imbalance"]):
        result["orderbook_imbalance"] = float(max(-1, min(1, latest["imbalance"])))
    if "spread_pct" in history.columns and not pd.isna(latest["spread_pct"]):
        result["orderbook_spread_norm"] = float(max(0, min(2, latest["spread_pct"])))
    if {"bid_volume", "ask_volume"}.issubset(history.columns):
        total_depth = history["bid_volume"] + history["ask_volume"]
        mean, std = total_depth.mean(), total_depth.std()
        if std and not pd.isna(std) and not pd.isna(total_depth.iloc[-1]):
            result["orderbook_depth_norm"] = float(max(-5, min(5, (total_depth.iloc[-1] - mean) / std)))
    return result


def merge_orderbook_features(frame: pd.DataFrame, orderbook_history: pd.DataFrame) -> pd.DataFrame:
    """`frame`'e ("timestamp" kolonu zorunlu) o sembolün emir defteri
    özelliklerini ekler. `orderbook_history` boşsa tüm kolonlar NaN kalır."""
    result = frame.copy()
    for col in ORDERBOOK_FEATURE_COLUMNS:
        result[col] = float("nan")

    if orderbook_history.empty or "timestamp" not in frame.columns:
        return result

    left = pd.DataFrame({"timestamp": pd.to_datetime(frame["timestamp"], unit="ns", utc=True)})
    left["_order"] = range(len(left))
    left = left.sort_values("timestamp")

    right = orderbook_history.rename(columns={"time": "timestamp"}).copy()
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True).astype("datetime64[ns, UTC]")

    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    merged = merged.sort_values("_order").reset_index(drop=True)

    if "imbalance" in merged.columns:
        result["orderbook_imbalance"] = merged["imbalance"].clip(-1, 1).to_numpy()
    if "spread_pct" in merged.columns:
        result["orderbook_spread_norm"] = merged["spread_pct"].clip(0, 2).to_numpy()
    if {"bid_volume", "ask_volume"}.issubset(merged.columns):
        total_depth = merged["bid_volume"] + merged["ask_volume"]
        mean = total_depth.mean()
        std = total_depth.std()
        if std and not pd.isna(std):
            result["orderbook_depth_norm"] = ((total_depth - mean) / std).clip(-5, 5).to_numpy()

    return result
