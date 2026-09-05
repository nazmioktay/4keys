"""Open interest (açık pozisyon, bkz. `app.openinterest`) verilerini OHLCV
tabanlı ML özellik çerçevesine, sembol bazında zaman bazlı en-yakın-geçmiş
(as-of, backward) eşleştirmeyle ekler.

`app.ml.orderbook_features` ile AYNI mantık — geleceğe bakma (look-ahead
bias) YARATMAZ, sembole özgüdür. Borsalar geçmişe dönük open interest
saklamaz/satmaz — bu tablo yalnızca toplamaya başladığımız andan itibaren
birikir. DB boşsa/erişilemezse veya henüz o sembol için hiç toplanmamışsa
tüm kolonlar NaN kalır (eğitim akışını bozmaz, XGBoost NaN'ı doğal olarak
ele alır)."""

import logging

import numpy as np
import pandas as pd

from .features import OPEN_INTEREST_FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def load_open_interest_history(symbol: str) -> pd.DataFrame:
    """Bir sembolün open interest geçmişini zaman sıralı DataFrame olarak
    döner. DB kapalı/erişilemezse veya o sembol için hiç veri yoksa boş
    DataFrame döner."""
    try:
        from app.db.repository import get_open_interest_snapshots  # local import: döngüsel bağımlılığı önler

        history = get_open_interest_snapshots(symbol, limit=200_000)
    except Exception:  # noqa: BLE001 - open interest geçmişi opsiyoneldir, ana akışı bozmamalı
        logger.exception("open interest geçmişi yüklenemedi: %s", symbol)
        return pd.DataFrame()

    if history.empty or "time" not in history.columns:
        return pd.DataFrame()

    history = history.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    history["time"] = pd.to_datetime(history["time"], utc=True)
    return history


def latest_open_interest_feature_row(symbol: str) -> dict[str, float]:
    """Canlı (tekil bar) tahmin için, o sembolün en son bilinen open
    interest değişim özelliklerini döner. Geçmiş yetersizse (0-1 satır)
    tüm değerler 0.0 (nötr)."""
    history = load_open_interest_history(symbol)
    result = {col: 0.0 for col in OPEN_INTEREST_FEATURE_COLUMNS}
    if len(history) < 2 or "open_interest" not in history.columns:
        return result

    oi = history["open_interest"].astype(float)
    change_pct = float(((oi.iloc[-1] / oi.iloc[-2]) - 1) * 100) if oi.iloc[-2] not in (0, None) and not pd.isna(oi.iloc[-2]) else 0.0
    result["oi_change_pct"] = float(np.clip(change_pct, -20, 20))
    # Bu fonksiyon fiyat geçmişine erişemez (yalnızca OI geçmişi) — yön
    # uyumu (`oi_price_divergence`) yalnızca `merge_open_interest_features`
    # (OHLCV ile birlikte) hesaplanabilir; canlıda bu alan 0.0 (nötr) kalır,
    # `merge`'ün eğitimdeki AYNI basitleştirmesi (bkz. dosya docstring'i).
    return result


def merge_open_interest_features(frame: pd.DataFrame, oi_history: pd.DataFrame) -> pd.DataFrame:
    """`frame`'e ("timestamp" VE "close" kolonları zorunlu) o sembolün open
    interest özelliklerini ekler. `oi_history` boşsa tüm kolonlar NaN kalır.

    - `oi_change_pct`: ardışık iki open interest anlık görüntüsü arasındaki
      yüzde değişim (%, -20..20 kırpılır).
    - `oi_price_divergence`: AYNI pencerede fiyat yönü ile OI yönünün
      işaret uyumu (+1 = ikisi de aynı yönde, ör. fiyat↑+OI↑ "yeni long
      pozisyonlar" veya fiyat↓+OI↓ "long tasfiyesi/kâr realizasyonu" —
      trend YENİ pozisyonlarla mı yoksa pozisyon KAPANIŞIYLA mı sürüyor
      ayrımının basit bir sayısal karşılığı; -1 = zıt yönde, ör. fiyat↑+OI↓
      "kısa pozisyon kapanışı" — trendin YENİ ilgi olmadan sürdüğüne işaret).
    """
    result = frame.copy()
    for col in OPEN_INTEREST_FEATURE_COLUMNS:
        result[col] = float("nan")

    if oi_history.empty or "timestamp" not in frame.columns or "open_interest" not in oi_history.columns:
        return result

    left = pd.DataFrame({"timestamp": pd.to_datetime(frame["timestamp"], unit="ns", utc=True), "close": frame["close"]})
    left["_order"] = range(len(left))
    left = left.sort_values("timestamp")

    right = oi_history.rename(columns={"time": "timestamp"})[["timestamp", "open_interest"]].copy()
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    right["oi_change_pct"] = (right["open_interest"].pct_change() * 100).clip(-20, 20)
    right["_oi_direction"] = np.sign(right["open_interest"].diff())

    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    merged = merged.sort_values("_order").reset_index(drop=True)

    result["oi_change_pct"] = merged["oi_change_pct"].to_numpy()

    price_direction = np.sign(merged["close"].diff())
    divergence = price_direction * merged["_oi_direction"]
    result["oi_price_divergence"] = divergence.fillna(0.0).clip(-1, 1).to_numpy()

    return result
