import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Mum bazlı agresif alım/satım akışı (order flow / trade aggression).
# `app.ml.orderbook_features`teki emir defteri ANLIK GÖRÜNTÜSÜNDEN
# (statik, yalnızca periyodik olarak toplanabilen, geçmişi olmayan)
# FARKLI ve TAMAMLAYICI bir mikro yapı sinyali: Binance'in kline uç
# noktasındaki "taker buy base asset volume" alanından türetilir, mum
# bazlıdır ve TAM GEÇMİŞE sahiptir (backfill/ayrı bir biriktirme işi
# gerektirmez) — ama yine de her istekte ekstra bir ağ çağrısı gerektirir
# ve her borsa adapter'ı bunu desteklemez (bkz. `fetch_taker_flow`
# opsiyonel yetenek kontrolü, `app.macro.data`teki `fetch_funding_rate`
# ile aynı desen).
TAKER_FLOW_FEATURE_COLUMNS = ["taker_buy_ratio_norm"]


def merge_taker_flow_features(features: pd.DataFrame, ohlcv: pd.DataFrame, exchange, symbol: str, timeframe: str) -> pd.DataFrame:
    """`features`e ("timestamp" kolonu zorunlu, `app.ml.features.build_features`
    çıktısı) `taker_buy_ratio_norm` (-1..1, 0 = dengeli alım/satım akışı,
    +1 = tamamen agresif ALIM, -1 = tamamen agresif SATIM) ekler.

    Borsa bu özelliği desteklemiyorsa veya ağ hatası olursa kolon NaN
    bırakılır (satır ELENMEZ) — çağıran taraf (`app.ml.dataset`/
    `app.ml.sequence_dataset`) diğer opsiyonel özelliklerle (makro,
    order-book) AYNI şekilde `fillna(0.0)` ile nötrler.
    """
    result = features.copy()
    result["taker_buy_ratio_norm"] = float("nan")

    fetch_taker_flow = getattr(exchange, "fetch_taker_flow", None)
    if fetch_taker_flow is None or "timestamp" not in features.columns or ohlcv.empty:
        return result

    try:
        flow = fetch_taker_flow(symbol, timeframe, len(ohlcv))
    except Exception as exc:  # noqa: BLE001 - opsiyonel özellik, ana akışı bozmamalı
        logger.warning("taker flow: %s için alınamadı: %s", symbol, exc)
        return result

    if flow.empty:
        return result

    left = pd.DataFrame({"timestamp": pd.to_datetime(features["timestamp"], unit="ns")})
    merged = pd.merge(left, flow, on="timestamp", how="left")
    ratio = merged["taker_buy_base_volume"] / merged["volume"].replace(0, float("nan"))
    result["taker_buy_ratio_norm"] = ((ratio - 0.5) * 2).to_numpy()
    return result


def latest_taker_buy_ratio_norm(exchange, symbol: str, timeframe: str) -> float:
    """Canlı tahmin (`app.engine.decision`/`POST /ml/predict`) için: son
    kapanmış mumun `taker_buy_ratio_norm` değerini döner; borsa
    desteklemiyorsa veya hata olursa nötr (0.0) döner."""
    fetch_taker_flow = getattr(exchange, "fetch_taker_flow", None)
    if fetch_taker_flow is None:
        return 0.0
    try:
        flow = fetch_taker_flow(symbol, timeframe, 2)
    except Exception as exc:  # noqa: BLE001 - opsiyonel özellik, ana akışı bozmamalı
        logger.warning("taker flow: %s için canlı değer alınamadı: %s", symbol, exc)
        return 0.0
    if flow.empty:
        return 0.0
    last = flow.iloc[-1]
    if last["volume"] <= 0:
        return 0.0
    ratio = float(last["taker_buy_base_volume"] / last["volume"])
    return (ratio - 0.5) * 2
