import logging

import numpy as np
import pandas as pd
from app.backtest.data import timeframe_to_minutes
from app.exchanges.base import Exchange
from app.exchanges.cache import fetch_ohlcv_cached

from .data_quality import warn_if_gaps
from .dataset import LabelingMethod, _compute_labels, _persist_feature_snapshots
from .features import ALL_FEATURE_COLUMNS, FEATURE_COLUMNS, build_features
from .macro_features import load_macro_history, merge_macro_features
from .orderbook_features import load_orderbook_history, merge_orderbook_features
from .orderflow_features import merge_taker_flow_features

logger = logging.getLogger(__name__)


def build_sequence_dataset(
    exchange: Exchange,
    symbols: list[str],
    timeframe: str,
    lookback: int,
    seq_len: int = 20,
    horizon: int = 5,
    threshold_pct: float = 1.0,
    labeling_method: LabelingMethod = "threshold",
    take_profit_pct: float = 2.0,
    stop_loss_pct: float = 2.0,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LSTM/PatchTST eğitimi için kayan pencereli (sliding window) sekans
    veri seti kurar. `app.ml.dataset.build_training_dataset`'ten farkı:
    XGBoost tekil özellik satırlarıyla çalışırken, sekans modelleri her
    tahmin için son `seq_len` barın özellik vektörünü sırayla (zaman
    bilgisini koruyarak) görür.

    Her sembol kendi içinde bağımsız işlenir ve pencereler yalnızca o
    sembolün KENDİ kronolojik/kesintisiz serisinden kurulur — semboller
    arası pencere sızıntısı olmaz.

    `feature_columns` verilmezse `ALL_FEATURE_COLUMNS` (teknik + makro +
    order-book, XGBoost eğitim yolu — `app.ml.dataset` — ile AYNI özellik
    kümesi) kullanılır. Önceki sürüm burada makro/order-book özelliklerini
    HİÇ görmüyordu (yalnızca 39 teknik özellik) — XGBoost ve canlı karar
    motoruyla (`app.engine.decision`) tutarsız bir gerçek eksiklikti,
    düzeltildi. `feature_columns` bir alt küme olarak verilirse (ör.
    `/ml/explain`'in SHAP önem sıralamasından seçilen en değerli N özellik),
    yalnızca o sütunlarla eğitim yapılır — küçük veri setlerinde
    boyut/örnek oranını iyileştirmek için.

    Döner: (X, y, time_frac)
    - X: şekil (n_pencere, seq_len, n_özellik)
    - y: şekil (n_pencere,) — her pencerenin SON barındaki etiket
    - time_frac: şekil (n_pencere,) — o sembol serisi içindeki göreli
      zaman konumu (0..1), out-of-sample holdout bölmesi için
      (bkz. `app.ml.validation.split_out_of_sample`, burada numpy
      maskesiyle eşdeğer mantık uygulanır).
    """
    columns = feature_columns or ALL_FEATURE_COLUMNS
    # Makro/order-book geçmişi olmayan barlarda (ör. testlerde ya da henüz
    # backfill edilmemiş erken dönemlerde) o kolonlar NaN kalır — XGBoost
    # yolundaki (`app.ml.dataset._build_symbol_frames`) DAVRANIŞLA TUTARLI
    # olması için yalnızca HER ZAMAN yoğun olan teknik özellikler (bkz.
    # `FEATURE_COLUMNS`) `dropna` zorunluluğuna tabidir; makro/order-book
    # NaN'ları satırı elemez, aşağıda `fillna(0.0)` ile nötrlenir.
    dense_columns = [c for c in columns if c in FEATURE_COLUMNS]
    macro_history = load_macro_history()
    # Not (bellek): önceki sürüm her pencereyi ayrı bir küçük numpy dizisi
    # olarak bir Python listesine ekliyordu (n_pencere adet nesne, her biri
    # kendi bellek tahsisi + Python nesne başlığıyla), sonra `np.stack` bunu
    # tek bir büyük diziye kopyalıyordu — liste hâlâ bellekteyken. 10.000
    # mum × ~20 sembol × seq_len=20 örtüşen pencerede bu, gerçek veri
    # boyutunun (örtüşme nedeniyle zaten ~seq_len kat şişmiş) üzerine bir
    # kat daha (parçalanma + geçici kopya) bindiriyor ve üretim sunucusunda
    # OOM'a yol açtı. Bunun yerine `sliding_window_view` ile sembol başına
    # bir VIEW (kopyasız) çıkarılır; tüm semboller tek seferde
    # `np.concatenate` ile birleştirilir (kaçınılmaz tek kopya, ama
    # sembol/pencere başına ayrı Python nesnesi yok).
    per_symbol_X: list[np.ndarray] = []
    per_symbol_y: list[np.ndarray] = []
    per_symbol_time_frac: list[np.ndarray] = []

    for symbol in symbols:
        try:
            ohlcv = fetch_ohlcv_cached(exchange, symbol, timeframe, lookback)
            if len(ohlcv) < 60:
                continue
            warn_if_gaps(symbol, timeframe, ohlcv, timeframe_to_minutes(timeframe))
            features = build_features(ohlcv)
            _persist_feature_snapshots(symbol, timeframe, features)
            features = merge_macro_features(features, macro_history)
            features = merge_orderbook_features(features, load_orderbook_history(symbol))
            features = merge_taker_flow_features(features, ohlcv, exchange, symbol, timeframe)
            labels = _compute_labels(ohlcv, labeling_method, horizon, threshold_pct, take_profit_pct, stop_loss_pct)
            frame = features.copy()
            frame["label"] = labels
            frame = frame.dropna(subset=dense_columns + ["label"]).reset_index(drop=True)

            n = len(frame)
            if n < seq_len + 1:
                continue

            feat_values = frame[columns].fillna(0.0).to_numpy(dtype="float32")
            label_values = frame["label"].to_numpy()

            windows = np.lib.stride_tricks.sliding_window_view(feat_values, seq_len, axis=0)
            # sliding_window_view (n - seq_len + 1, n_özellik, seq_len) döner;
            # beklenen (n_pencere, seq_len, n_özellik) şekline taşınır.
            windows = np.moveaxis(windows, -1, 1)

            per_symbol_X.append(windows)
            per_symbol_y.append(label_values[seq_len - 1 :])
            positions = np.arange(seq_len - 1, n, dtype="float64")
            per_symbol_time_frac.append(positions / max(n - 1, 1))
        except Exception as exc:  # noqa: BLE001 - tek sembol hatası tüm eğitimi durdurmamalı
            logger.warning("sequence dataset: skipping %s: %s", symbol, exc)
            continue

    if not per_symbol_X:
        return (
            np.empty((0, seq_len, len(columns)), dtype="float32"),
            np.empty((0,)),
            np.empty((0,)),
        )

    X = np.concatenate(per_symbol_X, axis=0)
    y = np.concatenate(per_symbol_y, axis=0)
    time_frac = np.concatenate(per_symbol_time_frac, axis=0)
    return X, y, time_frac


def latest_sequence_window(
    ohlcv: pd.DataFrame,
    symbol: str,
    seq_len: int,
    feature_columns: list[str] | None = None,
    exchange=None,
    timeframe: str | None = None,
) -> np.ndarray | None:
    """CANLI tahmin için: `ohlcv` (borsadan zaten çekilmiş — burada YENİDEN
    borsa çağrısı yapılmaz) üzerinden en son `seq_len` barın özellik
    penceresini kurar, `LSTMSignalModel.predict`/`PatchTSTSignalModel.predict`
    ile doğrudan uyumlu şekilde.

    `app.engine.decision.DecisionEngine`'in XGBoost+LSTM ensemble'ı için
    eklendi — `build_sequence_dataset` ile aynı özellik hazırlama mantığını
    (makro/order-book/order-flow birleştirme, `fillna(0.0)`) paylaşır ama
    tek bir sembol için tek bir pencere döner ve mevcut `ohlcv` verisini
    yeniden kullanır (ekstra ağ çağrısı yok) — `exchange`/`timeframe` yalnızca
    order-flow (`merge_taker_flow_features`, kendi ayrı ağ çağrısını yapar)
    için gerekli; verilmezse order-flow kolonu nötr (0.0) kalır.

    Yeterli veri yoksa (ör. `len(ohlcv) < seq_len`) `None` döner.
    """
    columns = feature_columns or ALL_FEATURE_COLUMNS
    dense_columns = [c for c in columns if c in FEATURE_COLUMNS]

    features = build_features(ohlcv)
    features = merge_macro_features(features, load_macro_history())
    features = merge_orderbook_features(features, load_orderbook_history(symbol))
    if exchange is not None and timeframe is not None:
        features = merge_taker_flow_features(features, ohlcv, exchange, symbol, timeframe)
    frame = features.dropna(subset=dense_columns).reset_index(drop=True)

    if len(frame) < seq_len:
        return None

    values = frame[columns].fillna(0.0).to_numpy(dtype="float32")
    return values[-seq_len:]
