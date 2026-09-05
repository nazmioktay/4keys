import numpy as np
import pandas as pd
import pytest

from app.engine.decision import DecisionEngine
from app.engine.positions import PaperPositionStore
from app.exchanges.base import Exchange
from app.ml.dataset import build_training_dataset
from app.ml.labeling import LONG, NEUTRAL, SHORT, triple_barrier_labels
from app.ml.meta_label import MetaLabelModel, build_meta_dataset
from app.ml.model import SignalModel


def _ohlcv_from_close(close: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(close), freq="4h"),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        }
    )


def test_triple_barrier_labels_upper_touched_first():
    # entry=100, hemen 105'e çıkıyor (TP=%2 -> 102 üstü), sonra düşüyor
    close = np.array([100, 106, 90, 90, 90, 90])
    ohlcv = _ohlcv_from_close(close)
    labels = triple_barrier_labels(ohlcv, take_profit_pct=2.0, stop_loss_pct=2.0, max_horizon=4)
    assert labels.iloc[0] == LONG


def test_triple_barrier_labels_lower_touched_first():
    close = np.array([100, 94, 110, 110, 110, 110])
    ohlcv = _ohlcv_from_close(close)
    labels = triple_barrier_labels(ohlcv, take_profit_pct=2.0, stop_loss_pct=2.0, max_horizon=4)
    assert labels.iloc[0] == SHORT


def test_triple_barrier_labels_timeout_is_neutral():
    # entry=100, bariyerler ±%10 -> dar hareketlerle hiçbiri tetiklenmez
    close = np.array([100, 100.5, 99.7, 100.2, 100.1, 99.9, 100.0])
    ohlcv = _ohlcv_from_close(close)
    labels = triple_barrier_labels(ohlcv, take_profit_pct=10.0, stop_loss_pct=10.0, max_horizon=5)
    assert labels.iloc[0] == NEUTRAL


def test_triple_barrier_labels_end_of_data_is_nan():
    close = np.array([100, 100.1, 100.2])  # max_horizon=5 ama sadece 2 mum kaldı
    ohlcv = _ohlcv_from_close(close)
    labels = triple_barrier_labels(ohlcv, take_profit_pct=10.0, stop_loss_pct=10.0, max_horizon=5)
    assert pd.isna(labels.iloc[0])


class _TrendExchange(Exchange):
    def __init__(self, seed: int = 1, n: int = 300) -> None:
        self._rng = np.random.default_rng(seed)
        self._n = n

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        return ["UPUSDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        n = max(limit, self._n)
        t = np.linspace(0, 12 * np.pi, n)
        base = 100 + 15 * np.sin(t) + np.linspace(0, 10, n)
        close = base + self._rng.normal(0, 0.8, n)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": self._rng.uniform(800, 1200, n),
            }
        )


def test_dataset_builds_with_triple_barrier_method():
    exchange = _TrendExchange(seed=2)
    X, y = build_training_dataset(
        exchange, ["UPUSDT"], "4h", 300, horizon=8,
        labeling_method="triple_barrier", take_profit_pct=1.5, stop_loss_pct=1.5,
    )
    assert len(X) > 50
    assert set(y.unique()) <= {-1.0, 0.0, 1.0}


def test_triple_barrier_labels_accepts_per_row_pct_array():
    """`take_profit_pct`/`stop_loss_pct` bir Series/dizi olarak da
    verilebilmeli (ATR-ölçekli, her bar için FARKLI bariyer genişliği) —
    bkz. `app.ml.dataset._compute_labels`'ın `atr_triple_barrier` yolu."""
    close = np.array([100, 106, 90, 90, 90, 90])
    ohlcv = _ohlcv_from_close(close)
    # satır 0 için TP=%2 (102) -> ilk barda 106 ile tetiklenir (LONG)
    tp_pct = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    sl_pct = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    labels = triple_barrier_labels(ohlcv, tp_pct, sl_pct, max_horizon=4)
    assert labels.iloc[0] == LONG

    # AYNI veri ama satır 0 için ÇOK GENİŞ bir TP/SL (%50) verilirse artık
    # hiçbir bariyer tetiklenmemeli -> zaman aşımı (NEUTRAL)
    wide_tp = pd.Series([50.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    wide_sl = pd.Series([50.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    labels_wide = triple_barrier_labels(ohlcv, wide_tp, wide_sl, max_horizon=4)
    assert labels_wide.iloc[0] == NEUTRAL


def test_dataset_builds_with_atr_triple_barrier_method():
    """`atr_triple_barrier`: take_profit_pct/stop_loss_pct ATR ÇARPANI
    olarak yorumlanır (sabit yüzde değil) — her bar kendi volatilitesine
    göre ölçeklenen bir bariyer genişliği alır. Gerçek çıkışla (ATR
    tabanlı stop/hedef, bkz. app.backtest.system_runner) aynı mantık."""
    exchange = _TrendExchange(seed=3)
    X, y = build_training_dataset(
        exchange, ["UPUSDT"], "4h", 300, horizon=8,
        labeling_method="atr_triple_barrier", take_profit_pct=1.5, stop_loss_pct=1.5,
    )
    assert len(X) > 50
    assert set(y.unique()) <= {-1.0, 0.0, 1.0}
    # tamamen tek bir sınıfa çökmemeli (en azından biraz yön çeşitliliği olmalı)
    assert y.nunique() >= 2


def test_signal_model_calibrates_with_enough_class_diversity():
    # Kalibrasyon eşiği artık her sınıftan cv katı başına en az 10 örnek
    # gerektiriyor (bkz. app.ml.model — küçük sınıflarda kalibrasyonun
    # gürültüyü öğrenip modeli çoğunluk sınıfına ittiği bulgusu üzerine
    # yükseltildi); 300 mumluk eski küçük set artık bunu karşılamıyor,
    # bu yüzden lookback büyütüldü.
    exchange = _TrendExchange(seed=3)
    X, y = build_training_dataset(exchange, ["UPUSDT"], "4h", 1500, horizon=5, threshold_pct=0.5)
    model = SignalModel(calibrate=True)
    model.fit(X, y)
    assert model.is_calibrated

    predictions, confidences = model.predict_batch(X)
    assert len(predictions) == len(X)
    assert ((confidences >= 0) & (confidences <= 1)).all()


def test_signal_model_falls_back_when_class_too_small():
    # Yapay olarak bir sınıftan sadece 1 örnek olan, dengesiz bir set kur.
    exchange = _TrendExchange(seed=4)
    X, y = build_training_dataset(exchange, ["UPUSDT"], "4h", 300, horizon=5, threshold_pct=0.5)
    y = y.copy()
    y.iloc[:] = 1.0
    y.iloc[0] = -1.0  # tek bir -1 örneği -> cv=3 için yetersiz

    model = SignalModel(calibrate=True)
    model.fit(X, y)
    assert not model.is_calibrated  # güvenli şekilde kalibrasyonsuz moda düştü

    pred = model.predict(X.iloc[1])
    assert pred.direction in ("long", "short", "neutral")


def test_meta_label_dataset_and_model_roundtrip():
    exchange = _TrendExchange(seed=5)
    X, y = build_training_dataset(exchange, ["UPUSDT"], "4h", 300, horizon=5, threshold_pct=0.5)
    primary = SignalModel()
    primary.fit(X, y)

    meta_X, meta_y = build_meta_dataset(primary, X, y)
    assert len(meta_X) == len(X)
    assert set(meta_y.unique()) <= {0, 1}

    if meta_y.nunique() < 2:
        pytest.skip("Bu tohumda birincil model hep doğru/hep yanlış tahmin etmiş, meta eğitilemez.")

    meta_model = MetaLabelModel()
    meta_model.fit(meta_X, meta_y)

    prediction = primary.predict(X.iloc[0])
    decision = meta_model.decide(X.iloc[0], prediction.confidence)
    assert isinstance(decision.act, bool)
    assert 0.0 <= decision.confidence <= 1.0


def test_decision_engine_respects_meta_label_veto():
    """Meta model her zaman 'güvenme' derse, motor asla pozisyon açmamalı."""
    exchange = _TrendExchange(seed=6)
    X, y = build_training_dataset(exchange, ["UPUSDT"], "4h", 300, horizon=5, threshold_pct=0.5)
    primary = SignalModel()
    primary.fit(X, y)

    class AlwaysVetoMetaModel(MetaLabelModel):
        def __init__(self):
            self._is_fitted = True

        def decide(self, feature_row, primary_confidence):
            from app.ml.meta_label import MetaDecision
            return MetaDecision(act=False, confidence=0.9)

    engine = DecisionEngine(
        exchange=exchange,
        model=primary,
        positions=PaperPositionStore(),
        timeframe="4h",
        lookback=300,
        open_confidence=0.0,
        close_confidence=0.0,
        meta_model=AlwaysVetoMetaModel(),
    )

    actions = engine.run_cycle(["UPUSDT"])
    assert len(actions) == 1
    assert actions[0].type == "hold"
    assert "meta-label" in actions[0].reason
