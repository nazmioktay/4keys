"""Dinamik çıkış (kâr-al/zarar-durdur) tahmini — kullanıcı önerisi
"Seviye 1: Süpervizeli Öğrenme (Regresyon)": sabit bir ATR çarpanı
(`app.ml.dataset._compute_labels`'ın `atr_triple_barrier` yolu) yerine,
giriş anındaki piyasa koşullarına (mevcut 63 özellik) bakıp "bu girişte
fiyat ne kadar yükselip/düşecek" sorusunun cevabını REGRESYON ile tahmin
eden ayrı bir model.

Bu modül BİLEREK üretim `SignalModel`'ından (XGBoost sınıflandırıcı)
TAMAMEN AYRI tutuluyor — `_ENSEMBLE_LABELING`i paylaşmıyor, kendi
regresyon hedefini (`label_future_peak_trough`) kullanıyor. Şu an
yalnızca İZOLE ölçüm (MAE/R²) için var; canlı karar döngüsüne/backtest'e
KABLOLANMADI — bugünkü oturumda tekrar tekrar görülen dersle uyumlu
("izole doğruluk artışı gerçek sistem performansını garanti etmez"),
önce bu modelin GERÇEKTEN anlamlı bir şey öğrenip öğrenmediğini
(izole metriklerle) ölçmeden tam sisteme entegre etmek riskli olur.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from app.core.memory_probe import log_rss
from app.exchanges.base import Exchange
from app.exchanges.cache import fetch_ohlcv_cached

from .features import FEATURE_COLUMNS, build_features
from .labeling import label_future_peak_trough
from .validation import split_out_of_sample

logger = logging.getLogger(__name__)

DEFAULT_DYNAMIC_EXIT_MODEL_PATH = Path(__file__).parent / "artifacts" / "dynamic_exit_model.joblib"


@dataclass
class DynamicExitEvalReport:
    rows_used: int
    peak_mae: float
    peak_r2: float
    trough_mae: float
    trough_r2: float


@dataclass
class DynamicExitTrainingResult:
    model: "DynamicExitModel"
    rows_used: int
    out_of_sample: DynamicExitEvalReport


class DynamicExitModel:
    """İki ayrı `XGBRegressor` — biri `future_peak_pct`, biri
    `future_trough_pct` tahmin eder. `SignalModel`'in aksine kalibrasyon
    YOK (regresyon çıktısı, olasılık değil)."""

    def __init__(self) -> None:
        self.peak_model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42)
        self.trough_model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42)

    def fit(self, X: pd.DataFrame, y_peak: pd.Series, y_trough: pd.Series) -> None:
        self.peak_model.fit(X, y_peak)
        self.trough_model.fit(X, y_trough)

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Döner: (tahmini future_peak_pct, tahmini future_trough_pct) —
        ikisi de yüzde, girişe göre. LONG pozisyon için kâr-al hedefi
        `entry * (1 + peak/100)`, zarar-durdur `entry * (1 + trough/100)`
        (trough negatif olduğu için otomatik aşağıda kalır)."""
        return self.peak_model.predict(X), self.trough_model.predict(X)

    def save(self, path=None) -> None:
        import joblib

        target = path or DEFAULT_DYNAMIC_EXIT_MODEL_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load_from(cls, path=None) -> "DynamicExitModel":
        import joblib

        return joblib.load(path or DEFAULT_DYNAMIC_EXIT_MODEL_PATH)


def _build_dynamic_exit_frame(exchange: Exchange, symbol: str, timeframe: str, lookback: int, horizon: int) -> pd.DataFrame | None:
    ohlcv = fetch_ohlcv_cached(exchange, symbol, timeframe, lookback)
    if len(ohlcv) < 60:
        return None
    features = build_features(ohlcv)
    targets = label_future_peak_trough(ohlcv, horizon=horizon)
    frame = features.copy()
    frame["future_peak_pct"] = targets["future_peak_pct"]
    frame["future_trough_pct"] = targets["future_trough_pct"]
    frame["time_frac"] = pd.RangeIndex(len(frame)) / max(len(frame) - 1, 1)
    frame = frame.dropna(subset=FEATURE_COLUMNS + ["future_peak_pct", "future_trough_pct"])
    return frame if not frame.empty else None


def train_dynamic_exit_model(
    exchange: Exchange,
    symbols: list[str],
    timeframe: str | None = None,
    lookback: int | None = None,
    horizon: int = 25,
    holdout_frac: float = 0.2,
    persist: bool = True,
) -> DynamicExitTrainingResult:
    """`app.ml.train.train_signal_model_validated` ile AYNI holdout
    prensibi (bkz. o fonksiyonun docstring'i): model son `holdout_frac`
    dilime HİÇ fit edilmez, yalnızca izole MAE/R² için kullanılır."""
    from app.core.config import settings

    tf = timeframe or settings.ml_train_timeframe
    lb = lookback or settings.ml_train_lookback

    frames = []
    for symbol in symbols:
        try:
            log_rss(f"dynamic_exit: {symbol} öncesi")
            frame = _build_dynamic_exit_frame(exchange, symbol, tf, lb, horizon)
            if frame is not None:
                frames.append(frame)
        except Exception as exc:  # noqa: BLE001 - tek sembol hatası tüm eğitimi durdurmamalı
            logger.warning("dynamic_exit: skipping %s: %s", symbol, exc)

    if not frames:
        raise ValueError("Eğitim için yeterli veri yok (hiçbir sembolden geçerli çerçeve üretilemedi).")

    data = pd.concat(frames, ignore_index=True)
    if len(data) < 60:
        raise ValueError(f"Eğitim için yeterli veri yok ({len(data)} satır).")

    X = data[FEATURE_COLUMNS]
    y_peak = data["future_peak_pct"]
    y_trough = data["future_trough_pct"]
    time_frac = data["time_frac"]

    X_train, y_peak_train, X_holdout, y_peak_holdout = split_out_of_sample(X, y_peak, time_frac, holdout_frac)
    # Aynı satır bölünmesini `future_trough_pct` için de uygular (index'ler
    # `split_out_of_sample`'dan AYNI şekilde geri gelir).
    y_trough_train = y_trough.loc[X_train.index]
    y_trough_holdout = y_trough.loc[X_holdout.index]

    model = DynamicExitModel()
    model.fit(X_train, y_peak_train, y_trough_train)

    if len(X_holdout) > 0:
        pred_peak, pred_trough = model.predict(X_holdout)
        report = DynamicExitEvalReport(
            rows_used=len(data),
            peak_mae=float(mean_absolute_error(y_peak_holdout, pred_peak)),
            peak_r2=float(r2_score(y_peak_holdout, pred_peak)),
            trough_mae=float(mean_absolute_error(y_trough_holdout, pred_trough)),
            trough_r2=float(r2_score(y_trough_holdout, pred_trough)),
        )
    else:
        report = DynamicExitEvalReport(rows_used=len(data), peak_mae=0.0, peak_r2=0.0, trough_mae=0.0, trough_r2=0.0)

    if persist:
        model.save()

    return DynamicExitTrainingResult(model=model, rows_used=len(data), out_of_sample=report)


@dataclass
class DynamicExitHorizonSweepPoint:
    horizon: int
    rows_used: int = 0
    peak_mae: float = 0.0
    peak_r2: float = 0.0
    trough_mae: float = 0.0
    trough_r2: float = 0.0
    error: str | None = None


def sweep_dynamic_exit_horizons(
    exchange: Exchange,
    symbols: list[str],
    horizon_values: list[int],
    timeframe: str | None = None,
    lookback: int | None = None,
) -> list[DynamicExitHorizonSweepPoint]:
    """`horizon` (girişten sonra kaç bar ileriye bakılacağı) ızgarasında
    ard arda `train_dynamic_exit_model` çalıştırır — `app.backtest.system_runner`'daki
    diğer sweep fonksiyonlarıyla AYNI desen: hiçbir model KAYDEDİLMEZ
    (`persist=False`), otomatik "en iyi" SEÇİLMEZ, karar operatöre kalır.

    NOT: `future_peak_pct=25 bar` `atr_triple_barrier`'ın `horizon`'uyla
    (üretim modelinin ZAMAN bariyeri) AYNI KAVRAM DEĞİLDİR — burası
    tamamen ayrı, izole bir regresyon deneyi; sonuçları üretim modelinin
    `_ENSEMBLE_LABELING["horizon"]`'unu DEĞİŞTİRMEZ."""
    points: list[DynamicExitHorizonSweepPoint] = []
    for horizon in horizon_values:
        try:
            result = train_dynamic_exit_model(exchange, symbols, timeframe=timeframe, lookback=lookback, horizon=horizon, persist=False)
            r = result.out_of_sample
            points.append(
                DynamicExitHorizonSweepPoint(
                    horizon=horizon,
                    rows_used=result.rows_used,
                    peak_mae=r.peak_mae,
                    peak_r2=r.peak_r2,
                    trough_mae=r.trough_mae,
                    trough_r2=r.trough_r2,
                )
            )
        except ValueError as exc:
            points.append(DynamicExitHorizonSweepPoint(horizon=horizon, error=str(exc)))
    return points
