from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = Path(__file__).parent / "artifacts" / "signal_model.joblib"


@dataclass
class Prediction:
    direction: str  # "long" | "short" | "neutral"
    confidence: float  # 0..1, tahmin edilen sınıfın olasılığı


def _build_base_pipeline(early_stopping: bool = True) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    max_iter=500,
                    random_state=42,
                    early_stopping=early_stopping,
                ),
            ),
        ]
    )


class SignalModel:
    """Tarama özelliklerinden yön (long/short/neutral) tahmini yapan
    çok katmanlı yapay sinir ağı (MLP) sarmalayıcısı.

    `calibrate=True` (varsayılan) iken model, ham MLP olasılık çıktısını
    doğrudan kullanmaz — `CalibratedClassifierCV` ile Platt scaling
    (`method="sigmoid"`) veya isotonic regression uygulanır. Bu önemlidir:
    kalibre edilmemiş bir "%60 güven" değeri gerçek bir olasılık değildir ve
    doğrudan Kelly kriterine (bkz. `app.portfolio.risk_manager`) verilirse
    pozisyon boyutları sistematik olarak hatalı büyür/küçülür.
    """

    def __init__(self, calibrate: bool = True, calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid") -> None:
        self._base_pipeline = _build_base_pipeline()
        self._pipeline = self._base_pipeline
        self._calibrate = calibrate
        self._calibration_method = calibration_method
        self._is_fitted = False
        self.is_calibrated = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_features = X[FEATURE_COLUMNS]
        class_counts = y.value_counts()
        min_class_count = int(class_counts.min()) if len(class_counts) else 0

        # MLPClassifier'ın kendi early_stopping mekanizması, iç doğrulama
        # bölmesi için sınıf başına en az birkaç örnek gerektirir; aşırı
        # dengesiz/küçük eğitim setlerinde (ör. bir sınıftan tek örnek)
        # bunu kapatmak gerekir, yoksa fit() hata fırlatır.
        self._base_pipeline = _build_base_pipeline(early_stopping=min_class_count >= 5)

        # CalibratedClassifierCV, StratifiedKFold(cv) kullanır; bir sınıfın örnek
        # sayısı cv'den azsa (küçük/dengesiz eğitim setlerinde olur) hata verir.
        # Bu durumda kalibrasyonu atlayıp ham (kalibre edilmemiş) modele düşülür —
        # sessizce yanlış/aşırı güvenli skorlar üretmektense bu tercih edilir.
        can_calibrate = bool(self._calibrate and len(class_counts) >= 2 and min_class_count >= 3)

        if can_calibrate:
            calibrated = CalibratedClassifierCV(self._base_pipeline, method=self._calibration_method, cv=3)
            calibrated.fit(X_features, y)
            self._pipeline = calibrated
        else:
            self._base_pipeline.fit(X_features, y)
            self._pipeline = self._base_pipeline
        self._is_fitted = True
        self.is_calibrated = can_calibrate

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model henüz eğitilmedi. Önce fit() veya load() çağırın.")

    def predict(self, feature_row: pd.Series) -> Prediction:
        self._require_fitted()
        x = feature_row[FEATURE_COLUMNS].to_frame().T
        proba = self._pipeline.predict_proba(x)[0]
        classes = self._pipeline.classes_
        best_idx = int(np.argmax(proba))
        label = classes[best_idx]
        confidence = float(proba[best_idx])

        direction = {1: "long", -1: "short", 0: "neutral"}[int(label)]
        return Prediction(direction=direction, confidence=confidence)

    def predict_batch(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Bir DataFrame'in tamamı için (tahmin edilen sınıf, o sınıfın
        kalibre edilmiş olasılığı) dizilerini döner. Meta-labeling eğitim
        seti kurmak ve toplu değerlendirme için kullanılır."""
        self._require_fitted()
        x = X[FEATURE_COLUMNS]
        proba = self._pipeline.predict_proba(x)
        classes = self._pipeline.classes_
        best_idx = np.argmax(proba, axis=1)
        predictions = classes[best_idx]
        confidences = proba[np.arange(len(proba)), best_idx]
        return predictions, confidences

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    def load(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        self._pipeline = joblib.load(path)
        self._is_fitted = True

    @classmethod
    def load_from(cls, path: Path = DEFAULT_MODEL_PATH) -> "SignalModel":
        model = cls()
        model.load(path)
        return model
