from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = Path(__file__).parent / "artifacts" / "signal_model.joblib"


@dataclass
class Prediction:
    direction: str  # "long" | "short" | "neutral"
    confidence: float  # 0..1, tahmin edilen sınıfın olasılığı


class SignalModel:
    """Tarama özelliklerinden yön (long/short/neutral) tahmini yapan
    çok katmanlı yapay sinir ağı (MLP) sarmalayıcısı.
    """

    def __init__(self) -> None:
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        max_iter=500,
                        random_state=42,
                        early_stopping=True,
                    ),
                ),
            ]
        )
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._pipeline.fit(X[FEATURE_COLUMNS], y)
        self._is_fitted = True

    def predict(self, feature_row: pd.Series) -> Prediction:
        if not self._is_fitted:
            raise RuntimeError("Model henüz eğitilmedi. Önce fit() veya load() çağırın.")

        x = feature_row[FEATURE_COLUMNS].to_frame().T
        proba = self._pipeline.predict_proba(x)[0]
        classes = self._pipeline.named_steps["mlp"].classes_
        best_idx = int(np.argmax(proba))
        label = classes[best_idx]
        confidence = float(proba[best_idx])

        direction = {1: "long", -1: "short", 0: "neutral"}[int(label)]
        return Prediction(direction=direction, confidence=confidence)

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
