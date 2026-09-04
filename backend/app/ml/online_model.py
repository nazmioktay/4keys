"""Gerçek çevrimiçi (online) öğrenme — kavram kayması (concept drift)
yönetimi için.

Kullanıcı, XGBoost/LSTM'i periyodik olarak toptan (batch) yeniden eğitmek
yerine (mevcut `job_auto_retrain`, sabit ve ampirik olarak doğrulanmamış
bir aralıkla çalışır) Hoeffding Ağaçları / Online Random Forest gibi
verinin akışından ANLIK öğrenen modelleri önerdi ve `river` kütüphanesini
eklemeyi onayladı (hafif, aktif bakımlı, saf Python + Cython).

`river.forest.ARFClassifier` (Adaptive Random Forest) kullanılıyor —
Hoeffding ağaçlarından oluşan bir topluluk, HER AĞACIN kendi ADWIN
(Adaptive Windowing) kavram kayması dedektörü var: bir ağacın performansı
düşerse o ağaç otomatik olarak değiştirilir. Bu, kullanıcının "Hoeffding
Ağaçları" ve "Online Random Forest" önerilerinin İKİSİNİ birden karşılar.

ÖNEMLİ FARK (XGBoost'a göre): bu model `fit(X, y)` ile toptan eğitilmez;
`learn_one(features_dict, label)` ile bar bar, SIRAYLA öğrenir — tıpkı
canlı piyasada olacağı gibi. Bu yüzden değerlendirmesi de farklıdır:
`run_prequential_evaluation` "test-then-train" (prequential) protokolünü
kullanır — her bar ÖNCE tahmin edilir (henüz o barı görmemiş haliyle),
SONRA gerçek etiketle öğrenilir. Bu, XGBoost'un tek seferlik
train/holdout ayrımından farklı ama online öğrenme literatüründe
standart, look-ahead'siz bir değerlendirme yöntemidir.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from river import forest

DEFAULT_ONLINE_MODEL_PATH = Path(__file__).parent / "artifacts" / "online_model.joblib"

_LABEL_TO_DIRECTION = {1: "long", -1: "short", 0: "neutral"}


@dataclass
class Prediction:
    direction: str
    confidence: float


class OnlineSignalModel:
    """`river.forest.ARFClassifier` sarmalayıcısı — `app.ml.model.SignalModel`
    ile BENZER bir arayüz (`predict`) sunar ama öğrenme tamamen farklıdır
    (bkz. modül docstring'i): `fit(X, y)` YOKTUR, `learn_one` vardır.
    """

    def __init__(self, n_models: int = 10, seed: int = 42) -> None:
        self.n_models = n_models
        self.seed = seed
        self._model = forest.ARFClassifier(n_models=n_models, seed=seed)
        self._is_fitted = False

    def learn_one(self, features: dict, label: int) -> None:
        self._model.learn_one(features, label)
        self._is_fitted = True

    def predict_one(self, features: dict) -> Prediction:
        proba = self._model.predict_proba_one(features)
        if not proba:
            return Prediction(direction="neutral", confidence=0.0)
        best_label = max(proba, key=proba.get)
        return Prediction(direction=_LABEL_TO_DIRECTION.get(int(best_label), "neutral"), confidence=float(proba[best_label]))

    def predict(self, feature_row: pd.Series) -> Prediction:
        return self.predict_one(feature_row.to_dict())

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Online model henüz hiç veri görmedi. Önce learn_one() çağırın.")

    def save(self, path: Path = DEFAULT_ONLINE_MODEL_PATH) -> None:
        self._require_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"n_models": self.n_models, "seed": self.seed, "model": self._model}, path)

    def load(self, path: Path = DEFAULT_ONLINE_MODEL_PATH) -> None:
        data = joblib.load(path)
        self.n_models = data["n_models"]
        self.seed = data["seed"]
        self._model = data["model"]
        self._is_fitted = True

    @classmethod
    def load_from(cls, path: Path = DEFAULT_ONLINE_MODEL_PATH) -> "OnlineSignalModel":
        model = cls()
        model.load(path)
        return model


@dataclass
class PrequentialWindowPoint:
    window_index: int
    rows: int
    accuracy: float
    balanced_accuracy: float


@dataclass
class PrequentialReport:
    rows_used: int
    overall_accuracy: float
    overall_balanced_accuracy: float
    windows: list[PrequentialWindowPoint]


def run_prequential_evaluation(
    X: pd.DataFrame, y: pd.Series, n_models: int = 10, seed: int = 42, window_size: int = 500
) -> tuple[OnlineSignalModel, PrequentialReport]:
    """"Test-then-train" (prequential) protokolüyle bir `OnlineSignalModel`
    değerlendirir: `X`/`y`'nin satırları KRONOLOJİK SIRAYLA (karıştırılmadan
    — çağıran taraf bunu garanti etmeli) işlenir; her satır için ÖNCE
    tahmin yapılır (model o satırı henüz görmeden), SONRA gerçek etiketle
    öğrenilir.

    Bu, XGBoost'un statik out-of-sample holdout'undan (`app.ml.validation`)
    FARKLI bir değerlendirmedir — modelin ZAMAN İÇİNDE nasıl adapte
    olduğunu gösterir. `windows` (varsayılan 500 barlık pencereler),
    modelin erken dönemde mi yoksa daha sonra mı iyileştiğini görmek için
    ayrı ayrı raporlanır — kavram kaymasına adaptasyonun bir göstergesi.

    Döner: (eğitilmiş model, rapor) — model, TÜM veriyi görmüş son hali
    ile döner, canlı kullanıma hazırdır.
    """
    model = OnlineSignalModel(n_models=n_models, seed=seed)
    predictions: list[int] = []
    actuals: list[int] = []

    feature_dicts = X.to_dict(orient="records")
    labels = y.to_numpy()

    for feats, label in zip(feature_dicts, labels):
        pred = model.predict_one(feats)
        pred_label = {"long": 1, "short": -1, "neutral": 0}[pred.direction]
        predictions.append(pred_label)
        actuals.append(int(label))
        model.learn_one(feats, int(label))

    predictions_arr = np.array(predictions)
    actuals_arr = np.array(actuals)

    def _balanced_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
        classes = np.unique(true)
        per_class = [float((pred[true == c] == c).mean()) for c in classes if (true == c).sum() > 0]
        return float(np.mean(per_class)) if per_class else 0.0

    overall_accuracy = float((predictions_arr == actuals_arr).mean()) if len(actuals_arr) > 0 else 0.0
    overall_balanced_accuracy = _balanced_accuracy(predictions_arr, actuals_arr)

    windows: list[PrequentialWindowPoint] = []
    n = len(actuals_arr)
    for i, start in enumerate(range(0, n, window_size)):
        end = min(start + window_size, n)
        window_pred = predictions_arr[start:end]
        window_true = actuals_arr[start:end]
        windows.append(
            PrequentialWindowPoint(
                window_index=i,
                rows=int(end - start),
                accuracy=float((window_pred == window_true).mean()) if end > start else 0.0,
                balanced_accuracy=_balanced_accuracy(window_pred, window_true),
            )
        )

    return model, PrequentialReport(
        rows_used=n, overall_accuracy=overall_accuracy, overall_balanced_accuracy=overall_balanced_accuracy, windows=windows
    )
