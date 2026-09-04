from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import ALL_FEATURE_COLUMNS as FEATURE_COLUMNS

DEFAULT_MODEL_PATH = Path(__file__).parent / "artifacts" / "signal_model.joblib"


def _select_features(X: pd.DataFrame) -> pd.DataFrame:
    """`FEATURE_COLUMNS`'ı seçer; eksik kolonları (ör. makro geçmişi henüz
    kısa olduğu için NaN kalan satırlar, veya makro merge'den geçmemiş
    eski/elle kurulmuş veri) 0.0 (normalize edilmiş makro özellikler için
    "nötr" değer) ile doldurur — StandardScaler (MLP yolu) ve SHAP gibi
    NaN kabul etmeyen adımların kırılmasını önler; XGBoost zaten NaN'ı
    kendi içinde ele alabilir ama tutarlılık için aynı yol izlenir."""
    return X.reindex(columns=FEATURE_COLUMNS, fill_value=0.0).fillna(0.0)

Algorithm = Literal["xgboost", "mlp"]


@dataclass
class Prediction:
    direction: str  # "long" | "short" | "neutral"
    confidence: float  # 0..1, tahmin edilen sınıfın olasılığı


class _XGBClassifierWrapper(ClassifierMixin, BaseEstimator):
    """`XGBClassifier`'ı, keyfi sınıf etiketleriyle (ör. -1/0/1) sklearn
    Pipeline/CalibratedClassifierCV ile sorunsuz çalışacak şekilde sarar.

    XGBoost'un sınıflandırıcısı içeride 0..n_classes-1 tamsayı etiket
    bekler; burada gerçek etiketler saklanıp 0-indexli hale eşlenir,
    tahminlerde geri çevrilir. `reg_alpha`/`reg_lambda` (L1/L2) ve
    `subsample`/`colsample_bytree`, rehberin "2.4 Overfitting ->
    Regularization" maddesindeki XGBoost önerisinin karşılığıdır.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def _make_xgb(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric="mlogloss",
        )

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._label_to_idx = {label: i for i, label in enumerate(self.classes_)}
        self._idx_to_label = {i: label for label, i in self._label_to_idx.items()}
        y_idx = np.array([self._label_to_idx[v] for v in y])

        # Sınıf ağırlıklandırma: rejim-bazlı eğitimde (bkz. app.ml.regime)
        # küçültülmüş alt kümelerde balanced_accuracy'nin tam olarak 1/3'e
        # (rastgele seviye) oturduğu görüldü — LSTM'de daha önce çözülen
        # AYNI çoğunluk-sınıf çöküşü. Ters frekans ağırlıklandırma burada
        # da (dahili olarak, Pipeline/CalibratedClassifierCV'ye harici bir
        # sample_weight parametresi geçirmenin kırılganlığından kaçınmak
        # için wrapper'ın kendi fit() adımında) uygulanıyor.
        class_counts = np.bincount(y_idx, minlength=len(self.classes_)).astype("float64")
        class_counts[class_counts == 0] = 1.0
        class_weights = class_counts.sum() / (len(class_counts) * class_counts)
        sample_weight = class_weights[y_idx]

        self._model = self._make_xgb()
        self._model.fit(X, y_idx, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        idx = self._model.predict(X)
        return np.array([self._idx_to_label[int(i)] for i in idx])

    @property
    def booster_model(self) -> XGBClassifier:
        """SHAP açıklaması için altta yatan (kalibrasyondan bağımsız) fit
        edilmiş XGBoost modeline erişim."""
        return self._model


def _build_base_pipeline(algorithm: Algorithm = "xgboost", early_stopping: bool = True) -> Pipeline:
    if algorithm == "xgboost":
        return Pipeline([("xgb", _XGBClassifierWrapper())])

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
    """Tarama özelliklerinden yön (long/short/neutral) tahmini yapan model.

    Varsayılan algoritma **XGBoost** (gradient boosted karar ağaçları) —
    "Kripto Bot Tam Rehber"in önerdiği Faz A modeli: tablo verisinde hızlı
    eğitilir, L1/L2 regularization ile overfitting'e karşı korunur, SHAP
    değerleriyle yorumlanabilir (bkz. `shap_values()`). `algorithm="mlp"`
    ile eski çok katmanlı sinir ağı karşılaştırma/geriye dönük uyumluluk
    için hâlâ seçilebilir.

    `calibrate=True` (varsayılan) iken model, ham olasılık çıktısını
    doğrudan kullanmaz — `CalibratedClassifierCV` ile Platt scaling
    (`method="sigmoid"`) veya isotonic regression uygulanır. Bu önemlidir:
    kalibre edilmemiş bir "%60 güven" değeri gerçek bir olasılık değildir ve
    doğrudan Kelly kriterine (bkz. `app.portfolio.risk_manager`) verilirse
    pozisyon boyutları sistematik olarak hatalı büyür/küçülür.
    """

    def __init__(
        self,
        algorithm: Algorithm = "xgboost",
        calibrate: bool = True,
        calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
    ) -> None:
        self.algorithm = algorithm
        self._base_pipeline = _build_base_pipeline(algorithm)
        self._pipeline = self._base_pipeline
        self._calibrate = calibrate
        self._calibration_method = calibration_method
        self._calibration_cv = 3
        self._is_fitted = False
        self.is_calibrated = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_features = _select_features(X)
        class_counts = y.value_counts()
        min_class_count = int(class_counts.min()) if len(class_counts) else 0

        # MLPClassifier'ın kendi early_stopping mekanizması, iç doğrulama
        # bölmesi için sınıf başına en az birkaç örnek gerektirir; aşırı
        # dengesiz/küçük eğitim setlerinde (ör. bir sınıftan tek örnek)
        # bunu kapatmak gerekir, yoksa fit() hata fırlatır. XGBoost için
        # bu kısıt yok.
        early_stopping = min_class_count >= 5
        self._base_pipeline = _build_base_pipeline(self.algorithm, early_stopping=early_stopping)

        # CalibratedClassifierCV, StratifiedKFold(cv) kullanır; bir sınıfın örnek
        # sayısı cv'den azsa (küçük/dengesiz eğitim setlerinde olur) hata verir.
        # Bu durumda kalibrasyonu atlayıp ham (kalibre edilmemiş) modele düşülür —
        # sessizce yanlış/aşırı güvenli skorlar üretmektense bu tercih edilir.
        #
        # Eşik önceden yalnızca `min_class_count >= 3` idi — bu, cv=3 katlı
        # kalibrasyonda AZINLIK sınıfa fold başına ~1 örnek düşmesine izin
        # veriyordu. Bu kadar az örnekle fit edilen kalibrasyon eğrisi
        # gürültüyü öğrenir ve genelde azınlık sınıfın olasılığını
        # SİSTEMATİK OLARAK bastırıp modeli her zaman çoğunluk sınıfını
        # tahmin etmeye iter — `balanced_accuracy` TAM OLARAK 1/3 (3 sınıflı
        # bir problemde rastgele seviye) veren tekrarlanan XGBoost
        # çöküşlerinin araştırılmasında bulunan, olası bir katkı nedeni.
        # `sample_weight` ile eğitim (bkz. `_XGBClassifierWrapper.fit`) HAM
        # modeli dengesizliğe karşı korusa da, kalibrasyon adımı bunu
        # geri BOZABİLİR. Eşik, cv katı başına istatistiksel olarak anlamlı
        # bir örnek sayısı (>=10/kat) gerektirecek şekilde yükseltildi.
        _MIN_SAMPLES_PER_FOLD = 10
        can_calibrate = bool(
            self._calibrate and len(class_counts) >= 2 and min_class_count >= self._calibration_cv * _MIN_SAMPLES_PER_FOLD
        )

        if can_calibrate:
            calibrated = CalibratedClassifierCV(
                clone(self._base_pipeline), method=self._calibration_method, cv=self._calibration_cv
            )
            calibrated.fit(X_features, y)
            self._pipeline = calibrated
        else:
            self._pipeline = self._base_pipeline

        # SHAP açıklaması ve overfitting teşhisi (train-vs-holdout karşılaştırması)
        # için, kalibrasyon yolundan bağımsız olarak tam veriyle fit edilmiş ham
        # bir kopya her zaman tutulur (CalibratedClassifierCV içeride kendi
        # katlarını kullanır, self._base_pipeline'ı fit etmez).
        self._base_pipeline.fit(X_features, y)

        self._is_fitted = True
        self.is_calibrated = can_calibrate

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model henüz eğitilmedi. Önce fit() veya load() çağırın.")

    def predict(self, feature_row: pd.Series) -> Prediction:
        self._require_fitted()
        x = feature_row.reindex(FEATURE_COLUMNS).fillna(0.0).to_frame().T
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
        x = _select_features(X)
        proba = self._pipeline.predict_proba(x)
        classes = self._pipeline.classes_
        best_idx = np.argmax(proba, axis=1)
        predictions = classes[best_idx]
        confidences = proba[np.arange(len(proba)), best_idx]
        return predictions, confidences

    def shap_values(self, X: pd.DataFrame, max_rows: int = 200) -> pd.DataFrame:
        """Her özelliğin tahmine ortalama katkısını (mutlak SHAP değeri)
        döner — rehberin "SHAP değerleri: her feature'ın katkısı ölçülür;
        anlamsız feature'lar elenir" maddesinin karşılığı. Yalnızca
        `algorithm="xgboost"` için desteklenir (SHAP'ın asıl gücü ağaç
        modellerinde; MLP bir kara kutudur, bkz. rehber tablosu).
        """
        self._require_fitted()
        if self.algorithm != "xgboost":
            raise ValueError("SHAP açıklaması yalnızca algorithm='xgboost' için desteklenir.")

        import shap

        xgb_wrapper = self._base_pipeline.named_steps["xgb"]
        x = _select_features(X).iloc[:max_rows]
        explainer = shap.TreeExplainer(xgb_wrapper.booster_model)
        raw = explainer.shap_values(x)

        # Çok sınıflı çıktı shap sürümüne göre (n_samples, n_features, n_classes)
        # ya da (n_classes, n_samples, n_features) / sınıf başına liste olabilir;
        # her durumda özellik ekseni tespit edilip (uzunluğu FEATURE_COLUMNS
        # kadar olan eksen) örnek+sınıf üzerinden ortalama mutlak katkı alınır.
        stacked = np.asarray(raw)
        n_features = len(FEATURE_COLUMNS)
        if stacked.ndim == 3:
            feature_axis = next(ax for ax, size in enumerate(stacked.shape) if size == n_features)
            other_axes = tuple(ax for ax in range(3) if ax != feature_axis)
            mean_abs = np.abs(stacked).mean(axis=other_axes)
        else:
            mean_abs = np.abs(stacked).mean(axis=0)

        importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": mean_abs})
        return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "base_pipeline": self._base_pipeline, "algorithm": self.algorithm}, path)

    def load(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        payload = joblib.load(path)
        if isinstance(payload, dict):
            self._pipeline = payload["pipeline"]
            self._base_pipeline = payload["base_pipeline"]
            self.algorithm = payload.get("algorithm", "xgboost")
        else:
            # Geriye dönük uyumluluk: eski model dosyaları çıplak pipeline'dı (MLP).
            self._pipeline = payload
            self._base_pipeline = payload
            self.algorithm = "mlp"
        self._is_fitted = True

    @classmethod
    def load_from(cls, path: Path = DEFAULT_MODEL_PATH) -> "SignalModel":
        model = cls()
        model.load(path)
        return model
