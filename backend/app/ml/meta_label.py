from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS
from .model import SignalModel

DEFAULT_META_MODEL_PATH = Path(__file__).parent / "artifacts" / "meta_label_model.joblib"

META_FEATURE_COLUMNS = [*FEATURE_COLUMNS, "primary_confidence"]


@dataclass
class MetaDecision:
    act: bool
    confidence: float  # meta modelin "act" kararına olan güveni (0..1)


def build_meta_dataset(primary_model: SignalModel, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Birincil modelin geçmişteki tahminlerinden meta-labeling eğitim seti kurar.

    Meta etiket: birincil model bu satırda DOĞRU tahmin ettiyse 1 ("bu
    sinyale güven, işlem aç"), yanlış tahmin ettiyse 0 ("bu sinyali atla").
    Bölüm 2.5'teki "sabit ağırlıklı ensemble yerine ikinci bir modelin
    'birincil modelin sinyaline gir/girme' kararı vermesi" yaklaşımının
    karşılığıdır.
    """
    predictions, confidences = primary_model.predict_batch(X)
    meta_y = pd.Series((predictions == y.to_numpy()).astype(int), index=X.index)

    meta_X = X[FEATURE_COLUMNS].copy()
    meta_X["primary_confidence"] = confidences
    return meta_X, meta_y


def build_meta_dataset_out_of_fold(
    X: pd.DataFrame,
    y: pd.Series,
    time_frac: pd.Series,
    primary_factory: Callable[[], SignalModel],
    n_splits: int = 5,
    embargo_frac: float = 0.02,
) -> tuple[pd.DataFrame, pd.Series]:
    """Meta-label eğitim setini birincil modelin OUT-OF-FOLD tahminlerinden
    kurar: her walk-forward katmanında birincil modelin taze bir kopyası
    yalnızca o katmandan ÖNCEKİ veriyle eğitilir, katmanın kendisini tahmin
    eder. `build_meta_dataset` ise birincil modelin ZATEN EĞİTİLDİĞİ satırlar
    üzerindeki tahminleri kullanıyordu — orada model neredeyse hep "doğru"
    göründüğü için meta model gerçek hatayı hiç öğrenemiyordu.

    Hedef yalnızca YÖNLÜ (long/short) tahminlerin doğruluğudur: meta model
    canlıda/backtest'te sadece bir pozisyon açılmadan önce sorulur, "nötr"
    tahminlerin doğru çıkması o soruya hiçbir bilgi katmaz.

    `X`/`y`/`time_frac` çağıran tarafından holdout'tan ÖNCEKİ satırlarla
    sınırlandırılmış olmalıdır (bkz. `app.ml.train.train_meta_label_model`).
    """
    from .validation import walk_forward_splits

    meta_parts: list[pd.DataFrame] = []
    label_parts: list[pd.Series] = []
    for train_idx, test_idx in walk_forward_splits(time_frac, n_splits=n_splits, embargo_frac=embargo_frac):
        y_train = y.iloc[train_idx]
        if y_train.nunique() < 2:
            continue
        model = primary_factory()
        model.fit(X.iloc[train_idx], y_train)
        X_test = X.iloc[test_idx]
        predictions, confidences = model.predict_batch(X_test)
        directional = predictions != 0
        if not directional.any():
            continue
        part = X_test.loc[directional, FEATURE_COLUMNS].copy()
        part["primary_confidence"] = np.asarray(confidences)[directional]
        meta_parts.append(part)
        label_parts.append(
            pd.Series((predictions[directional] == y.iloc[test_idx].to_numpy()[directional]).astype(int), index=part.index)
        )

    if not meta_parts:
        return pd.DataFrame(columns=META_FEATURE_COLUMNS), pd.Series(dtype=int)
    return pd.concat(meta_parts), pd.concat(label_parts)


class MetaLabelModel:
    """İkincil (meta) model: birincil modelin sinyaline "gir" mi "girme" mi
    kararını verir. Birincil modelin yönünü değiştirmez, yalnızca o sinyale
    ne kadar güvenilebileceğini filtreler."""

    def __init__(self) -> None:
        self._pipeline = self._build_pipeline(early_stopping=True)
        self._is_fitted = False

    @staticmethod
    def _build_pipeline(early_stopping: bool) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(hidden_layer_sizes=(16,), max_iter=300, random_state=42, early_stopping=early_stopping)),
            ]
        )

    def fit(self, meta_X: pd.DataFrame, meta_y: pd.Series) -> None:
        # XGBoost birincil model MLP'den çok daha isabetli olduğundan meta
        # etiketler ("doğru mu tahmin etti") ağır şekilde tek sınıfa
        # kayabilir (ör. 276 doğrudan 1 tanesi yanlış). MLPClassifier'ın
        # kendi early_stopping'i, iç doğrulama bölmesi için sınıf başına en
        # az 2 örnek gerektirir; bu durumda kapatılır (bkz. app.ml.model
        # aynı korumanın karşılığı).
        class_counts = meta_y.value_counts()
        min_class_count = int(class_counts.min()) if len(class_counts) else 0
        self._pipeline = self._build_pipeline(early_stopping=min_class_count >= 5)
        self._pipeline.fit(meta_X[META_FEATURE_COLUMNS], meta_y)
        self._is_fitted = True

    def decide(self, feature_row: pd.Series, primary_confidence: float, act_threshold: float = 0.5) -> MetaDecision:
        """`act_threshold`: "gir" (act=True) kararı için P(act=1)'in geçmesi
        gereken alt sınır. Varsayılan 0.5, ESKİ argmax davranışıyla MATEMATİKSEL
        olarak birebir aynıdır (2 sınıflı bir problemde P(1)>=0.5 <=> argmax
        sınıf 1'i seçer) — bu yüzden var olan tüm çağrı yerleri (canlı motor,
        `/ml/predict` vb.) parametre vermeden ÖNCEKİYLE AYNI şekilde çalışmaya
        devam eder. Daha yüksek bir eşik meta-label'ı daha SEÇİCİ (daha az
        işlem, teorik olarak daha güvenilir) yapar — bkz. README "Karlılık",
        `deploy/sweep-meta-label-threshold.sh` ile taranır."""
        if not self._is_fitted:
            raise RuntimeError("Meta-label modeli henüz eğitilmedi.")

        row = feature_row[FEATURE_COLUMNS].copy()
        row["primary_confidence"] = primary_confidence
        x = row[META_FEATURE_COLUMNS].to_frame().T

        proba = self._pipeline.predict_proba(x)[0]
        classes = self._pipeline.classes_
        act_class_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else None
        if act_class_idx is not None:
            act = bool(proba[act_class_idx] >= act_threshold)
            confidence = float(proba[act_class_idx]) if act else float(1.0 - proba[act_class_idx])
        else:
            # Eğitim setinde hiç "act=1" örneği görülmediyse (aşırı uç,
            # normalde olmaz) sınıf hiç yoktur — eski argmax davranışına düş.
            best_idx = int(np.argmax(proba))
            act = bool(classes[best_idx] == 1)
            confidence = float(proba[best_idx])
        return MetaDecision(act=act, confidence=confidence)

    def save(self, path: Path = DEFAULT_META_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    def load(self, path: Path = DEFAULT_META_MODEL_PATH) -> None:
        self._pipeline = joblib.load(path)
        self._is_fitted = True

    @classmethod
    def load_from(cls, path: Path = DEFAULT_META_MODEL_PATH) -> "MetaLabelModel":
        model = cls()
        model.load(path)
        return model
