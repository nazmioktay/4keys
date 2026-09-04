from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score


class ModelFactory(Protocol):
    def __call__(self) -> "_FittableModel": ...


class _FittableModel(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict_batch(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class FoldResult:
    fold: int
    train_rows: int
    test_rows: int
    accuracy: float
    balanced_accuracy: float


@dataclass
class WalkForwardReport:
    folds: list[FoldResult]
    mean_accuracy: float
    mean_balanced_accuracy: float
    overfit_gap: float  # eğitim-doğruluğu - ortalama test-doğruluğu (yüksekse overfitting işareti)


@dataclass
class OutOfSampleReport:
    holdout_rows: int
    accuracy: float
    balanced_accuracy: float
    # Sınıf başına GERÇEK ve TAHMİN EDİLEN satır sayıları — `balanced_accuracy`
    # tek bir sayıya sıkıştığı için "model çoğunluk sınıfına ÇÖKTÜ mü" sorusunu
    # gizleyebiliyordu (balanced_accuracy TAM OLARAK 1/3 ise, bu modelin HER
    # ZAMAN tek bir sınıfı tahmin ettiğinin matematiksel imzasıdır — bkz.
    # session notu: XGBoost'un tekrar tekrar aldığı 0.333 sonucu). Bu alanlar
    # olmadan bu teşhis yalnızca dolaylı yapılabiliyordu.
    true_class_counts: dict[str, int] = field(default_factory=dict)
    predicted_class_counts: dict[str, int] = field(default_factory=dict)


def split_out_of_sample(
    X: pd.DataFrame, y: pd.Series, time_frac: pd.Series, holdout_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Kronolojik olarak en yeni `holdout_frac` dilimini eğitim setinden
    tamamen ayırır — bu dilim ne walk-forward CV'de ne de nihai modelin
    fit() adımında ASLA kullanılmaz. Rehberin "son 6 ay hiçbir zaman
    eğitimde kullanılmaz" kuralının karşılığıdır (mutlak takvim yerine
    her sembolün kendi serisindeki göreli `time_frac` kullanılır, çünkü
    semboller farklı miktarda geçmiş veriyle dönebilir).

    Döner: (X_train, y_train, X_holdout, y_holdout)
    """
    cutoff = 1.0 - holdout_frac
    train_mask = time_frac <= cutoff
    holdout_mask = ~train_mask
    return X[train_mask], y[train_mask], X[holdout_mask], y[holdout_mask]


def walk_forward_splits(
    time_frac: pd.Series, n_splits: int = 5, embargo_frac: float = 0.02
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward + purged/embargo bölmeleri üretir.

    Zaman ekseni (`time_frac`, 0..1) `n_splits + 1` eşit parçaya bölünür;
    her adımda model önceki tüm parçalarla eğitilir (`train`), hemen
    sonraki parçayla test edilir (`test`) — pencere ileri kaydırılır
    (walk-forward). Eğitim ve test arasına `embargo_frac` genişliğinde
    bir zaman boşluğu konur (purge/embargo): test penceresine yakın
    eğitim örnekleri, etiketleme ufku (horizon) yüzünden test verisiyle
    örtüşen bilgi sızdırabileceğinden bu boşlukta atılır.
    """
    edges = np.linspace(0.0, 1.0, n_splits + 2)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    values = time_frac.to_numpy()

    for i in range(1, len(edges) - 1):
        train_end = edges[i] - embargo_frac
        test_start, test_end = edges[i], edges[i + 1]

        train_idx = np.where(values <= train_end)[0]
        test_idx = np.where((values > test_start) & (values <= test_end))[0]

        if len(train_idx) < 30 or len(test_idx) < 5:
            continue
        splits.append((train_idx, test_idx))

    return splits


def run_walk_forward_validation(
    X: pd.DataFrame, y: pd.Series, time_frac: pd.Series, model_factory: ModelFactory, n_splits: int = 5, embargo_frac: float = 0.02
) -> WalkForwardReport:
    """Walk-forward validation çalıştırır: her fold için model sıfırdan
    eğitilir (`model_factory()` ile taze bir örnek alınır — foldlar
    arası durum sızıntısı olmaması için), performansı ayrı bir test
    penceresinde ölçülür.

    `overfit_gap`, son fold'daki eğitim-seti doğruluğu ile foldlar
    arası ortalama test doğruluğu arasındaki farktır; büyük bir boşluk
    (ör. >%20) modelin geçmişi ezberlediğinin (overfitting) klasik
    işaretidir (bkz. rehber "2.4 Overfitting").
    """
    splits = walk_forward_splits(time_frac, n_splits=n_splits, embargo_frac=embargo_frac)
    if not splits:
        raise ValueError("Walk-forward doğrulama için yeterli veri/zaman aralığı yok.")

    folds: list[FoldResult] = []
    last_train_accuracy = 0.0

    for i, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        model = model_factory()
        model.fit(X_train, y_train)

        train_pred, _ = model.predict_batch(X_train)
        test_pred, _ = model.predict_batch(X_test)

        acc = accuracy_score(y_test, test_pred)
        bal_acc = balanced_accuracy_score(y_test, test_pred)
        last_train_accuracy = accuracy_score(y_train, train_pred)

        folds.append(FoldResult(fold=i, train_rows=len(train_idx), test_rows=len(test_idx), accuracy=acc, balanced_accuracy=bal_acc))

    mean_acc = float(np.mean([f.accuracy for f in folds]))
    mean_bal_acc = float(np.mean([f.balanced_accuracy for f in folds]))

    return WalkForwardReport(
        folds=folds,
        mean_accuracy=mean_acc,
        mean_balanced_accuracy=mean_bal_acc,
        overfit_gap=last_train_accuracy - mean_acc,
    )


def evaluate_out_of_sample(model: _FittableModel, X_holdout: pd.DataFrame, y_holdout: pd.Series) -> OutOfSampleReport:
    """Eğitimde hiç görülmemiş holdout dilimi üzerinde nihai doğrulama."""
    pred, _ = model.predict_batch(X_holdout)
    true_counts = pd.Series(y_holdout).value_counts()
    pred_counts = pd.Series(pred).value_counts()
    return OutOfSampleReport(
        holdout_rows=len(X_holdout),
        accuracy=float(accuracy_score(y_holdout, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_holdout, pred)),
        true_class_counts={str(k): int(v) for k, v in true_counts.items()},
        predicted_class_counts={str(k): int(v) for k, v in pred_counts.items()},
    )
