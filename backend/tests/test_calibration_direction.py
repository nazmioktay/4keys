import numpy as np
import pandas as pd

from app.ml.features import ALL_FEATURE_COLUMNS
from app.ml.model import SignalModel


def _imbalanced_but_learnable_dataset(rng: np.random.Generator, n_majority: int, n_minority: int, separation: float = 1.2):
    """Gerçek üretim durumunu taklit eder: ezici çoğunlukta bir 'nötr'
    sınıf + ZAYIF ama öğrenilebilir bir sinyal taşıyan iki azınlık sınıf
    (long/short) — gerçek BTC verisindeki gibi net değil, örtüşen
    dağılımlar (`separation=1.2`, std=1.0 iken). `ALL_FEATURE_COLUMNS`'tan
    yalnızca 2 tanesi gerçek sinyal taşır, kalanı gürültü. Bu zorluk
    seviyesi ÖNEMLİ: çok net ayrılmış sınıflarla hem ham hem kalibre
    edilmiş model kolayca doğru tahmin eder, bug hiç görünmez — asıl
    üretim hatası ancak ZAYIF sinyal + AĞIR dengesizlikte ortaya çıkıyordu.

    `rng` çağıran taraftan alınır (train/test için PAYLAŞILAN, sürekli bir
    akış) — her çağrıda taze bir `default_rng` oluşturmak, aynı seed'in
    train/test'te farklı miktarda tüketilmesi yüzünden beklenmedik/
    tekrarlanamaz sonuçlara yol açıyordu."""
    rows: list[dict] = []
    labels: list[float] = []

    for _ in range(n_majority):
        signal = rng.normal(0, 1.0)
        row = {col: rng.normal(0, 1) for col in ALL_FEATURE_COLUMNS}
        row["rsi_norm"] = signal
        row["momentum"] = signal
        rows.append(row)
        labels.append(0.0)

    for _ in range(n_minority):
        signal = rng.normal(separation, 1.0)
        row = {col: rng.normal(0, 1) for col in ALL_FEATURE_COLUMNS}
        row["rsi_norm"] = signal
        row["momentum"] = signal
        rows.append(row)
        labels.append(1.0)

    for _ in range(n_minority):
        signal = rng.normal(-separation, 1.0)
        row = {col: rng.normal(0, 1) for col in ALL_FEATURE_COLUMNS}
        row["rsi_norm"] = signal
        row["momentum"] = signal
        rows.append(row)
        labels.append(-1.0)

    idx = rng.permutation(len(rows))
    X = pd.DataFrame(rows).iloc[idx].reset_index(drop=True)
    y = pd.Series(labels).iloc[idx].reset_index(drop=True)
    return X, y


def test_calibration_does_not_collapse_direction_to_majority_class():
    """Regresyon testi: kalibrasyon (CalibratedClassifierCV), ağır sınıf
    dengesizliği + zayıf sinyal bir arada olduğunda (üretimde XGBoost'un
    tekrar tekrar `balanced_accuracy=0.333`e — modelin holdout'ta HER ZAMAN
    tek bir sınıfı tahmin ettiğinin matematiksel imzası — çökmesine yol
    açan asıl kombinasyon) argmax'ı HER ZAMAN çoğunluk sınıfına çekiyordu.
    Bu senaryo bilinçli olarak ZOR kuruldu (bkz. `_imbalanced_but_learnable_dataset`
    docstring'i) — kolay/net ayrılmış bir senaryo hem eski hem yeni kodda
    aynı sonucu verir ve regresyonu YAKALAMAZ. `predict_batch`'in YÖNÜ
    artık ham (kalibrasyonsuz) modelden alıyor olması gerekiyor."""
    rng = np.random.default_rng(1)
    X_train, y_train = _imbalanced_but_learnable_dataset(rng, n_majority=8000, n_minority=400)
    X_test, y_test = _imbalanced_but_learnable_dataset(rng, n_majority=1600, n_minority=200)

    model = SignalModel(calibrate=True)
    model.fit(X_train, y_train)
    assert model.is_calibrated is True  # bu testin kalibrasyon yolunu gerçekten çalıştırdığından emin ol

    predictions, confidences = model.predict_batch(X_test)

    unique_predictions = set(predictions.tolist())
    assert unique_predictions != {0.0}, (
        "Model TEK BİR sınıfa (muhtemelen çoğunluk/nötr) çöktü — kalibrasyonun "
        "yön kararını maskelediği regresyon geri gelmiş olabilir."
    )

    # Azınlık sınıflarının (long/short) tahmini rastgele seviyenin (3 sınıfta
    # ~1/3) ANLAMLI ÖLÇÜDE üzerinde olmalı — tam bir ayrım beklenmiyor
    # (senaryo bilinçli olarak zor/örtüşen), ama gerçek bir sinyal öğrenildiğini
    # doğrulamak için rastgele şanstan belirgin şekilde iyi olmalı.
    minority_mask = y_test != 0.0
    minority_recall = float((predictions[minority_mask.to_numpy()] == y_test[minority_mask].to_numpy()).mean())
    assert minority_recall > 0.35

    assert ((confidences >= 0) & (confidences <= 1)).all()
