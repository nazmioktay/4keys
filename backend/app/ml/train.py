import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.core.config import settings
from app.exchanges.base import Exchange

from .dataset import LabelingMethod, build_training_dataset, build_training_dataset_with_time
from .lstm_model import LSTMSignalModel, LSTMTrainingReport
from .meta_label import MetaLabelModel, build_meta_dataset
from .model import Algorithm, SignalModel
from .sequence_dataset import build_sequence_dataset
from .validation import OutOfSampleReport, WalkForwardReport, evaluate_out_of_sample, run_walk_forward_validation, split_out_of_sample

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model: SignalModel
    rows_used: int
    walk_forward: WalkForwardReport
    out_of_sample: OutOfSampleReport


def train_signal_model(
    exchange: Exchange,
    symbols: list[str],
    timeframe: str | None = None,
    lookback: int | None = None,
    horizon: int = 5,
    threshold_pct: float = 1.0,
    labeling_method: LabelingMethod = "threshold",
    take_profit_pct: float = 2.0,
    stop_loss_pct: float = 2.0,
    calibrate: bool = True,
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
) -> tuple[SignalModel, int]:
    """Geriye dönük uyumlu basit eğitim yolu: doğrulama raporu istemeyen
    çağıranlar (ör. meta-label eğitimi) için. Yeni kod `train_signal_model_validated`'i
    tercih etmeli."""
    X, y = build_training_dataset(
        exchange,
        symbols,
        timeframe or settings.ml_train_timeframe,
        lookback or settings.ml_train_lookback,
        horizon,
        threshold_pct,
        labeling_method,
        take_profit_pct,
        stop_loss_pct,
    )

    if len(X) < 30:
        raise ValueError(
            f"Eğitim için yeterli veri yok ({len(X)} satır). Daha fazla sembol veya daha uzun geçmiş kullanın."
        )

    model = SignalModel(calibrate=calibrate, calibration_method=calibration_method)
    model.fit(X, y)
    model.save()
    logger.info("model trained on %d rows (labeling=%s, calibrate=%s)", len(X), labeling_method, calibrate)
    return model, len(X)


def train_signal_model_validated(
    exchange: Exchange,
    symbols: list[str],
    timeframe: str | None = None,
    lookback: int | None = None,
    horizon: int = 5,
    threshold_pct: float = 1.0,
    labeling_method: LabelingMethod = "threshold",
    take_profit_pct: float = 2.0,
    stop_loss_pct: float = 2.0,
    calibrate: bool = True,
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
    algorithm: Algorithm = "xgboost",
    holdout_frac: float = 0.2,
    walk_forward_splits: int = 5,
    embargo_frac: float = 0.02,
    persist: bool = True,
) -> TrainingResult:
    """`app.ml.validation`'daki overfitting korumalarıyla (walk-forward +
    purged/embargo CV + out-of-sample holdout) eğitim yapar (bkz. rehber
    "2.4 Overfitting"). Nihai model, holdout dışındaki tüm veriyle
    eğitilir; holdout dilimi (varsayılan: son %20) modele HİÇBİR ZAMAN
    fit() sırasında gösterilmez, yalnızca `out_of_sample` metriği için
    kullanılır.

    `persist=False` verilirse model diske kaydedilmez — ör. `sweep_lookback_values`
    gibi yalnızca KARŞILAŞTIRMA amaçlı, art arda birden çok deneme yapan
    çağrılarda production modelinin yanlışlıkla üzerine yazılmasını önler.
    """
    X, y, time_frac = build_training_dataset_with_time(
        exchange,
        symbols,
        timeframe or settings.ml_train_timeframe,
        lookback or settings.ml_train_lookback,
        horizon,
        threshold_pct,
        labeling_method,
        take_profit_pct,
        stop_loss_pct,
    )

    if len(X) < 60:
        raise ValueError(
            f"Eğitim için yeterli veri yok ({len(X)} satır). Daha fazla sembol veya daha uzun geçmiş kullanın."
        )

    X_train, y_train, X_holdout, y_holdout = split_out_of_sample(X, y, time_frac, holdout_frac)
    train_time_frac = time_frac[X_train.index]

    def _factory() -> SignalModel:
        return SignalModel(algorithm=algorithm, calibrate=calibrate, calibration_method=calibration_method)

    wf_report = run_walk_forward_validation(
        X_train, y_train, train_time_frac, _factory, n_splits=walk_forward_splits, embargo_frac=embargo_frac
    )

    model = _factory()
    model.fit(X_train, y_train)

    oos_report = evaluate_out_of_sample(model, X_holdout, y_holdout) if len(X_holdout) > 0 else OutOfSampleReport(0, 0.0, 0.0)

    if persist:
        model.save()
    logger.info(
        "model trained (algorithm=%s) on %d rows; walk-forward mean_acc=%.3f overfit_gap=%.3f; oos_acc=%.3f (holdout=%d rows)",
        algorithm,
        len(X_train),
        wf_report.mean_accuracy,
        wf_report.overfit_gap,
        oos_report.accuracy,
        oos_report.holdout_rows,
    )
    return TrainingResult(model=model, rows_used=len(X_train), walk_forward=wf_report, out_of_sample=oos_report)


@dataclass
class LSTMTrainingResult:
    model: LSTMSignalModel
    rows_used: int
    training: LSTMTrainingReport
    out_of_sample: OutOfSampleReport


def train_lstm_signal_model(
    exchange: Exchange,
    symbols: list[str],
    timeframe: str | None = None,
    lookback: int | None = None,
    seq_len: int = 20,
    horizon: int = 5,
    threshold_pct: float = 1.0,
    labeling_method: LabelingMethod = "threshold",
    take_profit_pct: float = 2.0,
    stop_loss_pct: float = 2.0,
    holdout_frac: float = 0.2,
    epochs: int = 30,
) -> LSTMTrainingResult:
    """LSTM (Faz B) modelini kayan pencereli sekans veri setiyle eğitir
    (bkz. `app.ml.sequence_dataset.build_sequence_dataset`).

    XGBoost'un `train_signal_model_validated`'ı gibi, kronolojik olarak en
    yeni `holdout_frac` dilimi (varsayılan son %20) fit() sırasında modele
    HİÇBİR ZAMAN gösterilmez — yalnızca out-of-sample doğrulama için
    kullanılır. Walk-forward CV, LSTM'in eğitim maliyeti (her fold için
    sıfırdan sinir ağı eğitimi) nedeniyle burada uygulanmaz; bunun yerine
    holdout + eğitim geçmişindeki (loss/accuracy) yakınsama izlenir.
    """
    X, y, time_frac = build_sequence_dataset(
        exchange,
        symbols,
        timeframe or settings.ml_train_timeframe,
        lookback or settings.ml_train_lookback,
        seq_len=seq_len,
        horizon=horizon,
        threshold_pct=threshold_pct,
        labeling_method=labeling_method,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )

    if len(X) < 60:
        raise ValueError(
            f"LSTM eğitimi için yeterli veri yok ({len(X)} pencere). Daha fazla sembol, daha uzun geçmiş veya daha kısa seq_len deneyin."
        )

    cutoff = 1.0 - holdout_frac
    train_mask = time_frac <= cutoff
    X_train, y_train = X[train_mask], y[train_mask]
    X_holdout, y_holdout = X[~train_mask], y[~train_mask]

    model = LSTMSignalModel(seq_len=seq_len)
    training_report = model.fit(X_train, y_train, epochs=epochs)

    if len(X_holdout) > 0:
        pred, _ = model.predict_batch(X_holdout)
        accuracy = float((pred == y_holdout).mean())
        # sınıf başına dengeli doğruluk (balanced accuracy) — basit ortalama
        classes = np.unique(y_holdout)
        per_class_acc = [float((pred[y_holdout == c] == c).mean()) for c in classes if (y_holdout == c).sum() > 0]
        balanced_accuracy = float(np.mean(per_class_acc)) if per_class_acc else 0.0
        oos_report = OutOfSampleReport(holdout_rows=len(X_holdout), accuracy=accuracy, balanced_accuracy=balanced_accuracy)
    else:
        oos_report = OutOfSampleReport(0, 0.0, 0.0)

    model.save()
    logger.info(
        "LSTM model trained on %d pencere; train_loss=%.4f train_acc=%.3f; oos_acc=%.3f (holdout=%d pencere)",
        len(X_train),
        training_report.final_train_loss,
        training_report.final_train_accuracy,
        oos_report.accuracy,
        oos_report.holdout_rows,
    )
    return LSTMTrainingResult(model=model, rows_used=len(X_train), training=training_report, out_of_sample=oos_report)


@dataclass
class LookbackSweepPoint:
    lookback: int
    rows_used: int
    walk_forward_mean_accuracy: float
    walk_forward_mean_balanced_accuracy: float
    overfit_gap: float
    out_of_sample_rows: int
    out_of_sample_accuracy: float
    out_of_sample_balanced_accuracy: float
    error: str | None = None


def sweep_lookback_values(
    exchange: Exchange,
    symbols: list[str],
    lookback_values: list[int],
    timeframe: str | None = None,
    horizon: int = 5,
    threshold_pct: float = 1.0,
    labeling_method: LabelingMethod = "threshold",
    take_profit_pct: float = 2.0,
    stop_loss_pct: float = 2.0,
    algorithm: Algorithm = "xgboost",
    holdout_frac: float = 0.2,
    walk_forward_splits: int = 5,
) -> list[LookbackSweepPoint]:
    """Farklı `lookback` (geçmiş derinliği) değerleriyle art arda eğitim
    yapıp her biri için walk-forward + out-of-sample metriklerini döner —
    "en küçük yeterli lookback'i bul" (rehberin overfitting/veri yeterliliği
    ilkeleriyle uyumlu bir "diminishing returns" analizi) sorusuna
    CEVAP değil, CEVABI BULMAK İÇİN VERİ sağlar: hangi noktadan sonra daha
    fazla geçmişin doğruluğu anlamlı şekilde artırmadığını (platoya
    ulaştığını) gözlemleyip seçim yapmak çağıran tarafa (bkz. `/ml/sweep-lookback`
    endpoint'i ve onu çağıran operatöre) kalır — otomatik "en iyi" seçimi
    dayatmaz çünkü "en iyi" hem doğruluk hem hesaplama maliyeti arasında bir
    değer yargısıdır.

    Her lookback bağımsız değerlendirilir; biri başarısız olursa (ör.
    yetersiz veri) `error` alanıyla işaretlenir, taramanın geri kalanı
    durmaz.
    """
    results: list[LookbackSweepPoint] = []
    for lookback in lookback_values:
        try:
            result = train_signal_model_validated(
                exchange,
                symbols,
                timeframe=timeframe,
                lookback=lookback,
                horizon=horizon,
                threshold_pct=threshold_pct,
                labeling_method=labeling_method,
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
                algorithm=algorithm,
                holdout_frac=holdout_frac,
                walk_forward_splits=walk_forward_splits,
                persist=False,
            )
            wf, oos = result.walk_forward, result.out_of_sample
            results.append(
                LookbackSweepPoint(
                    lookback=lookback,
                    rows_used=result.rows_used,
                    walk_forward_mean_accuracy=wf.mean_accuracy,
                    walk_forward_mean_balanced_accuracy=wf.mean_balanced_accuracy,
                    overfit_gap=wf.overfit_gap,
                    out_of_sample_rows=oos.holdout_rows,
                    out_of_sample_accuracy=oos.accuracy,
                    out_of_sample_balanced_accuracy=oos.balanced_accuracy,
                )
            )
        except ValueError as exc:
            results.append(
                LookbackSweepPoint(
                    lookback=lookback,
                    rows_used=0,
                    walk_forward_mean_accuracy=0.0,
                    walk_forward_mean_balanced_accuracy=0.0,
                    overfit_gap=0.0,
                    out_of_sample_rows=0,
                    out_of_sample_accuracy=0.0,
                    out_of_sample_balanced_accuracy=0.0,
                    error=str(exc),
                )
            )
    return results


def train_meta_label_model(
    exchange: Exchange,
    symbols: list[str],
    primary_model: SignalModel,
    timeframe: str | None = None,
    lookback: int | None = None,
    horizon: int = 5,
    threshold_pct: float = 1.0,
    labeling_method: LabelingMethod = "threshold",
    take_profit_pct: float = 2.0,
    stop_loss_pct: float = 2.0,
) -> tuple[MetaLabelModel, int]:
    """Birincil modelin sinyaline "gir/girme" kararı verecek meta-label
    modelini eğitir (bkz. `app.ml.meta_label`).

    Aynı eğitim setini (aynı sembol/parametrelerle) yeniden kurup birincil
    modelin bu veri üzerindeki tahminlerinin doğru/yanlış olduğunu meta
    etiket olarak kullanır. Bu nedenle `primary_model`'in bu semboller
    üzerinde zaten eğitilmiş (veya en azından aynı özellik uzayına sahip)
    olması gerekir.
    """
    X, y = build_training_dataset(
        exchange,
        symbols,
        timeframe or settings.ml_train_timeframe,
        lookback or settings.ml_train_lookback,
        horizon,
        threshold_pct,
        labeling_method,
        take_profit_pct,
        stop_loss_pct,
    )

    if len(X) < 30:
        raise ValueError(
            f"Meta-label eğitimi için yeterli veri yok ({len(X)} satır)."
        )

    meta_X, meta_y = build_meta_dataset(primary_model, X, y)

    if meta_y.nunique() < 2:
        raise ValueError(
            "Birincil model bu veri setinde ya hep doğru ya hep yanlış tahmin etmiş; "
            "meta-label modeli iki sınıf olmadan eğitilemez. Daha fazla/çeşitli veri deneyin."
        )

    meta_model = MetaLabelModel()
    meta_model.fit(meta_X, meta_y)
    meta_model.save()
    logger.info("meta-label model trained on %d rows", len(meta_X))
    return meta_model, len(meta_X)
