import logging
from dataclasses import dataclass
from typing import Literal

from app.core.config import settings
from app.exchanges.base import Exchange

from .dataset import LabelingMethod, build_training_dataset, build_training_dataset_with_time
from .meta_label import MetaLabelModel, build_meta_dataset
from .model import Algorithm, SignalModel
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
        timeframe or settings.candle_timeframe,
        lookback or settings.candle_lookback,
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
) -> TrainingResult:
    """`app.ml.validation`'daki overfitting korumalarıyla (walk-forward +
    purged/embargo CV + out-of-sample holdout) eğitim yapar (bkz. rehber
    "2.4 Overfitting"). Nihai model, holdout dışındaki tüm veriyle
    eğitilir; holdout dilimi (varsayılan: son %20) modele HİÇBİR ZAMAN
    fit() sırasında gösterilmez, yalnızca `out_of_sample` metriği için
    kullanılır.
    """
    X, y, time_frac = build_training_dataset_with_time(
        exchange,
        symbols,
        timeframe or settings.candle_timeframe,
        lookback or settings.candle_lookback,
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
        timeframe or settings.candle_timeframe,
        lookback or settings.candle_lookback,
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
