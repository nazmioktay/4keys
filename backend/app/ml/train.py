import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.core.config import settings
from app.exchanges.base import Exchange

from .dataset import LabelingMethod, build_training_dataset, build_training_dataset_with_time
from .features import ALL_FEATURE_COLUMNS
from .lstm_model import LSTMSignalModel, LSTMTrainingReport
from .meta_label import MetaLabelModel, build_meta_dataset
from .model import DEFAULT_MODEL_PATH, Algorithm, SignalModel
from .online_model import OnlineSignalModel, PrequentialReport, run_prequential_evaluation
from .patchtst_model import PatchTSTSignalModel, PatchTSTTrainingReport
from .regime import RegimeModel, build_regime_labeled_dataset, fit_regime_model
from .sequence_dataset import build_sequence_dataset
from .validation import OutOfSampleReport, WalkForwardReport, evaluate_out_of_sample, run_walk_forward_validation, split_out_of_sample

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model: SignalModel
    rows_used: int
    walk_forward: WalkForwardReport
    out_of_sample: OutOfSampleReport
    accepted: bool = True
    rejection_reason: str | None = None


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

    # Kalite kapısı: holdout değerlendirmesi VARSA (0 satır ise yargılanamaz,
    # kapı uygulanmaz) ve dengeli doğruluk eşiğin (varsayılan 0.37) ALTINDAYSA
    # model diske KAYDEDİLMEZ — önceden eğitilmiş (varsa) model dosyası
    # KORUNUR, canlı karar motoru eski/iyi modeli kullanmaya devam eder.
    accepted = True
    rejection_reason: str | None = None
    if oos_report.holdout_rows > 0 and oos_report.balanced_accuracy < settings.ml_min_balanced_accuracy:
        accepted = False
        rejection_reason = (
            f"out_of_sample_balanced_accuracy ({oos_report.balanced_accuracy:.3f}) eşiğin "
            f"({settings.ml_min_balanced_accuracy}) altında — model KAYDEDİLMEDİ, önceki model (varsa) korunuyor."
        )
        logger.warning("model rejected (algorithm=%s): %s", algorithm, rejection_reason)
    elif persist:
        model.save()

    logger.info(
        "model trained (algorithm=%s) on %d rows; walk-forward mean_acc=%.3f overfit_gap=%.3f; oos_acc=%.3f (holdout=%d rows); accepted=%s",
        algorithm,
        len(X_train),
        wf_report.mean_accuracy,
        wf_report.overfit_gap,
        oos_report.accuracy,
        oos_report.holdout_rows,
        accepted,
    )
    return TrainingResult(
        model=model,
        rows_used=len(X_train),
        walk_forward=wf_report,
        out_of_sample=oos_report,
        accepted=accepted,
        rejection_reason=rejection_reason,
    )


@dataclass
class LSTMTrainingResult:
    model: LSTMSignalModel
    rows_used: int
    training: LSTMTrainingReport
    out_of_sample: OutOfSampleReport
    accepted: bool = True
    rejection_reason: str | None = None


def _train_sequence_model(
    model,
    exchange: Exchange,
    symbols: list[str],
    timeframe: str | None,
    lookback: int | None,
    seq_len: int,
    horizon: int,
    threshold_pct: float,
    labeling_method: LabelingMethod,
    take_profit_pct: float,
    stop_loss_pct: float,
    holdout_frac: float,
    val_frac: float,
    epochs: int,
    patience: int,
    feature_columns: list[str] | None,
    model_kind: str,
    persist: bool = True,
    seed: int | None = 42,
):
    """LSTM/PatchTST gibi sekans modellerinin ortak eğitim iskeleti —
    veri kurma, holdout/doğrulama bölme, erken durdurma ile fit ve
    out-of-sample raporlama. `app.ml.lstm_model.LSTMSignalModel` ve
    `app.ml.patchtst_model.PatchTSTSignalModel` AYNI `fit`/`predict_batch`/
    `save` arayüzünü paylaştığı için burada ortaklaştırıldı (bkz. o
    dosyalardaki fit() docstring'leri — erken durdurma/gradyan
    kırpma/sınıf ağırlıklandırma mantığı ikisinde de aynı).

    XGBoost'un `train_signal_model_validated`'ı gibi, kronolojik olarak en
    yeni `holdout_frac` dilimi (varsayılan son %20) fit() sırasında modele
    HİÇBİR ZAMAN gösterilmez — yalnızca out-of-sample doğrulama için
    kullanılır. Walk-forward CV burada uygulanmaz (her fold sıfırdan bir
    sinir ağı eğitimi gerektirir, maliyetli); bunun yerine kalan eğitim
    biriminin İÇİNDEN (holdout'a dokunmadan) kronolojik olarak en yeni
    `val_frac` dilimi bir doğrulama seti olarak ayrılır ve erken durdurma
    için kullanılır.

    Gerçekte kullanılan özellik listesi (`feature_columns` verilmezse
    `ALL_FEATURE_COLUMNS`) modele kaydedilir (`model.feature_columns`) —
    tahmin sırasında (`/ml/predict-lstm`/`predict-patchtst`) AYNI liste
    kullanılmalı; aksi halde özellik sayısı/sırası tutarsızlığı çalışma
    zamanı hatasına yol açar.
    """
    resolved_columns = feature_columns or ALL_FEATURE_COLUMNS
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
        feature_columns=resolved_columns,
    )

    if len(X) < 60:
        raise ValueError(
            f"{model_kind} eğitimi için yeterli veri yok ({len(X)} pencere). Daha fazla sembol, daha uzun geçmiş veya daha kısa seq_len deneyin."
        )

    cutoff = 1.0 - holdout_frac
    train_mask = time_frac <= cutoff
    X_train_full, y_train_full = X[train_mask], y[train_mask]
    X_holdout, y_holdout = X[~train_mask], y[~train_mask]

    # Doğrulama dilimi, eğitim biriminin İÇİNDEN (holdout'tan tamamen ayrı)
    # kronolojik olarak en yeni `val_frac` payı — erken durdurma bu dilimi
    # görür ama gerçek out-of-sample metriği yalnızca holdout'tan hesaplanır.
    train_time_frac = time_frac[train_mask]
    val_cutoff = np.quantile(train_time_frac, 1.0 - val_frac) if len(train_time_frac) > 0 else 1.0
    fit_mask = train_time_frac <= val_cutoff
    X_fit, y_fit = X_train_full[fit_mask], y_train_full[fit_mask]
    X_val, y_val = X_train_full[~fit_mask], y_train_full[~fit_mask]

    model.feature_columns = resolved_columns
    training_report = model.fit(X_fit, y_fit, epochs=epochs, X_val=X_val, y_val=y_val, patience=patience, seed=seed)
    X_train, y_train = X_train_full, y_train_full

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

    # Kalite kapısı — bkz. `train_signal_model_validated`'daki AYNI mantık.
    accepted = True
    rejection_reason: str | None = None
    if oos_report.holdout_rows > 0 and oos_report.balanced_accuracy < settings.ml_min_balanced_accuracy:
        accepted = False
        rejection_reason = (
            f"out_of_sample_balanced_accuracy ({oos_report.balanced_accuracy:.3f}) eşiğin "
            f"({settings.ml_min_balanced_accuracy}) altında — model KAYDEDİLMEDİ, önceki model (varsa) korunuyor."
        )
        logger.warning("%s model rejected: %s", model_kind, rejection_reason)
    elif persist:
        model.save()

    logger.info(
        "%s model trained on %d pencere; train_loss=%.4f train_acc=%.3f; oos_acc=%.3f (holdout=%d pencere); accepted=%s",
        model_kind,
        len(X_train),
        training_report.final_train_loss,
        training_report.final_train_accuracy,
        oos_report.accuracy,
        oos_report.holdout_rows,
        accepted,
    )
    return model, len(X_train), training_report, oos_report, accepted, rejection_reason


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
    val_frac: float = 0.15,
    epochs: int = 30,
    patience: int = 5,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    feature_columns: list[str] | None = None,
    persist: bool = True,
    seed: int | None = 42,
) -> LSTMTrainingResult:
    """LSTM (Faz B) modelini kayan pencereli sekans veri setiyle eğitir —
    bkz. `_train_sequence_model` (ortak iskelet). `persist=False`,
    üretim modelini DEĞİŞTİRMEDEN deneme yapmak için (bkz.
    `sweep_labeling_lstm`). `seed` (varsayılan 42) sonucu tekrarlanabilir
    kılar — etiketleme taramasında aynı hiperparametrelerin farklı
    çalıştırmalarda dalgalanmasının (bkz. README) nedeni buydu."""
    model = LSTMSignalModel(seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    model, rows_used, training_report, oos_report, accepted, rejection_reason = _train_sequence_model(
        model,
        exchange,
        symbols,
        timeframe,
        lookback,
        seq_len,
        horizon,
        threshold_pct,
        labeling_method,
        take_profit_pct,
        stop_loss_pct,
        holdout_frac,
        val_frac,
        epochs,
        patience,
        feature_columns,
        "LSTM",
        persist=persist,
        seed=seed,
    )
    return LSTMTrainingResult(
        model=model,
        rows_used=rows_used,
        training=training_report,
        out_of_sample=oos_report,
        accepted=accepted,
        rejection_reason=rejection_reason,
    )


@dataclass
class PatchTSTTrainingResult:
    model: PatchTSTSignalModel
    rows_used: int
    training: PatchTSTTrainingReport
    out_of_sample: OutOfSampleReport
    accepted: bool = True
    rejection_reason: str | None = None


def train_patchtst_signal_model(
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
    val_frac: float = 0.15,
    epochs: int = 30,
    patience: int = 5,
    patch_len: int = 5,
    stride: int = 5,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.3,
    feature_columns: list[str] | None = None,
    persist: bool = True,
    seed: int | None = 42,
) -> PatchTSTTrainingResult:
    """PatchTST'ten esinlenilmiş patch-tabanlı Transformer modelini eğitir
    (bkz. `app.ml.patchtst_model` — LSTM'e alternatif, LSTM'in BTC-only
    sınamalarda hem lookback artırma hem model küçültme ile ~%38-39
    balanced_accuracy tavanına takılı kalması üzerine eklendi). Ortak
    eğitim iskeleti için bkz. `_train_sequence_model`."""
    model = PatchTSTSignalModel(
        seq_len=seq_len, patch_len=patch_len, stride=stride, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout
    )
    model, rows_used, training_report, oos_report, accepted, rejection_reason = _train_sequence_model(
        model,
        exchange,
        symbols,
        timeframe,
        lookback,
        seq_len,
        horizon,
        threshold_pct,
        labeling_method,
        take_profit_pct,
        stop_loss_pct,
        holdout_frac,
        val_frac,
        epochs,
        patience,
        feature_columns,
        "PatchTST",
        persist=persist,
        seed=seed,
    )
    return PatchTSTTrainingResult(
        model=model,
        rows_used=rows_used,
        training=training_report,
        out_of_sample=oos_report,
        accepted=accepted,
        rejection_reason=rejection_reason,
    )


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


@dataclass
class LabelingSweepPoint:
    horizon: int
    threshold_pct: float
    rows_used: int
    final_train_accuracy: float
    out_of_sample_rows: int
    out_of_sample_accuracy: float
    out_of_sample_balanced_accuracy: float
    error: str | None = None


def sweep_labeling_lstm(
    exchange: Exchange,
    symbols: list[str],
    horizon_values: list[int],
    threshold_pct_values: list[float],
    timeframe: str | None = None,
    lookback: int | None = None,
    seq_len: int = 20,
    holdout_frac: float = 0.2,
    val_frac: float = 0.15,
    epochs: int = 30,
    patience: int = 5,
) -> list[LabelingSweepPoint]:
    """Farklı (`horizon`, `threshold_pct`) kombinasyonlarıyla LSTM'i art
    arda eğitip out-of-sample sonuçlarını döner (üretim modelini
    DEĞİŞTİRMEZ, `persist=False`).

    Neden: BTC-only sınamalarda lookback artırma, model kapasitesini
    değiştirme VE mimari değiştirme (LSTM->PatchTST) hiçbiri
    ~%38-39 balanced_accuracy tavanını aşamadı — dört bağımsız denemenin
    aynı noktada tıkanması, sınırlayıcı faktörün muhtemelen etiketleme
    (sabit `horizon`/`threshold_pct` piyasa gürültüsüne göre kötü
    kalibre olabilir) olduğuna işaret ediyor. Bu tarama o hipotezi
    doğrudan test eder.

    Her kombinasyon bağımsız değerlendirilir; biri başarısız olursa
    `error` alanıyla işaretlenir, taramanın geri kalanı durmaz.
    """
    results: list[LabelingSweepPoint] = []
    for horizon in horizon_values:
        for threshold_pct in threshold_pct_values:
            try:
                result = train_lstm_signal_model(
                    exchange,
                    symbols,
                    timeframe=timeframe,
                    lookback=lookback,
                    seq_len=seq_len,
                    horizon=horizon,
                    threshold_pct=threshold_pct,
                    holdout_frac=holdout_frac,
                    val_frac=val_frac,
                    epochs=epochs,
                    patience=patience,
                    persist=False,
                )
                oos = result.out_of_sample
                results.append(
                    LabelingSweepPoint(
                        horizon=horizon,
                        threshold_pct=threshold_pct,
                        rows_used=result.rows_used,
                        final_train_accuracy=result.training.final_train_accuracy,
                        out_of_sample_rows=oos.holdout_rows,
                        out_of_sample_accuracy=oos.accuracy,
                        out_of_sample_balanced_accuracy=oos.balanced_accuracy,
                    )
                )
            except ValueError as exc:
                results.append(
                    LabelingSweepPoint(
                        horizon=horizon,
                        threshold_pct=threshold_pct,
                        rows_used=0,
                        final_train_accuracy=0.0,
                        out_of_sample_rows=0,
                        out_of_sample_accuracy=0.0,
                        out_of_sample_balanced_accuracy=0.0,
                        error=str(exc),
                    )
                )
    return results


@dataclass
class RegimeTrainingResult:
    regime: int
    samples: int
    rows_used: int
    mean_volatility: float
    mean_trend: float
    walk_forward_mean_accuracy: float
    walk_forward_mean_balanced_accuracy: float
    overfit_gap: float
    out_of_sample_rows: int
    out_of_sample_accuracy: float
    out_of_sample_balanced_accuracy: float
    error: str | None = None


def train_signal_models_by_regime(
    exchange: Exchange,
    symbols: list[str],
    n_regimes: int = 3,
    timeframe: str | None = None,
    lookback: int | None = None,
    horizon: int = 5,
    threshold_pct: float = 1.0,
    labeling_method: LabelingMethod = "threshold",
    take_profit_pct: float = 2.0,
    stop_loss_pct: float = 2.0,
    holdout_frac: float = 0.2,
    walk_forward_splits: int = 5,
    persist: bool = True,
) -> tuple[RegimeModel, list[RegimeTrainingResult]]:
    """Hibrit rejim+ML yaklaşımı (kullanıcı önerisi): önce piyasa rejimini
    (volatilite+trend uzayında GMM ile, bkz. `app.ml.regime`) tespit eden
    paylaşılan bir model eğitilir; ardından HER REJİM İÇİN AYRI bir
    XGBoost modeli eğitilir (yalnızca o rejime ait satırlarla) —
    "her rejimde uzmanlaşma" fikri.

    Tek bir global XGBoost modeliyle (`train_signal_model_validated`)
    AYNI overfitting korumaları (walk-forward + out-of-sample holdout)
    her rejim için AYRI AYRI uygulanır. Bir rejimin örnek sayısı
    yetersizse (`<60`) o rejim `error` alanıyla işaretlenir, diğer
    rejimlerin eğitimi durmaz.

    `persist=True` ise rejim modeli (`RegimeModel.save`) ve her rejimin
    XGBoost modeli ayrı dosyalara (`model_regime_<r>.joblib`) kaydedilir
    — canlı karar motoruna henüz BAĞLANMADI (bkz. README); bu fonksiyon
    şimdilik yalnızca "rejime göre ayırmak tek global modelden daha mı
    iyi?" sorusuna OFFLINE veri sağlar.
    """
    regime_model, summaries = fit_regime_model(
        exchange, symbols, timeframe or settings.ml_train_timeframe, lookback or settings.ml_train_lookback, n_regimes=n_regimes
    )
    X, y, regime, time_frac = build_regime_labeled_dataset(
        exchange,
        symbols,
        timeframe or settings.ml_train_timeframe,
        lookback or settings.ml_train_lookback,
        regime_model,
        horizon=horizon,
        threshold_pct=threshold_pct,
        labeling_method=labeling_method,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )

    results: list[RegimeTrainingResult] = []
    for summary in summaries:
        r = summary.regime
        mask = (regime == r).to_numpy()
        X_r, y_r, time_frac_r = X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True), time_frac.loc[mask].reset_index(drop=True)

        if len(X_r) < 60:
            results.append(
                RegimeTrainingResult(
                    regime=r,
                    samples=summary.samples,
                    rows_used=len(X_r),
                    mean_volatility=summary.mean_volatility,
                    mean_trend=summary.mean_trend,
                    walk_forward_mean_accuracy=0.0,
                    walk_forward_mean_balanced_accuracy=0.0,
                    overfit_gap=0.0,
                    out_of_sample_rows=0,
                    out_of_sample_accuracy=0.0,
                    out_of_sample_balanced_accuracy=0.0,
                    error=f"yetersiz veri ({len(X_r)} satır)",
                )
            )
            continue

        try:
            X_train, y_train, X_holdout, y_holdout = split_out_of_sample(X_r, y_r, time_frac_r, holdout_frac)
            train_time_frac = time_frac_r[X_train.index]

            def _factory() -> SignalModel:
                return SignalModel()

            wf_report = run_walk_forward_validation(X_train, y_train, train_time_frac, _factory, n_splits=walk_forward_splits)

            model = _factory()
            model.fit(X_train, y_train)
            oos_report = evaluate_out_of_sample(model, X_holdout, y_holdout) if len(X_holdout) > 0 else OutOfSampleReport(0, 0.0, 0.0)

            # Kalite kapısı — bkz. `train_signal_model_validated`'daki AYNI
            # mantık, rejim başına uygulanır: bir rejimin modeli eşiğin
            # altındaysa YALNIZCA O REJİMİN dosyası kaydedilmez, diğer
            # rejimler etkilenmez.
            regime_error: str | None = None
            if oos_report.holdout_rows > 0 and oos_report.balanced_accuracy < settings.ml_min_balanced_accuracy:
                regime_error = (
                    f"out_of_sample_balanced_accuracy ({oos_report.balanced_accuracy:.3f}) eşiğin "
                    f"({settings.ml_min_balanced_accuracy}) altında — model KAYDEDİLMEDİ."
                )
                logger.warning("regime %d model rejected: %s", r, regime_error)
            elif persist:
                model.save(DEFAULT_MODEL_PATH.parent / f"model_regime_{r}.joblib")

            results.append(
                RegimeTrainingResult(
                    regime=r,
                    samples=summary.samples,
                    rows_used=len(X_train),
                    mean_volatility=summary.mean_volatility,
                    mean_trend=summary.mean_trend,
                    walk_forward_mean_accuracy=wf_report.mean_accuracy,
                    walk_forward_mean_balanced_accuracy=wf_report.mean_balanced_accuracy,
                    overfit_gap=wf_report.overfit_gap,
                    out_of_sample_rows=oos_report.holdout_rows,
                    out_of_sample_accuracy=oos_report.accuracy,
                    out_of_sample_balanced_accuracy=oos_report.balanced_accuracy,
                    error=regime_error,
                )
            )
        except ValueError as exc:
            results.append(
                RegimeTrainingResult(
                    regime=r,
                    samples=summary.samples,
                    rows_used=len(X_r),
                    mean_volatility=summary.mean_volatility,
                    mean_trend=summary.mean_trend,
                    walk_forward_mean_accuracy=0.0,
                    walk_forward_mean_balanced_accuracy=0.0,
                    overfit_gap=0.0,
                    out_of_sample_rows=0,
                    out_of_sample_accuracy=0.0,
                    out_of_sample_balanced_accuracy=0.0,
                    error=str(exc),
                )
            )

    if persist:
        regime_model.save()

    return regime_model, results


def train_online_signal_model(
    exchange: Exchange,
    symbols: list[str],
    timeframe: str | None = None,
    lookback: int | None = None,
    horizon: int = 5,
    threshold_pct: float = 1.0,
    labeling_method: LabelingMethod = "threshold",
    take_profit_pct: float = 2.0,
    stop_loss_pct: float = 2.0,
    n_models: int = 10,
    window_size: int = 500,
    persist: bool = True,
) -> tuple[OnlineSignalModel, PrequentialReport]:
    """Kullanıcı önerisi: XGBoost/LSTM'in periyodik toptan (batch)
    yeniden eğitimi yerine, verinin akışından ANLIK öğrenen bir model
    (`river.forest.ARFClassifier` — Hoeffding ağaçlarından oluşan,
    kendi kavram kayması tespitine sahip bir topluluk, bkz.
    `app.ml.online_model` docstring'i).

    `build_training_dataset_with_time` ile AYNI özellik/etiketleme işlem
    hattı kullanılır (XGBoost ile adil karşılaştırma için), ama eğitim
    `fit(X, y)` DEĞİL, `run_prequential_evaluation` ile bar-bar
    test-then-train'dir — bkz. o fonksiyonun docstring'i.

    NOT: Birden fazla sembol verilirse, semboller `time_frac`'e göre değil
    `_build_symbol_frames`'in sırasına göre ART ARDA (önce tüm A sembolü,
    sonra tüm B sembolü) işlenir — her sembolün KENDİ içinde kronolojik
    sıra korunur, ama semboller arası GERÇEK takvim sırası değildir. BTC-only
    (veya BTC-öncelikli az sayıda sembol) kullanmak bu basitleştirmeyi
    önemsiz kılar.
    """
    X, y, _time_frac = build_training_dataset_with_time(
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
        raise ValueError(f"Online model eğitimi için yeterli veri yok ({len(X)} satır).")

    model, report = run_prequential_evaluation(X, y, n_models=n_models, window_size=window_size)

    # Kalite kapısı — bkz. `train_signal_model_validated`'daki AYNI mantık.
    # Prequential değerlendirme her satırı işlediği için burada "holdout"
    # kavramı yok, `overall_balanced_accuracy` doğrudan kullanılır.
    if report.overall_balanced_accuracy < settings.ml_min_balanced_accuracy:
        report.accepted = False
        report.rejection_reason = (
            f"overall_balanced_accuracy ({report.overall_balanced_accuracy:.3f}) eşiğin "
            f"({settings.ml_min_balanced_accuracy}) altında — model KAYDEDİLMEDİ, önceki model (varsa) korunuyor."
        )
        logger.warning("online model rejected: %s", report.rejection_reason)
    elif persist:
        model.save()

    logger.info(
        "online model (river ARF) trained on %d rows (prequential); overall_acc=%.3f overall_balanced_acc=%.3f; accepted=%s",
        report.rows_used,
        report.overall_accuracy,
        report.overall_balanced_accuracy,
        report.accepted,
    )
    return model, report


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


@dataclass
class TrainAllStepResult:
    step: str
    ok: bool
    detail: str


def train_all_models(exchange: Exchange, symbols: list[str]) -> list[TrainAllStepResult]:
    """Tüm modelleri (XGBoost -> meta-label -> LSTM -> online -> regime)
    sırayla, deploy script'lerinde (`deploy/train-xgboost-best-labeling.sh`,
    `train-meta.sh`, `train-lstm-btc-best-labeling.sh`, `train-online-btc.sh`,
    `train-regime-multi.sh`) DOĞRULANMIŞ AYNI parametrelerle eğitir — ilk
    kurulumda (henüz hiçbir model yokken) veya toplu bir yeniden eğitim
    istendiğinde, kullanıcının bu 5 script'i tek tek elle çalıştırması
    YERİNE `POST /ml/train-all` ile tek çağrıda tetiklenebilir.

    Her adım BAĞIMSIZ try/except ile sarılır (bkz. `app.scheduler.jobs`
    aynı desen) — bir adımın hatası (ör. yetersiz veri) sonraki adımların
    çalışmasını ENGELLEMEZ; her adımın sonucu ayrı ayrı raporlanır.
    LSTM sırasıyla en yavaş adımdır, toplam çalışma süresi birkaç dakikayı
    bulabilir.
    """
    results: list[TrainAllStepResult] = []

    try:
        primary = train_signal_model_validated(exchange, symbols, horizon=3, threshold_pct=1.0)
        detail = (
            f"{primary.rows_used} satır, oos_balanced_acc={primary.out_of_sample.balanced_accuracy:.3f}, "
            f"gerçek={primary.out_of_sample.true_class_counts}, "
            f"tahmin={primary.out_of_sample.predicted_class_counts}"
        )
        if not primary.accepted:
            detail = f"REDDEDİLDİ: {primary.rejection_reason} ({detail})"
        results.append(TrainAllStepResult("xgboost", True, detail))
    except ValueError as exc:
        primary = None
        results.append(TrainAllStepResult("xgboost", False, str(exc)))

    if primary is None:
        # Meta-label birincil modele bağımlı olduğundan, birincil model hiç
        # eğitilemediyse meta-label'ı denemek anlamsız.
        results.append(TrainAllStepResult("meta_label", False, "atlandı: birincil model eğitilemedi"))
    elif not primary.accepted:
        # Birincil model kalite kapısından geçemedi (KAYDEDİLMEDİ) — meta-label'ı
        # bu REDDEDİLEN model üzerinde eğitmek, canlıda kullanılan (eski,
        # kaydedilmiş) modelle TUTARSIZ bir meta-label üretirdi.
        results.append(TrainAllStepResult("meta_label", False, "atlandı: birincil model reddedildi (kalite eşiğinin altında)"))
    else:
        try:
            _, meta_rows = train_meta_label_model(exchange, symbols, primary.model)
            results.append(TrainAllStepResult("meta_label", True, f"{meta_rows} satır"))
        except ValueError as exc:
            results.append(TrainAllStepResult("meta_label", False, str(exc)))

    try:
        lstm_result = train_lstm_signal_model(exchange, symbols, horizon=3, threshold_pct=1.0)
        detail = f"{lstm_result.rows_used} satır, oos_balanced_acc={lstm_result.out_of_sample.balanced_accuracy:.3f}"
        if not lstm_result.accepted:
            detail = f"REDDEDİLDİ: {lstm_result.rejection_reason} ({detail})"
        results.append(TrainAllStepResult("lstm", True, detail))
    except ValueError as exc:
        results.append(TrainAllStepResult("lstm", False, str(exc)))

    try:
        _, online_report = train_online_signal_model(exchange, symbols, window_size=500)
        detail = f"{online_report.rows_used} satır, overall_balanced_acc={online_report.overall_balanced_accuracy:.3f}"
        if not online_report.accepted:
            detail = f"REDDEDİLDİ: {online_report.rejection_reason} ({detail})"
        results.append(TrainAllStepResult("online", True, detail))
    except ValueError as exc:
        results.append(TrainAllStepResult("online", False, str(exc)))

    try:
        _, regime_results = train_signal_models_by_regime(
            exchange, symbols, n_regimes=3, walk_forward_splits=3, horizon=3, threshold_pct=1.0
        )
        summary = "; ".join(
            f"rejim {r.regime}: {r.rows_used} satır" + (f" (REDDEDİLDİ: {r.error})" if r.error else "") for r in regime_results
        )
        results.append(TrainAllStepResult("regime", True, summary or "sonuç yok"))
    except ValueError as exc:
        results.append(TrainAllStepResult("regime", False, str(exc)))

    return results
