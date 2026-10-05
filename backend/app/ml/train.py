from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.memory_probe import log_rss
from app.exchanges.base import Exchange
from app.exchanges.cache import timeframe_minutes

from .dataset import LabelingMethod, build_training_dataset, build_training_dataset_with_time
from .features import ALL_FEATURE_COLUMNS
from .meta_label import MetaLabelModel, build_meta_dataset, build_meta_dataset_out_of_fold
from .model import DEFAULT_MODEL_PATH, Algorithm, SignalModel
from .model_paths import DEFAULT_LSTM_MODEL_PATH, DEFAULT_PATCHTST_MODEL_PATH
from .model_status import get_champion_fit_end, get_split_boundary, write_model_status
from .online_model import (
    DEFAULT_ONLINE_MODEL_PATH,
    DEFAULT_ONLINE_PREHOLDOUT_MODEL_PATH,
    OnlineSignalModel,
    PrequentialReport,
    run_prequential_evaluation,
)
from .regime import RegimeModel, build_regime_labeled_dataset, fit_regime_model
from .sequence_dataset import build_sequence_dataset
from .validation import OutOfSampleReport, WalkForwardReport, evaluate_out_of_sample, run_walk_forward_validation, split_out_of_sample

# `torch` (LSTM/PatchTST) BİLEREK modül seviyesinde import edilmez — ölçüldü,
# TEK BAŞINA ~460MB'lık bir bellek maliyeti taşıyor (bkz. README). Bu modül
# `app/api/routes/ml.py` üzerinden HER API isteğinde (XGBoost eğitimi dahil)
# yükleniyor; torch'u yalnızca GERÇEKTEN bir LSTM/PatchTST modeli inşa
# edilecek fonksiyonların İÇİNDE (train_lstm_signal_model/
# train_patchtst_signal_model) import ederek bu maliyet, o fonksiyonlar
# GERÇEKTEN çağrılana kadar hiç ödenmez. Tip belirteçleri için (çalışma
# zamanında hiç çözülmez, yalnızca mypy/IDE için) TYPE_CHECKING kullanılır.
if TYPE_CHECKING:
    from .lstm_model import LSTMSignalModel, LSTMTrainingReport
    from .patchtst_model import PatchTSTSignalModel, PatchTSTTrainingReport

logger = logging.getLogger(__name__)


def _report_window_bars(report, timeframe: str) -> int:
    """Sistem backtest raporunun oynattığı pencerenin bar sayısı (period_start/period_end dahil)."""
    span = pd.Timestamp(report.period_end) - pd.Timestamp(report.period_start)
    return int(span / pd.Timedelta(minutes=timeframe_minutes(timeframe))) + 1


def _champion_challenger_verdict(
    exchange: Exchange, challenger: SignalModel, holdout_start_time
) -> tuple[bool, str | None]:
    """Yeni modeli (challenger) diskteki üretim modeliyle (champion) AYNI
    holdout penceresinde sistem backtest'iyle karşılaştırır (plan 3.4).
    İkisi de o pencereyi eğitimde görmedi: challenger holdout'u hiç görmedi,
    champion daha eski bir veriyle eğitildi. Skor: PnL / maks. drawdown.
    Champion yoksa, yüklenemiyorsa veya özellik listesi eskiyse challenger kabul edilir.

    Döner: (kabul mü, ret gerekçesi)."""
    from app.backtest.schemas import SystemBacktestRequest  # local import: backtest -> ml döngüsünü önler
    from app.backtest.system_runner import run_system_backtest
    from app.backtest.walk_forward import score_result

    if holdout_start_time is None or not DEFAULT_MODEL_PATH.exists():
        return True, None
    try:
        champion = SignalModel.load_from(DEFAULT_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - bozuk/eski champion yeni modelin önünü kesmemeli
        logger.warning("champion yüklenemedi, challenger kabul ediliyor: %s", exc)
        return True, None

    request = SystemBacktestRequest(
        symbol=settings.ml_primary_symbol,
        timeframe=settings.ml_train_timeframe,
        open_confidence=settings.live_open_confidence,
        close_confidence=settings.live_close_confidence,
        use_ensemble=False,
        use_meta_label=False,
        restrict_to_holdout=False,
    )
    def _naive(value) -> pd.Timestamp:
        stamp = pd.Timestamp(value)
        return stamp.tz_convert(None) if stamp.tzinfo is not None else stamp

    # İki model de GÖRMEDİĞİ pencerede karşılaştırılır: meydan okuyan holdout'u
    # hiç görmedi; şampiyon ise tam veriyle yeniden eğitildiyse (refit) eğitim
    # bitişine kadar görmüştür — pencere ikisinin SONRASINDA başlar.
    start = _naive(holdout_start_time)
    champion_fit_end = get_champion_fit_end(DEFAULT_MODEL_PATH)
    if champion_fit_end is not None:
        start = max(start, _naive(champion_fit_end))
    window = (start, pd.Timestamp.max)
    scores = {}
    for name, candidate in (("challenger", challenger), ("champion", champion)):
        try:
            report = run_system_backtest(exchange, candidate, None, request, persist=False, eval_window=window)
            scores[name] = (score_result(report.total_pnl_pct, report.max_drawdown_pct), report)
        except Exception as exc:  # noqa: BLE001 - ör. champion'ın özellik listesi eskimiş; karşılaştırma yapılamazsa yeni model engellenmez
            logger.warning("champion/challenger: %s değerlendirilemedi, yeni model kabul ediliyor: %s", name, exc)
            return True, None
    challenger_score, challenger_report = scores["challenger"]
    champion_score, champion_report = scores["champion"]
    # "Çok kısa pencere" PENCERENİN bar sayısına göre ölçülür, işlem sayısına göre DEĞİL:
    # işlem sayısına bakmak, az/hiç işlem yapan bir meydan okuyanı (kötü bir sonuç)
    # "karşılaştırılamadı" diye otomatik kabul ettirirdi. Pencere yeterince uzunsa
    # 0 işlem de normal skorla (PnL 0) değerlendirilir.
    window_bars = _report_window_bars(challenger_report, request.timeframe)
    if window_bars < settings.ml_champion_challenger_min_window_bars:
        logger.warning(
            "champion/challenger: karşılaştırma penceresi (%s sonrası) çok kısa — %d bar (en az %d gerekli, ~30 işlem "
            "üretebilecek uzunluk); karşılaştırma yapılamadı, challenger mutlak kalite kapısına göre kabul ediliyor.",
            start.isoformat(), window_bars, settings.ml_champion_challenger_min_window_bars,
        )
        return True, None
    logger.info(
        "champion/challenger: challenger skor=%.3f (PnL %%%.2f, DD %%%.2f, %d işlem) vs champion skor=%.3f (PnL %%%.2f, DD %%%.2f, %d işlem)",
        challenger_score, challenger_report.total_pnl_pct, challenger_report.max_drawdown_pct, challenger_report.trades_closed,
        champion_score, champion_report.total_pnl_pct, champion_report.max_drawdown_pct, champion_report.trades_closed,
    )
    if challenger_score + settings.ml_champion_challenger_tolerance < champion_score:
        return False, (
            f"champion/challenger: yeni modelin holdout sistem skoru ({challenger_score:.3f}) üretimdeki modelin "
            f"({champion_score:.3f}) altında — yeni model KAYDEDİLMEDİ, mevcut model korunuyor."
        )
    return True, None


def _production_boundary() -> pd.Timestamp | None:
    """Üretimdeki birincil modelin kayıtlı holdout sınırı (naive UTC) — yoksa `None`."""
    boundary = get_split_boundary(DEFAULT_MODEL_PATH)
    if boundary is None:
        return None
    stamp = pd.Timestamp(boundary)
    return stamp.tz_convert(None) if stamp.tzinfo is not None else stamp


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
    # Varsayılan "atr_triple_barrier": bkz. app.api.routes.ml.TrainRequest
    # aynı gerekçe — birincil (XGBoost) modelin etiketi artık gerçek ATR
    # tabanlı çıkış mantığıyla hizalı. LSTM/online/regime/meta-label
    # BİLEREK "threshold" ile bırakıldı (bu geçiş yalnızca birincil model
    # için, tek seferde tüm modelleri değiştirmek etkiyi ölçmeyi zorlaştırır).
    labeling_method: LabelingMethod = "atr_triple_barrier",
    take_profit_pct: float = 1.5,  # atr_triple_barrier'da ATR ÇARPANI
    stop_loss_pct: float = 1.5,  # atr_triple_barrier'da ATR ÇARPANI
    calibrate: bool = True,
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
    algorithm: Algorithm = "xgboost",
    holdout_frac: float = 0.2,
    walk_forward_splits: int = 5,
    embargo_frac: float = 0.02,
    persist: bool = True,
    xgb_params: dict | None = None,
    tie_neutral: bool | None = None,
    label_cost_pct: float | None = None,
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
    X, y, time_frac, bar_timestamp, symbol_col = build_training_dataset_with_time(
        exchange,
        symbols,
        timeframe or settings.ml_train_timeframe,
        lookback or settings.ml_train_lookback,
        horizon,
        threshold_pct,
        labeling_method,
        take_profit_pct,
        stop_loss_pct,
        tie_neutral=settings.ml_label_tie_neutral if tie_neutral is None else tie_neutral,
        label_cost_pct=settings.ml_label_cost_pct if label_cost_pct is None else label_cost_pct,
    )

    if len(X) < 60:
        raise ValueError(
            f"Eğitim için yeterli veri yok ({len(X)} satır). Daha fazla sembol veya daha uzun geçmiş kullanın."
        )

    X_train, y_train, X_holdout, y_holdout = split_out_of_sample(X, y, time_frac, holdout_frac)
    train_time_frac = time_frac[X_train.index]
    # KRİTİK: birden çok sembolle eğitim yapılıyorsa (bkz.
    # `app.ml.symbol_selection.select_training_symbols` — BTC + korelasyonlu
    # ek semboller), holdout'un TÜM sembollerdeki EN ERKEN tarihini almak
    # YANLIŞTIR — ikinci sembolün kendi geçmişi/DB önbellek kapsamı BTC'den
    # FARKLI (ör. daha kısa/daha eski) olabilir, bu da holdout_start_time'ı
    # yanlışlıkla ÇOK ERKEN bir tarihe çeker (gerçek üretimde gözlenen bir
    # regresyon: backtest hiçbir barı dışlamadı, çünkü kaydedilen tarih
    # backtest'in çektiği pencerenin BAŞINDAN bile ÖNCEYDİ). Backtest yalnızca
    # `settings.ml_primary_symbol`'ü (BTC) oynattığı için, holdout_start_time
    # de YALNIZCA o sembolün kendi holdout satırlarından hesaplanmalı.
    holdout_start_time = None
    if len(X_holdout) > 0:
        primary_mask = symbol_col.loc[X_holdout.index] == settings.ml_primary_symbol
        primary_holdout_timestamps = bar_timestamp.loc[X_holdout.index][primary_mask]
        if len(primary_holdout_timestamps) > 0:
            holdout_start_time = primary_holdout_timestamps.min()
        else:
            logger.warning(
                "holdout_start_time hesaplanamadı: %s için holdout satırı yok (symbols=%s)",
                settings.ml_primary_symbol,
                symbols,
            )

    def _factory() -> SignalModel:
        return SignalModel(algorithm=algorithm, calibrate=calibrate, calibration_method=calibration_method, xgb_params=xgb_params)

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
        if persist:
            write_model_status(DEFAULT_MODEL_PATH, enabled=False, balanced_accuracy=oos_report.balanced_accuracy, reason=rejection_reason)
    elif persist and settings.ml_champion_challenger_enabled and not (
        verdict := _champion_challenger_verdict(exchange, model, holdout_start_time)
    )[0]:
        # Champion korunur: durum kaydı (enabled/holdout) DEĞİŞTİRİLMEZ.
        accepted = False
        rejection_reason = verdict[1]
        logger.warning("model rejected (algorithm=%s): %s", algorithm, rejection_reason)
    elif persist:
        refit = False
        fit_end = None
        if settings.ml_refit_on_full_data and len(X_holdout) > 0:
            # Plan 3.4: doğrulama bittikten sonra dağıtılan model en yeni %20'yi
            # de görerek yeniden eğitilir. Bu durumda holdout artık "görülmemiş"
            # değildir — /backtest/system/run bunu uyarıyla bildirir; dürüst
            # ölçüm için walk-forward backtest kullanılmalı. Sınırın kendisi ve
            # eğitimin bittiği an kaydedilir ki sonraki karşılaştırmalar bu
            # şampiyonun gördüğü pencerede yapılmasın.
            model = _factory()
            model.fit(X, y)
            refit = True
            primary_stamps = bar_timestamp[symbol_col == settings.ml_primary_symbol]
            fit_end = (primary_stamps if len(primary_stamps) else bar_timestamp).max()
        model.save()
        write_model_status(
            DEFAULT_MODEL_PATH,
            enabled=True,
            balanced_accuracy=oos_report.balanced_accuracy,
            # Refit'te de sınır KORUNUR (meta/online eğitim sınırı + şampiyon
            # karşılaştırma penceresi); "görülmemiş veri" iddiası ise
            # `refit_on_full_data` ile kaldırılır (bkz. `get_holdout_start_time`).
            holdout_start_time=holdout_start_time.isoformat() if holdout_start_time is not None else None,
            fit_end_time=pd.Timestamp(fit_end).isoformat() if refit else None,
            refit_on_full_data=refit,
        )

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
    tie_neutral: bool | None = None,
    label_cost_pct: float | None = None,
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
        tie_neutral=settings.ml_label_tie_neutral if tie_neutral is None else tie_neutral,
        label_cost_pct=settings.ml_label_cost_pct if label_cost_pct is None else label_cost_pct,
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
    # Eşiği geçemeyen bir model yalnızca KAYDEDİLMEZ değil, aynı zamanda
    # canlı karar motorunda DEVRE DIŞI bırakılır (bkz. `write_model_status`)
    # — daha önce kaydedilmiş ESKİ bir dosya olsa bile o da artık
    # kullanılmaz; bir model yalnızca EN SON eğitiminde eşiği geçtiği
    # sürece aktif kalır.
    accepted = True
    rejection_reason: str | None = None
    if oos_report.holdout_rows > 0 and oos_report.balanced_accuracy < settings.ml_min_balanced_accuracy:
        accepted = False
        rejection_reason = (
            f"out_of_sample_balanced_accuracy ({oos_report.balanced_accuracy:.3f}) eşiğin "
            f"({settings.ml_min_balanced_accuracy}) altında — model KAYDEDİLMEDİ VE DEVRE DIŞI bırakıldı "
            "(eski dosyası olsa bile canlıda kullanılmayacak)."
        )
        logger.warning("%s model rejected: %s", model_kind, rejection_reason)
    elif persist:
        model.save()

    if persist:
        status_path = {"LSTM": DEFAULT_LSTM_MODEL_PATH, "PatchTST": DEFAULT_PATCHTST_MODEL_PATH}.get(model_kind)
        if status_path is not None:
            write_model_status(status_path, enabled=accepted, balanced_accuracy=oos_report.balanced_accuracy, reason=rejection_reason)

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
    tie_neutral: bool | None = None,
    label_cost_pct: float | None = None,
) -> LSTMTrainingResult:
    """LSTM (Faz B) modelini kayan pencereli sekans veri setiyle eğitir —
    bkz. `_train_sequence_model` (ortak iskelet). `persist=False`,
    üretim modelini DEĞİŞTİRMEDEN deneme yapmak için (bkz.
    `sweep_labeling_lstm`). `seed` (varsayılan 42) sonucu tekrarlanabilir
    kılar — etiketleme taramasında aynı hiperparametrelerin farklı
    çalıştırmalarda dalgalanmasının (bkz. README) nedeni buydu."""
    from .lstm_model import LSTMSignalModel  # lazy: torch yalnızca burada, gerçekten gerektiğinde yüklenir

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
        tie_neutral=tie_neutral,
        label_cost_pct=label_cost_pct,
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
    tie_neutral: bool | None = None,
    label_cost_pct: float | None = None,
) -> PatchTSTTrainingResult:
    """PatchTST'ten esinlenilmiş patch-tabanlı Transformer modelini eğitir
    (bkz. `app.ml.patchtst_model` — LSTM'e alternatif, LSTM'in BTC-only
    sınamalarda hem lookback artırma hem model küçültme ile ~%38-39
    balanced_accuracy tavanına takılı kalması üzerine eklendi). Ortak
    eğitim iskeleti için bkz. `_train_sequence_model`."""
    from .patchtst_model import PatchTSTSignalModel  # lazy: torch yalnızca burada, gerçekten gerektiğinde yüklenir

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
        tie_neutral=tie_neutral,
        label_cost_pct=label_cost_pct,
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
    tie_neutral: bool | None = None,
    label_cost_pct: float | None = None,
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
        tie_neutral=settings.ml_label_tie_neutral if tie_neutral is None else tie_neutral,
        label_cost_pct=settings.ml_label_cost_pct if label_cost_pct is None else label_cost_pct,
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
    holdout_frac: float = 0.2,
    tie_neutral: bool | None = None,
    label_cost_pct: float | None = None,
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

    Satırlar gerçek takvim sırasıyla (`bar_timestamp`) işlenir; her etiket
    ancak `horizon` bar sonra öğretilir (bkz. `run_prequential_evaluation`
    `label_delay`). Birincil modelle AYNI holdout sınırında (`holdout_frac`,
    birincil sembolün holdout başlangıcı) modelin bir kopyası
    `DEFAULT_ONLINE_PREHOLDOUT_MODEL_PATH`'e kaydedilir — sistem backtest'i
    holdout'u ölçerken o pencereyi hiç görmemiş bu kopyayı kullanır.
    """
    X, y, time_frac, bar_timestamp, symbol_col = build_training_dataset_with_time(
        exchange,
        symbols,
        timeframe or settings.ml_train_timeframe,
        lookback or settings.ml_train_lookback,
        horizon,
        threshold_pct,
        labeling_method,
        take_profit_pct,
        stop_loss_pct,
        tie_neutral=settings.ml_label_tie_neutral if tie_neutral is None else tie_neutral,
        label_cost_pct=settings.ml_label_cost_pct if label_cost_pct is None else label_cost_pct,
    )

    if len(X) < 60:
        raise ValueError(f"Online model eğitimi için yeterli veri yok ({len(X)} satır).")

    order = np.argsort(pd.to_datetime(bar_timestamp).to_numpy(), kind="stable")
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    time_frac = time_frac.iloc[order].reset_index(drop=True)
    bar_timestamp = pd.Series(pd.to_datetime(bar_timestamp).to_numpy()[order])
    symbol_col = symbol_col.iloc[order].reset_index(drop=True)

    # Sınır ÜRETİMDEKİ birincil modelin kayıtlı holdout sınırıdır (taze veriden
    # yeniden hesaplanırsa, şampiyon korunduğunda backtest'in "görülmemiş"
    # saydığı dönem öğrenilmiş olur). Kayıt yoksa eski hesap yedek olarak kalır.
    holdout_start = _production_boundary()
    if holdout_start is None:
        primary_holdout = (symbol_col == settings.ml_primary_symbol) & (time_frac > 1.0 - holdout_frac)
        if primary_holdout.any():
            holdout_start = bar_timestamp[primary_holdout].min()
    snapshot_at_row: int | None = None
    if holdout_start is not None:
        after_boundary = (bar_timestamp >= holdout_start).to_numpy()
        if after_boundary.any():
            snapshot_at_row = int(np.argmax(after_boundary))

    snapshots: list[OnlineSignalModel] = []
    model, report = run_prequential_evaluation(
        X,
        y,
        n_models=n_models,
        window_size=window_size,
        label_delay=horizon,
        snapshot_at_row=snapshot_at_row,
        on_snapshot=snapshots.append,
    )

    # Kalite kapısı — bkz. `train_signal_model_validated`'daki AYNI mantık.
    # Prequential değerlendirme her satırı işlediği için burada "holdout"
    # kavramı yok, `overall_balanced_accuracy` doğrudan kullanılır.
    if report.overall_balanced_accuracy < settings.ml_min_balanced_accuracy:
        report.accepted = False
        report.rejection_reason = (
            f"overall_balanced_accuracy ({report.overall_balanced_accuracy:.3f}) eşiğin "
            f"({settings.ml_min_balanced_accuracy}) altında — model KAYDEDİLMEDİ VE DEVRE DIŞI bırakıldı "
            "(eski dosyası olsa bile canlıda kullanılmayacak)."
        )
        logger.warning("online model rejected: %s", report.rejection_reason)
    elif persist:
        model.save()
        if snapshots:
            snapshots[0].save(DEFAULT_ONLINE_PREHOLDOUT_MODEL_PATH)
        else:
            DEFAULT_ONLINE_PREHOLDOUT_MODEL_PATH.unlink(missing_ok=True)

    if persist:
        write_model_status(
            DEFAULT_ONLINE_MODEL_PATH,
            enabled=report.accepted,
            balanced_accuracy=report.overall_balanced_accuracy,
            reason=report.rejection_reason,
        )

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
    holdout_frac: float = 0.2,
    walk_forward_splits: int = 5,
    embargo_frac: float = 0.02,
    tie_neutral: bool | None = None,
    label_cost_pct: float | None = None,
) -> tuple[MetaLabelModel, int]:
    """Birincil modelin sinyaline "gir/girme" kararı verecek meta-label
    modelini eğitir (bkz. `app.ml.meta_label`).

    SIZINTI KORUMASI: önceden meta etiketler, birincil modelin ZATEN
    EĞİTİLDİĞİ (ve backtest'in ölçtüğü holdout'u da kapsayan) TÜM satırlar
    üzerindeki tahminlerinden üretiliyordu. Artık:
    - yalnızca birincil modelle AYNI holdout sınırından (`holdout_frac`)
      ÖNCEKİ satırlar kullanılır — backtest'in test penceresi meta modele
      hiç gösterilmez;
    - birincil tahminler OUT-OF-FOLD üretilir (bkz.
      `build_meta_dataset_out_of_fold`), birincil modelin kendi
      hiperparametreleriyle her katmanda yeniden eğitilen kopyalarla;
    - hedef yalnızca yönlü tahminlerin doğruluğudur.

    `primary_model` yalnızca hiperparametre kaynağı olarak kullanılır.
    """
    X, y, time_frac, bar_timestamp, _symbol_col = build_training_dataset_with_time(
        exchange,
        symbols,
        timeframe or settings.ml_train_timeframe,
        lookback or settings.ml_train_lookback,
        horizon,
        threshold_pct,
        labeling_method,
        take_profit_pct,
        stop_loss_pct,
        tie_neutral=settings.ml_label_tie_neutral if tie_neutral is None else tie_neutral,
        label_cost_pct=settings.ml_label_cost_pct if label_cost_pct is None else label_cost_pct,
    )

    if len(X) < 30:
        raise ValueError(
            f"Meta-label eğitimi için yeterli veri yok ({len(X)} satır)."
        )

    boundary = _production_boundary()
    if boundary is not None:
        # Sınır ÜRETİMDEKİ birincil modelin kayıtlı holdout sınırıdır: taze veriden
        # yeniden hesaplanırsa, şampiyon korunduğunda (meydan okuyan reddedildiğinde)
        # backtest'in "görülmemiş" saydığı dönem meta modele öğretilmiş olur.
        stamps = pd.to_datetime(bar_timestamp)
        if stamps.dt.tz is not None:
            stamps = stamps.dt.tz_convert(None)
        pre_boundary = (stamps < boundary).to_numpy()
        X_train, y_train = X[pre_boundary], y[pre_boundary]
        if len(X_train) < 30:
            raise ValueError(
                f"Meta-label eğitimi için üretim modelinin holdout sınırından ({boundary.isoformat()}) önce "
                f"yeterli veri yok ({len(X_train)} satır)."
            )
        train_frac = time_frac[pre_boundary]
        train_time_frac = train_frac / max(float(train_frac.max()), 1e-9)
    else:
        X_train, y_train, _X_holdout, _y_holdout = split_out_of_sample(X, y, time_frac, holdout_frac)
        # `walk_forward_splits` 0..1 ekseni bekler — eğitim diliminin kendi içinde yeniden ölçeklenir.
        train_time_frac = time_frac[X_train.index] / max(1.0 - holdout_frac, 1e-9)

    def _primary_factory() -> SignalModel:
        return SignalModel(
            algorithm=primary_model.algorithm,
            calibrate=primary_model._calibrate,
            calibration_method=primary_model._calibration_method,
            xgb_params=primary_model.xgb_params,
        )

    meta_X, meta_y = build_meta_dataset_out_of_fold(
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        train_time_frac.reset_index(drop=True),
        _primary_factory,
        n_splits=walk_forward_splits,
        embargo_frac=embargo_frac,
    )

    if len(meta_y) < 30 or meta_y.nunique() < 2:
        raise ValueError(
            f"Meta-label için yeterli out-of-fold yönlü tahmin yok ({len(meta_y)} satır) ya da birincil model "
            "bu tahminlerde hep doğru/hep yanlış; meta-label modeli iki sınıf olmadan eğitilemez."
        )

    meta_model = MetaLabelModel()
    meta_model.fit(meta_X, meta_y)
    meta_model.save()
    logger.info(
        "meta-label model trained on %d out-of-fold directional rows (pre-holdout, hit rate %.3f)",
        len(meta_X),
        float(meta_y.mean()),
    )
    return meta_model, len(meta_X)


@dataclass
class TrainAllStepResult:
    step: str
    ok: bool
    detail: str


# Ensemble üyelerinin (XGBoost/LSTM/online/rejim) PAYLAŞTIĞI TEK etiketleme
# tanımı — bkz. `train_all_models` docstring'i ("KRİTİK" notu). Buradaki tek
# bir değişiklik tüm üyeleri birlikte taşır; üyelerin farklı hedefler
# öğrenmesi (ve dolayısıyla kıyaslanamaz güven/beceri değerleri üretmesi)
# yapısal olarak engellenir.
_ENSEMBLE_LABELING = {
    "labeling_method": "atr_triple_barrier",
    # GÜNCELLEME (temel sadeleşme, kullanıcı isteği — bkz. README): 12
    # bar/1.5xATR -> 3 bar/1.0xATR. Sentetik saf-gürültü kontrolüyle
    # doğrulandı: bu kombinasyon %18.7 nötr / %40-41 yönlü veriyor (eski
    # "84.7% nötr" felaketine — o da FARKLI bir etiketleme yöntemindeydi —
    # BENZEMİYOR). Hedef artık "3 mumda 1 ATR kazanım mı 1 ATR zarar mı
    # önce gelir" — daha kısa vadeli, daha sık sinyal üreten bir soru.
    #
    # GÜNCELLEME 2 (BTC-only üretim verisiyle `sweep_xgboost_labeling_targets`
    # taraması, bkz. README "Karlılık"): 3 -> 8 bar. 20 kombinasyonluk tam
    # sistem backtest taramasında horizon=3/ATR=1.0 (o zamanki varsayılan)
    # 65 işlem/%55,38 kazanma/+%1,27 PnL veriyordu — tablonun alt sıralarında.
    # horizon=8/ATR=1.0 en yüksek PnL'i verdi (145 işlem, %55,17 kazanma,
    # +%5,12 PnL, %1,46 düşüş) — hem büyük örneklem hem güçlü PnL artışı.
    #
    # GÜNCELLEME 3 (DENENDİ, GERİ ALINDI — bkz. README "Karlılık"): sweep
    # (`deploy/sweep-labeling-targets.sh`, horizon=8 sabit, yalnızca ATR
    # çarpanı taranarak) 1.25'i açık farkla en iyi nokta gösterdi (448
    # işlem, +%17,24 PnL, %60,04 kazanma). Ama bu sweep YALNIZCA birincil
    # XGBoost adayını yeniden eğitiyor — `meta_model`/`online_model`
    # diskteki ESKİ (atr=1.0 hedefiyle eğitilmiş) modellerden olduğu gibi
    # yükleniyor (bkz. `sweep_xgboost_labeling_targets` çağrısındaki
    # `_load_ensemble_models()`) — yani sweep sonucu YENİ birincil + ESKİ
    # meta/online karışımıyla ölçülmüş, tutarlı bir ensemble DEĞİL. `bash
    # deploy/train-all.sh` ile TÜM üyeler (xgboost+meta+online) atr=1.25
    # ile TUTARLI şekilde yeniden eğitilip gerçek `POST /backtest/system/run`
    # ile doğrulanınca sonuç NET biçimde KÖTÜLEŞTİ: 394 işlem, %55,33
    # kazanma, PnL +%1,724 (önceki zirve +%13,34'ten ÇOK düşük), drawdown
    # %1,621 (önceki %0,65'ten kötü). 1.0'a GERİ ALINDI. Ders: bu sweep
    # türü (yalnızca birincil modeli değiştiren), LSTM/çok-sembol
    # vakalarındaki AYNI tuzağa düşüyor — tek başına güvenilir değil, karar
    # HER ZAMAN train-all.sh + gerçek tam sistem backtest'iyle doğrulanmalı.
    "horizon": 8,  # ATR bariyerlerinde ZAMAN bariyeri (bar) — bkz. train_all_models
    "threshold_pct": 1.0,  # atr_triple_barrier'da kullanılmaz, imza uyumu için
    "take_profit_pct": 1.0,  # ATR ÇARPANI
    "stop_loss_pct": 1.0,  # ATR ÇARPANI
}


def ensemble_labeling() -> dict:
    """`_ENSEMBLE_LABELING` + etiketleme seçenekleri (`Settings.ml_label_tie_neutral`,
    `Settings.ml_label_cost_pct`, plan 3.1; varsayılan = eski davranış). Üretim
    eğitimi, walk-forward ve etiketleme taraması BU tek tanımı kullanır — böylece
    birincil model, meta-label ve online model AYNI soruyu öğrenir."""
    return {
        **_ENSEMBLE_LABELING,
        "tie_neutral": settings.ml_label_tie_neutral,
        "label_cost_pct": settings.ml_label_cost_pct,
    }


def train_all_models(
    exchange: Exchange,
    symbols: list[str],
    skip_steps: frozenset[str] = frozenset(),
    lookback: int | None = None,
) -> list[TrainAllStepResult]:
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

    `skip_steps` (ör. `{"lstm"}`) verilen adımları HİÇ ÇALIŞTIRMAZ — bkz.
    README "OOM üretim olayı": bu beş adımı TEK process'te zincirlemek,
    her adımın (özellikle LSTM'in — torch/PyTorch) bıraktığı belleğin bir
    sonrakine kümülatif olarak taşınmasına yol açıyor; küçük bellekli bir
    kutuda LSTM'i (kalite eşiğini hiç geçemiyorsa, atlamanın pratik bir
    kaybı da yoktur) atlamak, kalan adımların tamamlanma şansını belirgin
    şekilde artırabilir.

    `lookback` verilirse `settings.ml_train_lookback`'i EZER — ör. kutunun
    belleğine göre mum sayısını hızlıca deneyerek (kod/`.env` değişikliği
    olmadan) test etmek için (bkz. `python -m app.cli train-all --lookback`).

    KRİTİK — TÜM ensemble üyeleri AYNI hedefi öğrenir: XGBoost, LSTM ve
    online model `_ENSEMBLE_LABELING` ile TEK BİR etiketleme tanımını
    paylaşır. Önceden her biri FARKLI bir soruyu öğreniyordu (XGBoost:
    "12 bar içinde ATR-ölçekli hedefe mi stopa mı önce ulaşır", LSTM:
    "3 bar sonra %1 hareket eder mi", online: "5 bar sonra %1 hareket eder
    mi"). Farklı soruların cevaplarını harmanlamak sağlam bir ensemble
    değildir: güvenleri aynı ölçekte olmaz ve `DecisionEngine._skill_weight`
    FARKLI problemlerde ölçülmüş doğrulukları kıyaslar (gerçekte gözlendi:
    XGBoost'un zor/dengeli problemdeki 0.375'i, online'ın kolay/nötr-ağırlıklı
    problemdeki 0.502'siyle kıyaslanıp haksız yere düşük ağırlık aldı).
    """
    results: list[TrainAllStepResult] = []
    log_rss("train_all_models başladı")

    if "xgboost" in skip_steps:
        primary = None
        results.append(TrainAllStepResult("xgboost", True, "atlandı (skip_steps)"))
    else:
        try:
            # `horizon`, ATR-triple-barrier etiketlemesinde ZAMAN bariyeridir
            # (bir bariyere ulaşmak için kaç bar tanınır). Eski `horizon=3`
            # değeri, eski "N mum sonraki getiri" etiketlemesi için seçilmişti;
            # ATR bariyerleriyle 3 bar, 1.5xATR'lik bir hareket için ÇOK KISA —
            # örneklerin ezici çoğunluğu zaman aşımına uğrayıp NÖTR etiketleniyor
            # (üretimde gözlendi: holdout'ta %84.7 nötr), model yönlü sınıfları
            # düşük güvenle tahmin ediyor ve `open_confidence` eşiği hiç
            # aşılamıyordu ("0 işlem" backtest'i). 12 bar (1h'de yarım gün),
            # dengeli bir 3 sınıf dağılımı veriyor (üretimde doğrulandı: nötr
            # oranı %84.7 -> %17.7); gerçek dağılım her eğitimde
            # `true_class_counts` ile raporlanır.
            primary = train_signal_model_validated(exchange, symbols, lookback=lookback, **ensemble_labeling())
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
    log_rss("xgboost adımı bitti")

    if "meta_label" in skip_steps:
        results.append(TrainAllStepResult("meta_label", True, "atlandı (skip_steps)"))
    elif primary is None:
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
            # KRİTİK — bkz. _ENSEMBLE_LABELING ("KRİTİK" notu, train_all_models
            # docstring'i): meta-label "birincil DOĞRU tahmin etti mi" sorusunu
            # bu ÇAĞRIDA KULLANILAN etikete göre ölçer. `_ENSEMBLE_LABELING`
            # GEÇİLMEZSE bu fonksiyonun kendi eski varsayılanı (threshold/
            # horizon=5) kullanılır — yani "doğru" ölçütü, XGBoost'un GERÇEKTE
            # öğrendiği (atr_triple_barrier/horizon=12) soru değil, TAMAMEN
            # FARKLI bir soru olur. Bu, meta-label'ı YANLIŞ YERE PESİMİST
            # yapar (XGBoost, kendi öğrendiği sorunun cevabını doğru verse
            # bile FARKLI bir soruya göre "yanlış" sayılır) — üretimde tam
            # olarak bu yaşandı: 768 açılış girişiminin 767'si veto edildi.
            _, meta_rows = train_meta_label_model(exchange, symbols, primary.model, lookback=lookback, **ensemble_labeling())
            results.append(TrainAllStepResult("meta_label", True, f"{meta_rows} satır"))
        except ValueError as exc:
            results.append(TrainAllStepResult("meta_label", False, str(exc)))
    log_rss("meta_label adımı bitti")

    if "lstm" in skip_steps:
        results.append(TrainAllStepResult("lstm", True, "atlandı (skip_steps)"))
    else:
        try:
            lstm_result = train_lstm_signal_model(exchange, symbols, **ensemble_labeling())
            detail = f"{lstm_result.rows_used} satır, oos_balanced_acc={lstm_result.out_of_sample.balanced_accuracy:.3f}"
            if not lstm_result.accepted:
                detail = f"REDDEDİLDİ: {lstm_result.rejection_reason} ({detail})"
            results.append(TrainAllStepResult("lstm", True, detail))
        except ValueError as exc:
            results.append(TrainAllStepResult("lstm", False, str(exc)))
    log_rss("lstm adımı bitti")

    if "online" in skip_steps:
        results.append(TrainAllStepResult("online", True, "atlandı (skip_steps)"))
    else:
        try:
            _, online_report = train_online_signal_model(exchange, symbols, window_size=500, lookback=lookback, **ensemble_labeling())
            detail = f"{online_report.rows_used} satır, overall_balanced_acc={online_report.overall_balanced_accuracy:.3f}"
            if not online_report.accepted:
                detail = f"REDDEDİLDİ: {online_report.rejection_reason} ({detail})"
            results.append(TrainAllStepResult("online", True, detail))
        except ValueError as exc:
            results.append(TrainAllStepResult("online", False, str(exc)))
    log_rss("online adımı bitti")

    if "regime" in skip_steps:
        results.append(TrainAllStepResult("regime", True, "atlandı (skip_steps)"))
    else:
        try:
            _, regime_results = train_signal_models_by_regime(
                exchange, symbols, n_regimes=3, walk_forward_splits=3, **ensemble_labeling()
            )
            summary = "; ".join(
                f"rejim {r.regime}: {r.rows_used} satır" + (f" (REDDEDİLDİ: {r.error})" if r.error else "") for r in regime_results
            )
            results.append(TrainAllStepResult("regime", True, summary or "sonuç yok"))
        except ValueError as exc:
            results.append(TrainAllStepResult("regime", False, str(exc)))
    log_rss("regime adımı bitti (train_all_models tamamlandı)")

    return results
