import logging
from typing import Literal

from app.core.config import settings
from app.exchanges.base import Exchange

from .dataset import LabelingMethod, build_training_dataset
from .meta_label import MetaLabelModel, build_meta_dataset
from .model import SignalModel

logger = logging.getLogger(__name__)


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
    """Verilen sembollerin geçmiş verisiyle sinyal modelini eğitir.

    `labeling_method="triple_barrier"` ile klasik sabit-eşikli etiketleme
    yerine kâr hedefi/stop-loss/zaman aşımı bariyerlerine dayalı etiketleme
    kullanılabilir (bkz. `app.ml.labeling.triple_barrier_labels`).
    `calibrate=True` (varsayılan) modelin ham olasılık çıktısını Platt
    scaling / isotonic regression ile kalibre eder — Kelly kriterine
    (`app.portfolio`) verilecek güven skorlarının gerçek olasılığa yakın
    olması için önemlidir.

    Döner: (eğitilmiş model, eğitimde kullanılan satır sayısı)
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
            f"Eğitim için yeterli veri yok ({len(X)} satır). Daha fazla sembol veya daha uzun geçmiş kullanın."
        )

    model = SignalModel(calibrate=calibrate, calibration_method=calibration_method)
    model.fit(X, y)
    model.save()
    logger.info("model trained on %d rows (labeling=%s, calibrate=%s)", len(X), labeling_method, calibrate)
    return model, len(X)


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
