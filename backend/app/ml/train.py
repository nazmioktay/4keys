import logging

from app.core.config import settings
from app.exchanges.base import Exchange

from .dataset import build_training_dataset
from .model import SignalModel

logger = logging.getLogger(__name__)


def train_signal_model(
    exchange: Exchange,
    symbols: list[str],
    timeframe: str | None = None,
    lookback: int | None = None,
    horizon: int = 5,
    threshold_pct: float = 1.0,
) -> tuple[SignalModel, int]:
    """Verilen sembollerin geçmiş verisiyle sinyal modelini eğitir.

    Döner: (eğitilmiş model, eğitimde kullanılan satır sayısı)
    """
    X, y = build_training_dataset(
        exchange,
        symbols,
        timeframe or settings.candle_timeframe,
        lookback or settings.candle_lookback,
        horizon,
        threshold_pct,
    )

    if len(X) < 30:
        raise ValueError(
            f"Eğitim için yeterli veri yok ({len(X)} satır). Daha fazla sembol veya daha uzun geçmiş kullanın."
        )

    model = SignalModel()
    model.fit(X, y)
    model.save()
    logger.info("model trained on %d rows", len(X))
    return model, len(X)
