import logging
from dataclasses import dataclass

from app.exchanges.base import Exchange
from app.ml.features import latest_feature_vector
from app.ml.model import Prediction, SignalModel

from .positions import PaperPositionStore

logger = logging.getLogger(__name__)


@dataclass
class Action:
    symbol: str
    type: str  # "open_long" | "open_short" | "close" | "hold"
    reason: str
    price: float
    confidence: float


class DecisionEngine:
    """ML sinyalini mevcut pozisyon durumuyla birleştirip aksiyon üretir.

    Bu sınıf yalnızca karar üretir ve `PaperPositionStore` üzerinde simüle
    eder; gerçek borsaya emir göndermez. Canlı emir yürütmek isteyen bir
    katman, buradan gelen `Action` nesnelerini tüketip borsa API'sine
    dönüştürmelidir — ve bu, kullanıcının API anahtarlarını ve açık onayını
    gerektiren ayrı bir adım olmalıdır.
    """

    def __init__(
        self,
        exchange: Exchange,
        model: SignalModel,
        positions: PaperPositionStore,
        timeframe: str,
        lookback: int,
        open_confidence: float = 0.6,
        close_confidence: float = 0.55,
    ) -> None:
        self.exchange = exchange
        self.model = model
        self.positions = positions
        self.timeframe = timeframe
        self.lookback = lookback
        self.open_confidence = open_confidence
        self.close_confidence = close_confidence

    def _predict(self, symbol: str) -> tuple[Prediction, float] | None:
        ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, self.lookback)
        feature_row = latest_feature_vector(ohlcv)
        if feature_row is None:
            return None
        prediction = self.model.predict(feature_row)
        return prediction, float(feature_row["close"])

    def evaluate(self, symbol: str) -> Action | None:
        result = self._predict(symbol)
        if result is None:
            return None
        prediction, price = result
        position = self.positions.get(symbol)

        if position is None:
            if prediction.direction in ("long", "short") and prediction.confidence >= self.open_confidence:
                return Action(symbol, f"open_{prediction.direction}", "model açılış sinyali", price, prediction.confidence)
            return Action(symbol, "hold", "yeterli güven yok / nötr sinyal", price, prediction.confidence)

        opposing = (position.direction == "long" and prediction.direction == "short") or (
            position.direction == "short" and prediction.direction == "long"
        )
        if prediction.confidence >= self.close_confidence and (opposing or prediction.direction == "neutral"):
            return Action(symbol, "close", "model kapanış/ters sinyali", price, prediction.confidence)

        return Action(symbol, "hold", "pozisyon açık, sinyal değişmedi", price, prediction.confidence)

    def apply(self, action: Action) -> None:
        if action.type == "open_long":
            self.positions.open(action.symbol, "long", action.price)
        elif action.type == "open_short":
            self.positions.open(action.symbol, "short", action.price)
        elif action.type == "close":
            self.positions.close(action.symbol, action.price)

    def run_cycle(self, symbols: list[str]) -> list[Action]:
        actions: list[Action] = []
        for symbol in symbols:
            try:
                action = self.evaluate(symbol)
            except Exception as exc:  # noqa: BLE001 - tek sembol hatası döngüyü durdurmamalı
                logger.warning("engine: skipping %s: %s", symbol, exc)
                continue
            if action is None:
                continue
            self.apply(action)
            actions.append(action)
        return actions
