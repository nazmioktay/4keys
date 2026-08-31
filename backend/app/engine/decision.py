import logging
from dataclasses import dataclass

import pandas as pd

from app.db import repository as db
from app.exchanges.base import Exchange
from app.ml.features import latest_feature_vector
from app.ml.meta_label import MetaLabelModel
from app.ml.model import Prediction, SignalModel
from app.portfolio.manager import PortfolioManager

from .positions import PaperPositionStore

logger = logging.getLogger(__name__)


@dataclass
class Action:
    symbol: str
    type: str  # "open_long" | "open_short" | "close" | "hold" | "blocked"
    reason: str
    price: float
    confidence: float


class DecisionEngine:
    """ML sinyalini mevcut pozisyon durumuyla birleştirip aksiyon üretir.

    Bu sınıf yalnızca karar üretir; gerçek borsaya emir göndermez. Canlı emir
    yürütmek isteyen bir katman, buradan gelen `Action` nesnelerini tüketip
    borsa API'sine dönüştürmelidir — ve bu, kullanıcının API anahtarlarını ve
    açık onayını gerektiren ayrı bir adım olmalıdır.

    `portfolio` verilirse (önerilen), açılış sinyalleri ham haliyle
    uygulanmaz: `app.portfolio.manager.PortfolioManager` üzerinden risk
    kurallarına (işlem başına risk, toplam/sembol maruziyeti, eşzamanlı
    pozisyon sayısı, günlük zarar limiti) göre boyutlandırılır ve gerekirse
    reddedilir ("blocked" aksiyonu). `portfolio` verilmezse eski, sınırsız
    `PaperPositionStore` davranışına geri düşer (geriye dönük uyumluluk).

    `meta_model` verilirse (opsiyonel, bkz. `app.ml.meta_label`), birincil
    modelin açılış sinyali ham haliyle uygulanmaz: meta model bu sinyale
    "güvenilir mi" kararını verir; güvenilmezse işlem açılmaz, "hold"a
    düşülür. Bu, sabit ağırlıklı bir ensemble yerine ikinci bir modelin
    filtre görevi görmesini sağlar (Kripto Bot Rehberi Bölüm 2.5).
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
        portfolio: PortfolioManager | None = None,
        assumed_stop_loss_pct: float = 3.0,
        meta_model: MetaLabelModel | None = None,
    ) -> None:
        self.exchange = exchange
        self.model = model
        self.positions = positions
        self.timeframe = timeframe
        self.lookback = lookback
        self.open_confidence = open_confidence
        self.close_confidence = close_confidence
        self.portfolio = portfolio
        self.assumed_stop_loss_pct = assumed_stop_loss_pct
        self.meta_model = meta_model

    def _predict(self, symbol: str) -> tuple[Prediction, float, pd.Series] | None:
        ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, self.lookback)
        feature_row = latest_feature_vector(ohlcv)
        if feature_row is None:
            return None
        prediction = self.model.predict(feature_row)
        return prediction, float(feature_row["close"]), feature_row

    def evaluate(self, symbol: str) -> Action | None:
        result = self._predict(symbol)
        if result is None:
            return None
        prediction, price, feature_row = result
        db.record_signal(symbol, source="ml", direction=prediction.direction, confidence=prediction.confidence, price=price)
        position = self.portfolio.get(symbol) if self.portfolio is not None else self.positions.get(symbol)

        if position is None:
            if prediction.direction in ("long", "short") and prediction.confidence >= self.open_confidence:
                if self.meta_model is not None:
                    meta_decision = self.meta_model.decide(feature_row, prediction.confidence)
                    if not meta_decision.act:
                        return Action(
                            symbol,
                            "hold",
                            f"meta-label: sinyale güvenilmiyor (meta güven={meta_decision.confidence:.2f})",
                            price,
                            prediction.confidence,
                        )
                return Action(symbol, f"open_{prediction.direction}", "model açılış sinyali", price, prediction.confidence)
            return Action(symbol, "hold", "yeterli güven yok / nötr sinyal", price, prediction.confidence)

        opposing = (position.direction == "long" and prediction.direction == "short") or (
            position.direction == "short" and prediction.direction == "long"
        )
        if prediction.confidence >= self.close_confidence and (opposing or prediction.direction == "neutral"):
            return Action(symbol, "close", "model kapanış/ters sinyali", price, prediction.confidence)

        return Action(symbol, "hold", "pozisyon açık, sinyal değişmedi", price, prediction.confidence)

    def _open(self, symbol: str, direction: str, price: float) -> Action | None:
        if self.portfolio is None:
            self.positions.open(symbol, direction, price)
            return None

        stop_loss_price = (
            price * (1 - self.assumed_stop_loss_pct / 100)
            if direction == "long"
            else price * (1 + self.assumed_stop_loss_pct / 100)
        )
        decision = self.portfolio.propose_open(symbol, direction, price, stop_loss_price)
        if not decision.allowed or decision.size_quote <= 0:
            reason = "; ".join(decision.reasons) or "risk kuralları nedeniyle reddedildi"
            return Action(symbol, "blocked", reason, price, 0.0)

        self.portfolio.open(symbol, direction, price, decision.size_quote)
        return None

    def apply(self, action: Action) -> Action:
        if action.type == "open_long":
            blocked = self._open(action.symbol, "long", action.price)
            return blocked or action
        if action.type == "open_short":
            blocked = self._open(action.symbol, "short", action.price)
            return blocked or action
        if action.type == "close":
            if self.portfolio is not None:
                self.portfolio.close(action.symbol, action.price)
            else:
                self.positions.close(action.symbol, action.price)
        return action

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
            actions.append(self.apply(action))
        return actions
