import logging
from dataclasses import dataclass

import pandas as pd

from app.db import repository as db
from app.exchanges.base import Exchange
from app.ml.features import latest_feature_vector
from app.ml.lstm_model import LSTMSignalModel
from app.ml.macro_features import latest_macro_feature_row
from app.monitoring.metrics import record_ml_prediction
from app.ml.meta_label import MetaLabelModel
from app.ml.model import Prediction, SignalModel
from app.ml.orderbook_features import latest_orderbook_feature_row
from app.ml.sequence_dataset import latest_sequence_window
from app.portfolio.manager import PortfolioManager
from app.security import kill_switch

from .positions import PaperPositionStore

logger = logging.getLogger(__name__)


@dataclass
class Action:
    symbol: str
    type: str  # "open_long" | "open_short" | "add_entry_tranche" | "close" | "close_partial" | "hold" | "blocked"
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
    (XGBoost, veya XGBoost+LSTM ensemble'ı) sinyali ham haliyle
    uygulanmaz: meta model bu sinyale "güvenilir mi" kararını verir;
    güvenilmezse işlem açılmaz, "hold"a düşülür.

    `lstm_model` verilirse (opsiyonel, bkz. `app.ml.lstm_model`), XGBoost'un
    tahmini TEK BAŞINA kullanılmaz — basit, kural tabanlı bir ensemble
    (`_combine_predictions`) ile LSTM'in tahminiyle birleştirilir: ikisi
    AYNI yönü işaret ediyorsa güven artırılır (iki bağımsız modelin
    mutabakatı); biri nötr diğeri yönlüyse yönlü olan indirimli güvenle
    kullanılır; ZIT yönleri işaret ediyorlarsa (biri long biri short)
    belirsizlik nedeniyle nötre düşülür. Bu, ağırlıklı oy birliği veya
    RL tabanlı bir meta-ensemble'ın YERİNE geçmez (henüz yok, bkz. README
    roadmap) — LSTM'in artık rastgele seviyenin belirgin üzerinde
    (bkz. BTC-only etiketleme taraması) olduğu doğrulandıktan sonra
    eklenen ilk, en basit birleştirme kuralıdır.
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
        lstm_model: LSTMSignalModel | None = None,
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
        self.lstm_model = lstm_model

    @staticmethod
    def _combine_predictions(xgb: Prediction, lstm: Prediction | None) -> Prediction:
        """XGBoost + LSTM için basit, kural tabanlı ensemble — bkz. sınıf
        docstring'i. `lstm=None` (model yok veya bu sembol için yeterli
        sekans verisi yoksa) XGBoost'un tahmini olduğu gibi döner."""
        if lstm is None:
            return xgb
        if xgb.direction == lstm.direction:
            # İki bağımsız model AYNI yönde mutabık -> güveni artır (üst sınır 1.0).
            return Prediction(direction=xgb.direction, confidence=min(1.0, (xgb.confidence + lstm.confidence) / 2 * 1.1))
        if xgb.direction == "neutral":
            return Prediction(direction=lstm.direction, confidence=lstm.confidence * 0.7)
        if lstm.direction == "neutral":
            return Prediction(direction=xgb.direction, confidence=xgb.confidence * 0.7)
        # İkisi de yönlü ama ZIT (biri long biri short) -> belirsizlik, işlem açma.
        return Prediction(direction="neutral", confidence=min(xgb.confidence, lstm.confidence))

    def _predict(self, symbol: str) -> tuple[Prediction, float, pd.Series] | None:
        ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, self.lookback)
        feature_row = latest_feature_vector(ohlcv)
        if feature_row is None:
            return None
        # Eğitimde kullanılan makro/order-book özellikleriyle tutarlı olması
        # için canlı tahmine de eklenir (bkz. `POST /ml/predict` aynı deseni
        # kullanır) — aksi halde model, eğitimde gördüğü 13 özelliği (11
        # makro + 3 order book... bkz. ALL_FEATURE_COLUMNS) canlıda hiç
        # görmeden (sessizce 0.0/nötr varsayarak) tahmin üretirdi.
        for col, value in latest_macro_feature_row().items():
            feature_row[col] = value
        for col, value in latest_orderbook_feature_row(symbol).items():
            feature_row[col] = value
        prediction = self.model.predict(feature_row)

        if self.lstm_model is not None:
            window = latest_sequence_window(ohlcv, symbol, self.lstm_model.seq_len, self.lstm_model.feature_columns)
            lstm_prediction = self.lstm_model.predict(window) if window is not None else None
            prediction = self._combine_predictions(prediction, lstm_prediction)

        return prediction, float(feature_row["close"]), feature_row

    def evaluate(self, symbol: str) -> Action | None:
        result = self._predict(symbol)
        if result is None:
            return None
        prediction, price, feature_row = result
        db.record_signal(symbol, source="ml", direction=prediction.direction, confidence=prediction.confidence, price=price)
        record_ml_prediction(symbol, prediction.direction, prediction.confidence)
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

        # Kademeli alım: pozisyon hâlâ AYNI yönde ve yeterince güvenli
        # sinyal veriyorsa (yani sinyal bir sonraki döngüde de kalıcıysa)
        # ve tüm alım dilimleri henüz dolmadıysa, bir sonraki dilim eklenir
        # (bkz. `PortfolioManager.add_entry_tranche`). Yalnızca `portfolio`
        # katmanı varken anlamlıdır — basit `PaperPositionStore` dilim
        # takibi desteklemez.
        if (
            self.portfolio is not None
            and hasattr(position, "entry_fully_filled")
            and not position.entry_fully_filled()
            and prediction.direction == position.direction
            and prediction.confidence >= self.open_confidence
        ):
            return Action(symbol, "add_entry_tranche", "kademeli alım: sonraki dilim", price, prediction.confidence)

        return Action(symbol, "hold", "pozisyon açık, sinyal değişmedi", price, prediction.confidence)

    def _open(self, symbol: str, direction: str, price: float, confidence: float | None = None) -> Action | None:
        if kill_switch.is_active():
            return Action(symbol, "blocked", f"kill switch aktif: {kill_switch.status().reason}", price, 0.0)

        if self.portfolio is None:
            self.positions.open(symbol, direction, price)
            return None

        stop_loss_price = (
            price * (1 - self.assumed_stop_loss_pct / 100)
            if direction == "long"
            else price * (1 + self.assumed_stop_loss_pct / 100)
        )
        vix_zscore = None
        if self.portfolio.rules.vix_regime_filter_enabled:
            vix_zscore = latest_macro_feature_row().get("macro_vix_norm")

        decision = self.portfolio.propose_open(
            symbol, direction, price, stop_loss_price, confidence=confidence, vix_zscore=vix_zscore
        )
        if not decision.allowed or decision.size_quote <= 0:
            reason = "; ".join(decision.reasons) or "risk kuralları nedeniyle reddedildi"
            return Action(symbol, "blocked", reason, price, 0.0)

        self.portfolio.open(symbol, direction, price, decision.size_quote)
        return None

    def apply(self, action: Action) -> Action:
        if action.type == "open_long":
            blocked = self._open(action.symbol, "long", action.price, action.confidence)
            return blocked or action
        if action.type == "open_short":
            blocked = self._open(action.symbol, "short", action.price, action.confidence)
            return blocked or action
        if action.type == "add_entry_tranche":
            if self.portfolio is not None:
                self.portfolio.add_entry_tranche(action.symbol, action.price)
            return action
        if action.type == "close":
            if self.portfolio is not None:
                record = self.portfolio.close_tranche(action.symbol, action.price)
                if record and record.get("partial"):
                    remaining = self.portfolio.get(action.symbol)
                    total_tranches = len(remaining.exit_tranche_weights) if remaining else record["tranche"]
                    return Action(
                        action.symbol,
                        "close_partial",
                        f"{action.reason} (dilim {record['tranche']}/{total_tranches})",
                        action.price,
                        action.confidence,
                    )
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
