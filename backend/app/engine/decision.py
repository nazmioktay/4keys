import logging
from dataclasses import dataclass

import pandas as pd

from app.db import repository as db
from app.exchanges.base import Exchange
from app.exchanges.cache import fetch_ohlcv_cached
from app.ml.features import latest_feature_vector
from app.ml.lstm_model import DEFAULT_LSTM_MODEL_PATH, LSTMSignalModel
from app.ml.macro_features import latest_macro_feature_row
from app.monitoring.metrics import record_ml_prediction
from app.ml.meta_label import MetaLabelModel
from app.ml.model import DEFAULT_MODEL_PATH, Prediction, SignalModel
from app.ml.model_status import get_balanced_accuracy
from app.ml.multi_timeframe_features import MULTI_TIMEFRAME_FEATURE_COLUMNS, compute_multi_timeframe_features
from app.ml.online_model import DEFAULT_ONLINE_MODEL_PATH, OnlineSignalModel
from app.ml.openinterest_features import latest_open_interest_feature_row
from app.ml.orderbook_features import latest_orderbook_feature_row
from app.ml.orderflow_features import latest_taker_buy_ratio_norm
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
    tahmini TEK BAŞINA kullanılmaz — kural tabanlı, her modelin KENDİ
    doğrulanmış becerisine göre AĞIRLIKLANDIRILMIŞ bir ensemble
    (`_combine_predictions`/`_skill_weight`, bkz. `app.ml.model_status`'a
    EN SON eğitimde yazılan `balanced_accuracy`) ile LSTM'in tahminiyle
    birleştirilir: ikisi AYNI yönü işaret ediyorsa güven, beceri ağırlıklı
    ortalamayla artırılır (daha yüksek doğrulukla eğitilmiş model daha
    fazla ağırlık taşır — önceden ikisi HER ZAMAN eşit ağırlıklıydı); biri
    nötr diğeri yönlüyse yönlü olanın indirimi KENDİ becerisine göre
    ölçeklenir (yüksek beceri -> az indirim); ZIT yönleri işaret
    ediyorlarsa (biri long biri short) belirsizlik nedeniyle nötre
    düşülür (bu dal DEĞİŞTİRİLMEDİ — çakışan sinyallerde temkinli kalmak
    kasıtlı). Bu, RL tabanlı bir meta-ensemble'ın YERİNE geçmez (henüz
    yok, bkz. README roadmap).

    `online_model` verilirse (opsiyonel, bkz. `app.ml.online_model` —
    `river` ile gerçek çevrimiçi öğrenme, kavram kaymasına karşı), AYNI
    `_combine_predictions` kuralı üçüncü bir "oy" olarak (önce XGBoost+LSTM
    birleştirilir, sonra sonuç online modelle birleştirilir) uygulanır.
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
        online_model: OnlineSignalModel | None = None,
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
        self.online_model = online_model
        # Ensemble birleştirmesi (`_combine_predictions`) her modelin KENDİ
        # doğrulanmış becerisine (bkz. `app.ml.model_status`'a EN SON eğitimde
        # yazılan `balanced_accuracy`) göre ağırlıklandırılır — önceden
        # (0.7/1.1 gibi) sabit, elle yazılmış katsayılar tüm modellere
        # AYNI şekilde uygulanıyordu, oysa doğrulanmış skorları (ör. XGBoost
        # ~0.45, online ~0.50) FARKLI. `_skill_weight` bilinmeyen (None,
        # ör. status.json henüz yok) doğruluk için nötr 0.5 döner — eski
        # davranışla YAKLAŞIK aynı büyüklükte, geriye dönük uyumlu.
        self._xgb_skill = self._skill_weight(get_balanced_accuracy(DEFAULT_MODEL_PATH))
        self._lstm_skill = self._skill_weight(get_balanced_accuracy(DEFAULT_LSTM_MODEL_PATH))
        self._online_skill = self._skill_weight(get_balanced_accuracy(DEFAULT_ONLINE_MODEL_PATH))

    @staticmethod
    def _skill_weight(balanced_accuracy: float | None, n_classes: int = 3) -> float:
        """Dengeli doğruluğu, rastgele seviyenin (n_classes sınıflı bir
        problemde ~1/n_classes) ÜZERİNDEKİ beceriyi yansıtan 0..1 arası bir
        ağırlığa dönüştürür: `balanced_accuracy=1/n_classes` (tam rastgele)
        -> 0.0 ağırlık (bu modelin ensemble'a hiçbir katkısı olmamalı);
        `balanced_accuracy=1.0` (mükemmel) -> 1.0. Bilinmeyen (`None`,
        henüz kaydedilmemiş) doğruluk için nötr 0.5 döner."""
        if balanced_accuracy is None:
            return 0.5
        baseline = 1.0 / n_classes
        skill = (balanced_accuracy - baseline) / (1.0 - baseline)
        return float(min(1.0, max(0.0, skill)))

    @staticmethod
    def _combine_predictions(
        xgb: Prediction, lstm: Prediction | None, xgb_skill: float = 0.5, lstm_skill: float = 0.5
    ) -> Prediction:
        """XGBoost + (LSTM veya online) için kural tabanlı, BECERİ AĞIRLIKLI
        ensemble — bkz. sınıf docstring'i. `lstm=None` (model yok veya bu
        sembol için yeterli sekans verisi yoksa) XGBoost'un tahmini olduğu
        gibi döner. `xgb_skill`/`lstm_skill` (bkz. `_skill_weight`) her
        modelin KENDİ doğrulanmış becerisini yansıtır — daha yüksek beceri,
        birleşik güvene daha fazla ağırlıkla katkı verir."""
        if lstm is None:
            return xgb
        if xgb.direction == lstm.direction:
            # İki bağımsız model AYNI yönde mutabık -> beceriye göre
            # ağırlıklı ortalama, güveni artır (üst sınır 1.0).
            total_skill = xgb_skill + lstm_skill
            if total_skill <= 0:
                blended = (xgb.confidence + lstm.confidence) / 2
            else:
                blended = (xgb.confidence * xgb_skill + lstm.confidence * lstm_skill) / total_skill
            return Prediction(direction=xgb.direction, confidence=min(1.0, blended * 1.1))
        if xgb.direction == "neutral":
            # Yönlü modelin KENDİ becerisi yüksekse indirim daha az olur
            # (skill=1 -> indirim yok, skill=0 -> %50 indirim).
            return Prediction(direction=lstm.direction, confidence=lstm.confidence * (0.5 + 0.5 * lstm_skill))
        if lstm.direction == "neutral":
            return Prediction(direction=xgb.direction, confidence=xgb.confidence * (0.5 + 0.5 * xgb_skill))
        # İkisi de yönlü ama ZIT (biri long biri short) -> belirsizlik, işlem açma.
        return Prediction(direction="neutral", confidence=min(xgb.confidence, lstm.confidence))

    def _predict(self, symbol: str) -> tuple[Prediction, float, pd.Series] | None:
        ohlcv = fetch_ohlcv_cached(self.exchange, symbol, self.timeframe, self.lookback)
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
        for col, value in latest_open_interest_feature_row(symbol).items():
            feature_row[col] = value
        htf_row = compute_multi_timeframe_features(ohlcv).iloc[-1]
        for col in MULTI_TIMEFRAME_FEATURE_COLUMNS:
            feature_row[col] = htf_row[col]
        feature_row["taker_buy_ratio_norm"] = latest_taker_buy_ratio_norm(self.exchange, symbol, self.timeframe)
        prediction = self.model.predict(feature_row)
        # `running_skill`, o ana kadar birleştirilen tahminin "etkin"
        # becerisini izler — bir model başarıyla katılınca (bkz. aşağıdaki
        # None kontrolleri) ortalamaya dahil edilir, katılamadıysa (ör.
        # yeterli sekans verisi yok) DEĞİŞMEZ.
        running_skill = self._xgb_skill

        if self.lstm_model is not None:
            window = latest_sequence_window(
                ohlcv,
                symbol,
                self.lstm_model.seq_len,
                self.lstm_model.feature_columns,
                exchange=self.exchange,
                timeframe=self.timeframe,
            )
            lstm_prediction = self.lstm_model.predict(window) if window is not None else None
            prediction = self._combine_predictions(prediction, lstm_prediction, running_skill, self._lstm_skill)
            if lstm_prediction is not None:
                running_skill = (running_skill + self._lstm_skill) / 2

        if self.online_model is not None:
            # Online model (river ARF), prequential değerlendirmede BTC-only
            # veride overall_balanced_accuracy ~%49.7 gösterdi (soğuk
            # başlangıç sonrası ~%43-45 istikrarlı) — XGBoost/LSTM'den daha
            # iyi. AYNI ikili birleştirme kuralı (`_combine_predictions`)
            # burada da uygulanır — üçüncü bir "oy" olarak.
            online_prediction = self.online_model.predict(feature_row)
            prediction = self._combine_predictions(prediction, online_prediction, running_skill, self._online_skill)

        return prediction, float(feature_row["close"]), feature_row

    def evaluate(self, symbol: str) -> Action | None:
        result = self._predict(symbol)
        if result is None:
            return None
        prediction, price, feature_row = result
        db.record_signal(symbol, source="ml", direction=prediction.direction, confidence=prediction.confidence, price=price)
        record_ml_prediction(symbol, prediction.direction, prediction.confidence)
        position = self.portfolio.get(symbol) if self.portfolio is not None else self.positions.get(symbol)

        # Stop-loss: modelin sinyalinden BAĞIMSIZ, sabit bir risk kapısı —
        # önceden `stop_loss_price` yalnızca Kelly boyutlandırma hesabında
        # kullanılıp atılıyordu, fiyat o seviyeyi geçse bile HİÇBİR ZAMAN
        # kontrol edilmiyordu (bkz. `RiskRules.stop_loss_enabled`,
        # `PortfolioPosition.stop_loss_breached`). Model hâlâ "tut" diyor
        # olsa bile bu kontrol önceliklidir.
        if (
            position is not None
            and self.portfolio is not None
            and self.portfolio.rules.stop_loss_enabled
            and hasattr(position, "stop_loss_breached")
            and position.stop_loss_breached(price)
        ):
            return Action(symbol, "close", f"stop-loss tetiklendi (seviye={position.stop_loss_price:.4f})", price, 1.0)

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

        self.portfolio.open(symbol, direction, price, decision.size_quote, stop_loss_price=stop_loss_price)
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
