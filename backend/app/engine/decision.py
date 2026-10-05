from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from app.db import repository as db
from app.exchanges.base import Exchange
from app.exchanges.cache import fetch_ohlcv_cached, timeframe_minutes
from app.ml.advanced_indicators import average_true_range
from app.ml.features import latest_feature_vector
from app.ml.macro_features import latest_macro_feature_row
from app.monitoring.metrics import record_ml_prediction
from app.ml.meta_label import MetaLabelModel
from app.ml.model import DEFAULT_MODEL_PATH, Prediction, SignalModel
from app.ml.model_paths import DEFAULT_LSTM_MODEL_PATH
from app.ml.model_status import get_balanced_accuracy

if TYPE_CHECKING:
    from app.ml.lstm_model import LSTMSignalModel
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
        # DİKKAT — bu eşikler KALİBRE EDİLMİŞ 3 sınıflı olasılık ölçeğindedir.
        # Eski 0.6/0.55 varsayılanları, güvenin HAM (kalibre edilmemiş) argmax
        # olasılığından okunduğu dönemden kalmaydı. `SignalModel.predict`
        # artık yönü ham modelden, GÜVENİ ise kalibre edilmiş olasılıktan
        # alıyor (bkz. kalibrasyon çökmesi düzeltmesi) — kalibrasyon,
        # olasılıkları taban orana doğru SIKIŞTIRIR: 3 sınıfta rastgele
        # seviye 0.333'tür ve gerçek ölçümde yönlü tahminlerin güveni
        # p50=0.44, p99=0.53, MAKSİMUM 0.56 çıkıyor — yani 0.6 eşiği
        # MATEMATİKSEL OLARAK ULAŞILAMAZDI ve sistem (hem backtest hem
        # CANLI motor) hiçbir zaman pozisyon açamıyordu. 0.5 (ulaşılabilir
        # ama seçici) bu yüzden ilk varsayılan yapılmıştı.
        #
        # `POST /backtest/system/sweep-confidence` ile GERÇEK holdout
        # verisinde ölçüldü (bkz. README "güven eşiği taraması"): 0.5'te
        # total_pnl_pct NEGATİF (-%0.25); 0.55'te 89 işlem, kazanma=%53.9,
        # PnL=+%2.09, max_drawdown=%0.64 (taramanın EN DÜŞÜK drawdown'ı);
        # 0.6'da 48 işlem, PnL=+%2.70 (en yüksek) ama drawdown biraz daha
        # yüksek; 0.65/0.7'de örneklem (22/12 işlem) güvenilir yorum için
        # çok küçük. 0.55 seçildi: 0.6'ya göre ~2x daha büyük örneklem VE
        # taramanın en düşük drawdown'ı — ham PnL farkı (0.6'nın lehine
        # ~%0.6) küçülen örneklemle örtüşen işlemler arasındaki gürültü
        # payının içinde. TEK bir holdout penceresinde ölçüldü — kalıcı bir
        # kanun değil, yeni veriyle periyodik olarak yeniden ölçülmeli.
        #
        # GÜNCELLEME (bkz. README "karlılık" — OOM düzeltmelerinden ve
        # "temel sadeleşme"den sonra model/veri kökten değişti, sweep
        # YENİDEN ölçüldü): `open_confidence=0.55`'te ARTIK yalnızca 31
        # işlem, kazanma=%45.16, PnL **NEGATİF** (-%0.888) — eski ölçümün
        # tam tersi. `0.50`'de ise 149 işlem (5× daha büyük, çok daha
        # güvenilir örneklem), kazanma=%63.09, PnL=+%2.68 — taramanın en
        # iyi sonucu. 0.6+ eşiklerde örneklem 3/0/0'a çöküyor, yorumlanamaz.
        # Şemanın ORİJİNAL varsayılanına (0.5/0.45) geri dönüldü — bu
        # tersine dönüş, eşiğin kendisinin "doğru" olmadığını, periyodik
        # yeniden ölçümün NEDEN gerekli olduğunu gösteriyor (yukarıdaki not
        # zaten bunu öngörmüştü).
        open_confidence: float = 0.5,
        close_confidence: float = 0.45,
        portfolio: PortfolioManager | None = None,
        # GÜNCELLEME: önceden sabit bir yüzdelik (`assumed_stop_loss_pct=3.0`)
        # kullanılıyordu — bu, `app.backtest.system_runner.run_system_backtest`in
        # DOĞRULADIĞI risk yönetimiyle (ATR tabanlı, `SystemBacktestRequest.atr_*`)
        # AYNI DEĞİLDİ: yani paper trading, backtest'in ölçtüğü sistemin
        # birebir aynısı değildi (bkz. README "canlı vs backtest tutarlılığı").
        # `atr_period`/`atr_stop_loss_mult` VARSAYILANLARI backtest'in
        # varsayılanlarıyla (14 / 1.5×ATR) BİREBİR aynı tutulur.
        atr_period: int = 14,
        atr_stop_loss_mult: float = 1.5,
        meta_model: MetaLabelModel | None = None,
        lstm_model: LSTMSignalModel | None = None,
        online_model: OnlineSignalModel | None = None,
        # `bash deploy/sweep-meta-label-threshold.sh`'a KARŞILIK GELEN backtest
        # taramasıyla (bkz. README "Karlılık") ölçüldü: 0.4..0.7 arasında PnL
        # 0.6'da (511 işlem, %66,93 kazanma, +%32,37 PnL — tabandaki 0.5'in
        # +%30,50'sinden iyi) TEPE yapıp sonrasında (0.65: +%30,04, 0.7:
        # +%20,67 — örneklem küçüldükçe) geriliyor. `SystemBacktestRequest`
        # ile SENKRON tutulmalı (bkz. o alanın kendi yorumu).
        meta_label_act_threshold: float = 0.6,
        # Sembol -> sinyali en son değerlendirilen KAPANMIŞ mumun zaman damgası.
        # Verilirse (canlı döngü verir, bkz. `engine.service`) model sinyali her
        # kapanmış mum için YALNIZCA BİR KEZ uygulanır — backtest de her mumu
        # bir kez değerlendirir. Döngü 5 dakikada bir çalıştığı için bu olmadan
        # aynı saatlik mum 12 kez değerlendiriliyor, kapat/aç çalkalanması
        # mümkün oluyordu. Stop-loss kontrolü bu kapıdan BAĞIMSIZDIR.
        signal_bar_memory: dict[str, pd.Timestamp] | None = None,
    ) -> None:
        self.exchange = exchange
        self.model = model
        self.positions = positions
        self.timeframe = timeframe
        self.lookback = lookback
        self.open_confidence = open_confidence
        self.close_confidence = close_confidence
        self.portfolio = portfolio
        self.atr_period = atr_period
        self.atr_stop_loss_mult = atr_stop_loss_mult
        self.meta_label_act_threshold = meta_label_act_threshold
        self.meta_model = meta_model
        self.lstm_model = lstm_model
        self.online_model = online_model
        self._last_atr: dict[str, float] = {}  # sembol -> en son hesaplanan ATR (bkz. `_predict`/`_open`)
        self._last_bar_ts: dict[str, pd.Timestamp] = {}  # sembol -> `_predict`'in kullandığı son kapanmış mum
        self._last_bar: dict[str, pd.Series] = {}  # sembol -> son kapanmış mumun OHLC'si (stop simülasyonu için)
        self.signal_bar_memory = signal_bar_memory
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
            # Yönlü modelin KENDİ becerisi yüksekse indirim daha az olur.
            # ARALIK BİLEREK 0.7..1.0: eski (beceri-körü) sabit katsayı 0.7'ydi
            # ve bu, TABAN olarak korunur — beceri ağırlıklandırması hiçbir
            # zaman eski davranıştan DAHA SERT indirim yapmamalı. Önceki
            # sürümde aralık 0.5..1.0'dı; gerçek modellerin becerisi düşük
            # (balanced_accuracy ~0.40-0.46 -> skill ~0.10-0.19) olduğu için
            # bu, iki ardışık indirim adımıyla birleşince güveni
            # `open_confidence` eşiğinin altına çekip HİÇ işlem açılmamasına
            # yol açtı (üretimde gözlenen "0 işlem" backtest'i).
            return Prediction(direction=lstm.direction, confidence=lstm.confidence * (0.7 + 0.3 * lstm_skill))
        if lstm.direction == "neutral":
            return Prediction(direction=xgb.direction, confidence=xgb.confidence * (0.7 + 0.3 * xgb_skill))
        # İkisi de yönlü ama ZIT (biri long biri short) -> belirsizlik, işlem açma.
        return Prediction(direction="neutral", confidence=min(xgb.confidence, lstm.confidence))

    def _predict(self, symbol: str) -> tuple[Prediction, float, pd.Series] | None:
        ohlcv = fetch_ohlcv_cached(self.exchange, symbol, self.timeframe, self.lookback)
        feature_row = latest_feature_vector(ohlcv)
        if feature_row is None:
            return None
        if len(ohlcv):
            self._last_bar_ts[symbol] = pd.Timestamp(ohlcv["timestamp"].iloc[-1])
            self._last_bar[symbol] = ohlcv.iloc[-1]
        # Backtest'in ATR bazlı stop-loss'uyla (bkz. `_open`) AYNI formülle
        # (`average_true_range`, AYNI varsayılan `atr_period`) hesaplanır.
        atr_series = average_true_range(ohlcv, length=self.atr_period)
        atr_now = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
        if pd.notna(atr_now) and atr_now > 0:
            self._last_atr[symbol] = atr_now
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

    def _live_price(self, symbol: str) -> float | None:
        try:
            price = self.exchange.fetch_ticker_price(symbol)
        except Exception:  # noqa: BLE001 - canlı fiyat alınamazsa son kapanmış mumun kapanışına düşülür
            logger.warning("engine: %s için canlı fiyat alınamadı, mum kapanışı kullanılacak", symbol)
            return None
        if price is None or not price > 0:
            return None
        return float(price)

    def _stop_action(self, symbol: str, position, observed_price: float) -> Action:
        """Stop tetiklendiğinde kapanış aksiyonu. `simulate_exchange_stop`
        açıksa borsadaki bir stop emri gibi seviyeden dolar; kapalıysa
        gözlenen fiyattan."""
        fill = observed_price
        if self.portfolio is not None and self.portfolio.rules.simulate_exchange_stop:
            fill = position.stop_loss_price
        return Action(symbol, "close", f"stop-loss tetiklendi (seviye={position.stop_loss_price:.4f})", fill, 1.0)

    def _bar_since_open(self, symbol: str, position) -> pd.Series | None:
        """Son kapanmış mum, pozisyon açıldıktan SONRA başladıysa onu döner
        (öncesindeki fiyat hareketi pozisyona ait değildir)."""
        bar = self._last_bar.get(symbol)
        if bar is None:
            return None
        bar_start = pd.Timestamp(bar["timestamp"])
        opened_at = pd.Timestamp(position.opened_at)
        if opened_at.tzinfo is not None:
            opened_at = opened_at.tz_convert(None)
        if bar_start.tzinfo is not None:
            bar_start = bar_start.tz_convert(None)
        return bar if bar_start >= opened_at else None

    def _bar_stop_fill(self, symbol: str, position) -> float | None:
        """Pozisyon açıldıktan SONRA başlamış son kapanmış mum stop seviyesini
        mum içinde geçtiyse (yoklamalar arasında kaçırılan tetikleme) dolum
        fiyatını döner: normalde stop seviyesi, mum seviyenin ötesinde
        açıldıysa (gap) açılış — backtest'in `intrabar_stops` kuralıyla aynı."""
        bar = self._bar_since_open(symbol, position)
        if bar is None or position.stop_loss_price is None:
            return None
        stop = position.stop_loss_price
        if position.direction == "long" and float(bar["low"]) <= stop:
            return min(stop, float(bar["open"]))
        if position.direction == "short" and float(bar["high"]) >= stop:
            return max(stop, float(bar["open"]))
        return None

    def _apply_breakeven(self, symbol: str, position, observed_price: float) -> None:
        """Backtest'teki `_apply_breakeven` ile aynı kural: lehe hareket giriş
        ATR'sinin `breakeven_atr_mult` katına ulaşınca stop girişe (+maliyet)
        çekilir, bir kez. En iyi fiyat canlı fiyat ve son kapanmış mumun
        high/low'undan izlenir."""
        rules = self.portfolio.rules
        if rules.breakeven_atr_mult is None or position.breakeven_done or position.stop_loss_price is None or not position.entry_atr:
            return
        is_long = position.direction == "long"
        candidates = [observed_price]
        bar = self._bar_since_open(symbol, position)
        if bar is not None:
            candidates.append(float(bar["high"] if is_long else bar["low"]))
        best = max(candidates) if is_long else min(candidates)
        if position.best_price is None:
            position.best_price = best
        position.best_price = max(position.best_price, best) if is_long else min(position.best_price, best)

        trigger = rules.breakeven_atr_mult * position.entry_atr
        cost_pct = (rules.commission_pct + rules.slippage_pct) * 2
        if is_long and position.best_price - position.entry_price >= trigger:
            position.stop_loss_price = max(position.stop_loss_price, position.entry_price * (1 + cost_pct / 100))
            position.breakeven_done = True
        elif not is_long and position.entry_price - position.best_price >= trigger:
            position.stop_loss_price = min(position.stop_loss_price, position.entry_price * (1 - cost_pct / 100))
            position.breakeven_done = True
        if position.breakeven_done:
            self.portfolio.persist()

    def _time_exit_reason(self, position) -> str | None:
        max_bars = self.portfolio.rules.max_holding_bars
        if max_bars is None:
            return None
        opened_at = pd.Timestamp(position.opened_at)
        now = pd.Timestamp.now(tz="UTC")
        if opened_at.tzinfo is None:
            opened_at = opened_at.tz_localize("UTC")
        held = now - opened_at
        if held >= pd.Timedelta(minutes=timeframe_minutes(self.timeframe)) * max_bars:
            return f"zaman çıkışı ({max_bars} mum doldu)"
        return None

    def evaluate(self, symbol: str) -> Action | None:
        position = self.portfolio.get(symbol) if self.portfolio is not None else self.positions.get(symbol)
        stop_guarded = (
            position is not None
            and self.portfolio is not None
            and self.portfolio.rules.stop_loss_enabled
            and hasattr(position, "stop_loss_breached")
        )

        # Stop-loss: modelin sinyalinden BAĞIMSIZ, her döngüde (5 dk) CANLI
        # fiyatla kontrol edilir — önceden yalnızca saatlik mumun (ve bayat
        # önbelleğin) kapanışıyla bakıldığı için stop seviyenin çok ötesinde
        # tetikleniyordu. Model hâlâ "tut" diyor olsa bile önceliklidir.
        live_price = self._live_price(symbol) if stop_guarded else None
        if stop_guarded and live_price is not None and position.stop_loss_breached(live_price):
            return self._stop_action(symbol, position, live_price)

        result = self._predict(symbol)
        if result is None:
            return None
        prediction, price, feature_row = result

        # Yoklamalar arasında kapanmış mumun high/low'u stop'u geçtiyse (borsa
        # stop'u burada tetiklenirdi) — yalnızca stop simülasyonu açıkken.
        if stop_guarded and self.portfolio.rules.simulate_exchange_stop:
            bar_fill = self._bar_stop_fill(symbol, position)
            if bar_fill is not None:
                return Action(symbol, "close", f"stop-loss tetiklendi (seviye={position.stop_loss_price:.4f}, mum içi)", bar_fill, 1.0)

        # Canlı fiyat alınamadıysa son kapanmış mumla aynı kontrol (yedek yol).
        if stop_guarded and live_price is None and position.stop_loss_breached(price):
            return self._stop_action(symbol, position, price)

        if position is not None and self.portfolio is not None:
            self._apply_breakeven(symbol, position, live_price if live_price is not None else price)
            time_exit = self._time_exit_reason(position)
            if time_exit is not None:
                return Action(symbol, "close", time_exit, live_price if live_price is not None else price, 1.0)

        bar_ts = self._last_bar_ts.get(symbol)
        if self.signal_bar_memory is not None and bar_ts is not None:
            if self.signal_bar_memory.get(symbol) == bar_ts:
                return Action(symbol, "hold", "yeni kapanmış mum yok — bu mumun sinyali zaten değerlendirildi", price, prediction.confidence)
            self.signal_bar_memory[symbol] = bar_ts
            # Backtest sinyal mumundan SONRAKİ ilk fiyattan (bir sonraki mumun
            # açılışı) işlem yapar; canlıda bunun karşılığı şu anki fiyattır.
            if live_price is None:
                live_price = self._live_price(symbol)
            if live_price is not None:
                price = live_price

        db.record_signal(symbol, source="ml", direction=prediction.direction, confidence=prediction.confidence, price=price)
        record_ml_prediction(symbol, prediction.direction, prediction.confidence)

        if position is None:
            if prediction.direction in ("long", "short") and prediction.confidence >= self.open_confidence:
                if self.portfolio is not None and self.portfolio.rules.trend_filter == "block":
                    trend = feature_row.get("htf_1d_ema_gap", 0.0)
                    trend = 0.0 if trend is None or pd.isna(trend) else float(trend)
                    if (prediction.direction == "long" and trend < 0) or (prediction.direction == "short" and trend > 0):
                        return Action(symbol, "hold", "trend filtresi: günlük eğilimin tersine açılmıyor", price, prediction.confidence)
                if self.meta_model is not None:
                    meta_decision = self.meta_model.decide(
                        feature_row, prediction.confidence, act_threshold=self.meta_label_act_threshold
                    )
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

        # ATR bazlı stop-loss — `app.backtest.system_runner.run_system_backtest`'in
        # pozisyon açma bloğuyla AYNI formül: `price ± atr_stop_loss_mult * atr_now`.
        # `_last_atr` bu döngüde `_predict` tarafından doldurulmuş olmalı; yoksa
        # (ör. ilk çağrıda hiç ATR hesaplanamadıysa) eski sabit %3'e düşülür —
        # tamamen stop-loz'suz açmaktansa daha güvenli bir yaklaşıklık.
        atr_now = self._last_atr.get(symbol)
        if atr_now is not None:
            stop_loss_price = (
                price - self.atr_stop_loss_mult * atr_now if direction == "long" else price + self.atr_stop_loss_mult * atr_now
            )
        else:
            stop_loss_price = price * (1 - 0.03) if direction == "long" else price * (1 + 0.03)
        vix_zscore = None
        if self.portfolio.rules.vix_regime_filter_enabled:
            vix_zscore = latest_macro_feature_row().get("macro_vix_norm")

        decision = self.portfolio.propose_open(
            symbol, direction, price, stop_loss_price, confidence=confidence, vix_zscore=vix_zscore
        )
        if not decision.allowed or decision.size_quote <= 0:
            reason = "; ".join(decision.reasons) or "risk kuralları nedeniyle reddedildi"
            return Action(symbol, "blocked", reason, price, 0.0)

        # Paper trading, borsanın MIN_NOTIONAL kısıtını hiç GÖRMEZ (bkz.
        # `app.trading.executor._validate_order_limits`, canlı emirlerde
        # AYNI kontrol var) — bu kontrol olmadan, gerçek hesapta Binance'in
        # anında reddedeceği kadar küçük bir pozisyon paper'da sorunsuz
        # "başarılı" görünüp performans istatistiklerini olduğundan
        # gerçekçi/iyi gösterebilir. Limit bilgisi alınamazsa (`None`)
        # sessizce atlanır.
        limits = self.exchange.fetch_market_limits(symbol, "future")
        if limits and limits.get("cost_min") and decision.size_quote < limits["cost_min"]:
            return Action(
                symbol,
                "blocked",
                f"hesaplanan pozisyon büyüklüğü (${decision.size_quote:.2f}) borsanın asgari emir değerinin "
                f"(${limits['cost_min']:g}) altında — gerçek hesapta bu boyutta açılamazdı",
                price,
                confidence or 0.0,
            )

        position = self.portfolio.open(symbol, direction, price, decision.size_quote, stop_loss_price=stop_loss_price)
        position.entry_atr = atr_now
        position.best_price = price
        self.portfolio.persist()
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
                record = self.portfolio.close_tranche(action.symbol, action.price, reason=action.reason)
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
