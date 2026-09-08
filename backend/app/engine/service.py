from app.core.config import settings
from app.exchanges import get_exchange
from app.ml.meta_label import DEFAULT_META_MODEL_PATH, MetaLabelModel
from app.ml.model import DEFAULT_MODEL_PATH, SignalModel
from app.ml.model_paths import DEFAULT_LSTM_MODEL_PATH
from app.ml.model_status import is_model_enabled
from app.ml.online_model import DEFAULT_ONLINE_MODEL_PATH, OnlineSignalModel
from app.portfolio.shared import get_portfolio
from app.security import kill_switch

from .decision import Action, DecisionEngine
from .positions import PaperPositionStore

# Geriye dönük uyumluluk / portföy yöneticisi olmadan tek başına kullanım için.
_positions = PaperPositionStore()


class ModelNotTrained(Exception):
    pass


def run_cycle_once() -> list[Action]:
    """`settings.ml_primary_symbol` (BTC-only) üzerinde tek bir ML karar
    döngüsü çalıştırır.

    Hem `POST /engine/run-cycle` API'si hem de periyodik zamanlayıcı
    (`app.scheduler`) bu fonksiyonu çağırır — mantık tek bir yerde, iki
    tetikleyici arasında tutarlı davranış garanti eder.

    GÜNCELLEME (bkz. README "karlılık"): ÖNCEDEN screener'ın Top Long/Short
    listesi (piyasadaki HERHANGİ bir sembol) doğrudan buraya besleniyordu —
    ama model yalnızca `ml_primary_symbol` ile eğitiliyor, yani model hiç
    görmediği bir dağılımdan ("out-of-distribution") tahmin üretmeye
    zorlanıyordu. Artık yalnızca eğitildiği sembolde çalışıyor; screener
    hâlâ ayrı bir bilgi kaynağı (`/screener` API'si) olarak kullanılabilir
    ama ML karar motorunu artık YÖNLENDİRMİYOR.
    """
    if kill_switch.is_active():
        raise kill_switch.KillSwitchActive(f"Kill switch aktif: {kill_switch.status().reason}")

    if not DEFAULT_MODEL_PATH.exists():
        raise ModelNotTrained("Model henüz eğitilmedi. Önce /ml/train çağırın (veya train_signal_model).")

    exchange = get_exchange(settings.exchange_id)
    model = SignalModel.load_from()
    meta_model = MetaLabelModel.load_from() if DEFAULT_META_MODEL_PATH.exists() else None
    # LSTM/online modelinin canlı karar motoruna (ensemble olarak) katılması
    # ARTIK statik bir ayar bayrağıyla (`FOURKEYS_ENSEMBLE_LSTM_ENABLED` vb.)
    # DEĞİL, en son eğitimin kalite kapısından geçip geçmediğine göre OTOMATİK
    # belirlenir (bkz. `app.ml.model_status` — `Settings.ml_min_balanced_accuracy`,
    # varsayılan 0.37). Bir model eşiği geçtiği eğitimden sonra otomatik
    # devreye girer; bir sonraki eğitiminde eşiğin altında kalırsa (eski
    # dosyası hâlâ diskte olsa bile) otomatik devre dışı kalır.
    lstm_model = None
    if is_model_enabled(DEFAULT_LSTM_MODEL_PATH):
        from app.ml.lstm_model import LSTMSignalModel  # lazy: torch (~460MB) yalnızca LSTM gerçekten aktifse yüklenir

        lstm_model = LSTMSignalModel.load_from()
    online_model = (
        OnlineSignalModel.load_from() if is_model_enabled(DEFAULT_ONLINE_MODEL_PATH) else None
    )

    # GÜNCELLEME (bkz. README "karlılık" — kullanıcı sorusu: "paper trade'de
    # gelen ticker'lar nasıl seçiliyor, güvenli mi?"): ÖNCEDEN screener'ın
    # Top Long/Short'u (piyasadaki HERHANGİ bir semboldü) doğrudan karar
    # motoruna gönderiliyordu — ama model yalnızca `ml_primary_symbol`
    # (BTC-only) ile eğitiliyor (bkz. README madde 9, çok-sembollü eğitim
    # denenip geriletildi). Yani model, hiç görmediği bir dağılımdan
    # (başka bir coin'in fiyat/hacim davranışı) tahmin üretmeye
    # zorlanıyordu — "out-of-distribution" riski. Artık canlı karar
    # döngüsü de SADECE `ml_primary_symbol` üzerinde çalışıyor; screener
    # taraması hâlâ /screener API'sinde ayrı bir bilgi kaynağı olarak
    # görüntülenebilir, ama artık ML karar motorunu YÖNLENDİRMİYOR.
    symbols = [settings.ml_primary_symbol]

    engine = DecisionEngine(
        exchange=exchange,
        model=model,
        positions=_positions,
        # ML modeli artık `ml_train_timeframe`/`ml_train_lookback` (1h,
        # 10.000 mum) ile eğitiliyor — karar motoru da eğitimle AYNI zaman
        # dilimini kullanmalı, aksi halde model eğitimde görmediği bir
        # dağılımdan (4h) tahmin üretmeye çalışır (screener'ın kendi
        # candle_timeframe'i, 4h, teknik skor gösterimi için ayrı kalmaya
        # devam eder — bkz. app/screener/scanner.py).
        timeframe=settings.ml_train_timeframe,
        lookback=settings.ml_train_lookback,
        # Sabit kod değeri DEĞİL, `settings.live_open_confidence`/
        # `live_close_confidence`'tan okunur — bkz. `Settings` docstring'i:
        # `job_periodic_optimization` (auto-apply açıksa) bunu ÇALIŞMA
        # ZAMANINDA güncelleyebilir, `docker build` gerekmeden.
        open_confidence=settings.live_open_confidence,
        close_confidence=settings.live_close_confidence,
        portfolio=get_portfolio(),
        meta_model=meta_model,
        lstm_model=lstm_model,
        online_model=online_model,
    )
    return engine.run_cycle(symbols)
