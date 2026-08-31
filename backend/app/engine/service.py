from app.core.config import settings
from app.exchanges import get_exchange
from app.ml.meta_label import DEFAULT_META_MODEL_PATH, MetaLabelModel
from app.ml.model import DEFAULT_MODEL_PATH, SignalModel
from app.portfolio.shared import get_portfolio
from app.screener.scanner import top_long, top_short
from app.screener.service import get_scan_results
from app.security import kill_switch

from .decision import Action, DecisionEngine
from .positions import PaperPositionStore

# Geriye dönük uyumluluk / portföy yöneticisi olmadan tek başına kullanım için.
_positions = PaperPositionStore()


class ModelNotTrained(Exception):
    pass


def run_cycle_once() -> list[Action]:
    """Screener'ın (önbelleğe alınmış) Top Long/Short listesi üzerinde tek bir
    ML karar döngüsü çalıştırır.

    Hem `POST /engine/run-cycle` API'si hem de periyodik zamanlayıcı
    (`app.scheduler`) bu fonksiyonu çağırır — mantık tek bir yerde, iki
    tetikleyici arasında tutarlı davranış garanti eder. Screener taramasını
    tekrarlamaz; `app.screener.service` önbelleğini paylaşır.
    """
    if kill_switch.is_active():
        raise kill_switch.KillSwitchActive(f"Kill switch aktif: {kill_switch.status().reason}")

    if not DEFAULT_MODEL_PATH.exists():
        raise ModelNotTrained("Model henüz eğitilmedi. Önce /ml/train çağırın (veya train_signal_model).")

    exchange = get_exchange(settings.exchange_id)
    model = SignalModel.load_from()
    meta_model = MetaLabelModel.load_from() if DEFAULT_META_MODEL_PATH.exists() else None

    results = get_scan_results()
    picks = top_long(results, settings.screener_top_n) + top_short(results, settings.screener_top_n)
    symbols = [r.symbol for r in picks]

    engine = DecisionEngine(
        exchange=exchange,
        model=model,
        positions=_positions,
        timeframe=settings.candle_timeframe,
        lookback=settings.candle_lookback,
        portfolio=get_portfolio(),
        meta_model=meta_model,
    )
    return engine.run_cycle(symbols)
