import logging

from app.core.config import settings
from app.engine.service import ModelNotTrained, run_cycle_once
from app.exchanges import get_exchange
from app.macro.service import refresh_and_record_macro_snapshot
from app.ml.meta_label import DEFAULT_META_MODEL_PATH
from app.ml.model import SignalModel
from app.ml.train import train_meta_label_model, train_signal_model_validated
from app.orderbook.service import refresh_all_configured_symbols
from app.screener.scanner import top_long, top_short
from app.screener.service import refresh as refresh_screener
from app.security.kill_switch import KillSwitchActive

from . import status

logger = logging.getLogger(__name__)

SCREENER_REFRESH_JOB_ID = "screener_refresh"
ENGINE_CYCLE_JOB_ID = "engine_cycle"
MACRO_REFRESH_JOB_ID = "macro_refresh"
ORDERBOOK_REFRESH_JOB_ID = "orderbook_refresh"
AUTO_RETRAIN_JOB_ID = "auto_retrain"


def job_refresh_screener() -> None:
    """Periyodik iş: screener önbelleğini tazeler.

    API isteği gelen kullanıcı taramanın bitmesini beklemesin diye bu iş
    düzenli aralıklarla arka planda çalışır.
    """
    try:
        results = refresh_screener()
        status.record(SCREENER_REFRESH_JOB_ID, ok=True, detail=f"{len(results)} sembol tarandı")
    except Exception as exc:  # noqa: BLE001 - zamanlayıcı thread'i asla çökmemeli
        logger.exception("screener refresh job failed")
        status.record(SCREENER_REFRESH_JOB_ID, ok=False, detail=str(exc))


def job_run_engine_cycle() -> None:
    """Periyodik iş: ML karar motorunun bir döngüsünü çalıştırır.

    Model henüz eğitilmemişse bu normal bir durumdur (kullanıcı henüz
    `/ml/train` çağırmamış olabilir) — hata olarak değil, "atlandı" olarak
    kaydedilir; zamanlayıcı bir sonraki turda tekrar dener.
    """
    try:
        actions = run_cycle_once()
        summary = ", ".join(f"{a.symbol}:{a.type}" for a in actions) or "aksiyon yok"
        status.record(ENGINE_CYCLE_JOB_ID, ok=True, detail=summary)
    except ModelNotTrained as exc:
        status.record(ENGINE_CYCLE_JOB_ID, ok=True, detail=f"atlandı: {exc}")
    except KillSwitchActive as exc:
        status.record(ENGINE_CYCLE_JOB_ID, ok=True, detail=f"atlandı: {exc}")
    except Exception as exc:  # noqa: BLE001 - zamanlayıcı thread'i asla çökmemeli
        logger.exception("engine cycle job failed")
        status.record(ENGINE_CYCLE_JOB_ID, ok=False, detail=str(exc))


def job_refresh_macro() -> None:
    """Periyodik iş: ücretsiz makro veri kaynaklarının (TOTAL, BTC
    dominansı, funding rate, VIX, altın, dünya endeksleri, Fed/ECB faiz
    oranları) bir anlık görüntüsünü alıp kaydeder (bkz. `app.macro`)."""
    try:
        exchange = get_exchange(settings.exchange_id)
        snapshot = refresh_and_record_macro_snapshot(exchange)
        missing = [k for k, v in snapshot.items() if v is None]
        detail = "tüm kaynaklar alındı" if not missing else f"eksik kaynaklar: {', '.join(missing)}"
        status.record(MACRO_REFRESH_JOB_ID, ok=True, detail=detail)
    except Exception as exc:  # noqa: BLE001 - zamanlayıcı thread'i asla çökmemeli
        logger.exception("macro refresh job failed")
        status.record(MACRO_REFRESH_JOB_ID, ok=False, detail=str(exc))


def job_refresh_orderbook() -> None:
    """Periyodik iş: `feature_snapshot_symbols` ayarındaki sembollerin
    emir defteri (order book) özetinin bir anlık görüntüsünü alıp kaydeder
    (bkz. `app.orderbook`). Geçmişe dönük emir defteri verisi yoktur —
    bu tablo yalnızca bugünden itibaren birikir."""
    try:
        exchange = get_exchange(settings.exchange_id)
        results = refresh_all_configured_symbols(exchange, settings.feature_snapshot_symbols_list)
        missing = [symbol for symbol, metrics in results.items() if metrics is None]
        detail = "tüm semboller alındı" if not missing else f"eksik semboller: {', '.join(missing)}"
        status.record(ORDERBOOK_REFRESH_JOB_ID, ok=True, detail=detail)
    except Exception as exc:  # noqa: BLE001 - zamanlayıcı thread'i asla çökmemeli
        logger.exception("orderbook refresh job failed")
        status.record(ORDERBOOK_REFRESH_JOB_ID, ok=False, detail=str(exc))


def job_auto_retrain() -> None:
    """Periyodik iş (varsayılan KAPALI, bkz. `FOURKEYS_ML_AUTO_RETRAIN_ENABLED`):
    XGBoost'u (ve varsa meta-label modelini) screener'ın top long/short
    listesiyle otomatik olarak yeniden eğitir. Aralık `ml_auto_retrain_seconds`
    ile kontrol edilir (bkz. `app.core.config` — bu değerin gerekçesi ve
    ampirik olarak doğrulanamadığı notu orada belgelidir)."""
    try:
        exchange = get_exchange(settings.exchange_id)
        results = refresh_screener()
        picks = top_long(results, settings.screener_top_n) + top_short(results, settings.screener_top_n)
        symbols = [r.symbol for r in picks]
        if not symbols:
            status.record(AUTO_RETRAIN_JOB_ID, ok=True, detail="atlandı: screener'dan sembol gelmedi")
            return

        train_result = train_signal_model_validated(exchange, symbols)
        detail = (
            f"XGBoost: {train_result.rows_used} satır, "
            f"oos_balanced_acc={train_result.out_of_sample.balanced_accuracy:.3f}"
        )

        # Meta-label modeli daha önce eğitilmişse (kullanıcı bu katmanı
        # kullanıyor demektir), birincil modelle senkron kalması için o da
        # yenilenir; hiç eğitilmemişse otomatik olarak BAŞLATILMAZ (bu,
        # kullanıcının bilinçli bir tercihi olmalı, bkz. `/ml/train-meta`).
        if DEFAULT_META_MODEL_PATH.exists():
            try:
                primary_model = SignalModel.load_from()
                _, meta_rows = train_meta_label_model(exchange, symbols, primary_model)
                detail += f"; meta-label: {meta_rows} satır"
            except ValueError as exc:
                detail += f"; meta-label atlandı: {exc}"

        status.record(AUTO_RETRAIN_JOB_ID, ok=True, detail=detail)
    except Exception as exc:  # noqa: BLE001 - zamanlayıcı thread'i asla çökmemeli
        logger.exception("auto retrain job failed")
        status.record(AUTO_RETRAIN_JOB_ID, ok=False, detail=str(exc))
