import logging

from app.backtest.data import timeframe_to_minutes
from app.core.config import settings
from app.engine.service import ModelNotTrained, run_cycle_once
from app.exchanges import get_exchange
from app.macro.service import refresh_and_record_macro_snapshot
from app.ml.lstm_model import DEFAULT_LSTM_MODEL_PATH, LSTMSignalModel
from app.ml.meta_label import DEFAULT_META_MODEL_PATH
from app.ml.model import SignalModel
from app.ml.online_model import DEFAULT_ONLINE_MODEL_PATH
from app.ml.regime import DEFAULT_REGIME_MODEL_PATH
from app.ml.train import (
    train_lstm_signal_model,
    train_meta_label_model,
    train_online_signal_model,
    train_signal_model_validated,
    train_signal_models_by_regime,
)
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
AUTO_RETRAIN_LSTM_JOB_ID = "auto_retrain_lstm"
AUTO_RETRAIN_ONLINE_JOB_ID = "auto_retrain_online"
AUTO_RETRAIN_REGIME_JOB_ID = "auto_retrain_regime"


def compute_auto_retrain_interval_seconds() -> int:
    """Otomatik yeniden eğitim aralığını hesaplar — bkz. `Settings.ml_auto_retrain_seconds`
    docstring'i (sabit takvim süresi yerine, eğitim penceresinin ne kadarının
    YENİ veriyle değiştiğine dayalı bir gerekçe). `ml_auto_retrain_seconds`
    açıkça verilmişse (None değilse) doğrudan onu döner."""
    if settings.ml_auto_retrain_seconds is not None:
        return settings.ml_auto_retrain_seconds
    minutes = timeframe_to_minutes(settings.ml_train_timeframe)
    interval = int(settings.ml_train_lookback * minutes * 60 * settings.ml_auto_retrain_refresh_fraction)
    interval = max(interval, 3600)  # en az 1 saat
    return min(interval, settings.ml_auto_retrain_max_seconds)  # en geç ml_auto_retrain_max_seconds (varsayılan 7 gün)


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
    """Periyodik iş (bkz. `FOURKEYS_ML_AUTO_RETRAIN_ENABLED`, varsayılan AÇIK):
    XGBoost'u (ve varsa meta-label modelini) screener'ın top long/short
    listesiyle otomatik olarak yeniden eğitir. Aralık `compute_auto_retrain_interval_seconds()`
    ile hesaplanır (bkz. `app.core.config.Settings.ml_auto_retrain_seconds`
    docstring'i — veri hacmine dayalı gerekçe)."""
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


def _auto_retrain_symbols() -> list[str] | None:
    """`job_auto_retrain` ile AYNI sembol seçimi (screener top long/short) —
    LSTM/online/regime otomatik yenileme job'ları da bunu paylaşır, böylece
    tüm modeller AYNI evrenle senkron kalır."""
    results = refresh_screener()
    picks = top_long(results, settings.screener_top_n) + top_short(results, settings.screener_top_n)
    return [r.symbol for r in picks] or None


def job_auto_retrain_lstm() -> None:
    """Periyodik iş: LSTM'i otomatik yeniden eğitir — YALNIZCA model daha
    önce en az bir kez elle eğitilmişse (disk'te dosyası varsa) çalışır;
    hiç kullanılmayan bir modeli sıfırdan eğitmeye BAŞLAMAZ. Canlı karar
    motorunda kullanılıp kullanılmayacağı (`app.ml.model_status`) HER
    eğitim sonunda otomatik olarak yeniden belirlenir — statik bir "açık/
    kapalı" bayrağı YOKTUR. Aralık `compute_auto_retrain_interval_seconds()`
    ile AYNI (bkz. `Settings.ml_auto_retrain_seconds`).

    Bilinen risk (README'de de belgeli): ağır eğitim işleri şu an ayrı bir
    process'te DEĞİL, aynı uzun ömürlü uvicorn process'i içinde çalışıyor —
    PyTorch'un bellek ayırıcısı belleği işletim sistemine tam geri vermeyebilir,
    tekrarlanan LSTM eğitimleri kümülatif bellek artışına yol açabilir. Bu
    job'un periyodu (varsayılan ~20 gün) bunu pratikte seyrek kılar, ama
    kesin çözüm ayrı bir eğitim process'i/worker'ı (henüz yapılmadı)."""
    if not DEFAULT_LSTM_MODEL_PATH.exists():
        status.record(AUTO_RETRAIN_LSTM_JOB_ID, ok=True, detail="atlandı: LSTM hiç eğitilmemiş")
        return
    try:
        exchange = get_exchange(settings.exchange_id)
        symbols = _auto_retrain_symbols()
        if symbols is None:
            status.record(AUTO_RETRAIN_LSTM_JOB_ID, ok=True, detail="atlandı: screener'dan sembol gelmedi")
            return

        result = train_lstm_signal_model(exchange, symbols)
        status.record(
            AUTO_RETRAIN_LSTM_JOB_ID,
            ok=True,
            detail=f"LSTM: {result.rows_used} satır, oos_balanced_acc={result.out_of_sample.balanced_accuracy:.3f}",
        )
    except Exception as exc:  # noqa: BLE001 - zamanlayıcı thread'i asla çökmemeli
        logger.exception("auto retrain (LSTM) job failed")
        status.record(AUTO_RETRAIN_LSTM_JOB_ID, ok=False, detail=str(exc))


def job_auto_retrain_online() -> None:
    """Periyodik iş: online modeli (river ARF) otomatik yeniden eğitir —
    YALNIZCA model daha önce en az bir kez elle eğitilmişse çalışır. Canlı
    karar motorunda kullanılıp kullanılmayacağı (`app.ml.model_status`) HER
    eğitim sonunda otomatik belirlenir. `river`'ın Hoeffding ağaçları
    XGBoost/LSTM'e göre çok daha hafif eğitildiğinden (bkz. README) bu
    job'un OOM riski YOK."""
    if not DEFAULT_ONLINE_MODEL_PATH.exists():
        status.record(AUTO_RETRAIN_ONLINE_JOB_ID, ok=True, detail="atlandı: online model hiç eğitilmemiş")
        return
    try:
        exchange = get_exchange(settings.exchange_id)
        symbols = _auto_retrain_symbols()
        if symbols is None:
            status.record(AUTO_RETRAIN_ONLINE_JOB_ID, ok=True, detail="atlandı: screener'dan sembol gelmedi")
            return

        _, report = train_online_signal_model(exchange, symbols)
        status.record(
            AUTO_RETRAIN_ONLINE_JOB_ID,
            ok=True,
            detail=f"online: {report.rows_used} satır, overall_balanced_acc={report.overall_balanced_accuracy:.3f}",
        )
    except Exception as exc:  # noqa: BLE001 - zamanlayıcı thread'i asla çökmemeli
        logger.exception("auto retrain (online) job failed")
        status.record(AUTO_RETRAIN_ONLINE_JOB_ID, ok=False, detail=str(exc))


def job_auto_retrain_regime() -> None:
    """Periyodik iş: rejim (GMM) + rejim-başına XGBoost modellerini otomatik
    yeniden eğitir — YALNIZCA daha önce en az bir kez elle eğitilmişse
    (`GET /ml/train-regime` ile) çalışır; canlı karar motoruna henüz
    BAĞLANMADIĞI için (bkz. README) bir ensemble bayrağı yok, tek koşul
    dosyanın varlığı."""
    if not DEFAULT_REGIME_MODEL_PATH.exists():
        status.record(AUTO_RETRAIN_REGIME_JOB_ID, ok=True, detail="atlandı: rejim modeli hiç eğitilmemiş")
        return
    try:
        exchange = get_exchange(settings.exchange_id)
        symbols = _auto_retrain_symbols()
        if symbols is None:
            status.record(AUTO_RETRAIN_REGIME_JOB_ID, ok=True, detail="atlandı: screener'dan sembol gelmedi")
            return

        _, results = train_signal_models_by_regime(exchange, symbols)
        summary = "; ".join(
            f"rejim {r.regime}: {r.rows_used} satır" + (f" (hata: {r.error})" if r.error else "") for r in results
        )
        status.record(AUTO_RETRAIN_REGIME_JOB_ID, ok=True, detail=summary or "sonuç yok")
    except Exception as exc:  # noqa: BLE001 - zamanlayıcı thread'i asla çökmemeli
        logger.exception("auto retrain (regime) job failed")
        status.record(AUTO_RETRAIN_REGIME_JOB_ID, ok=False, detail=str(exc))
