import logging

from app.backtest.data import timeframe_to_minutes
from app.backtest.schemas import SystemBacktestRequest
from app.backtest.system_runner import run_periodic_optimization
from app.core.config import settings
from app.db import repository as db
from app.engine.service import ModelNotTrained, run_cycle_once
from app.exchanges import get_exchange
from app.macro.service import refresh_and_record_macro_snapshot
from app.ml.meta_label import DEFAULT_META_MODEL_PATH, MetaLabelModel
from app.ml.model import DEFAULT_MODEL_PATH, SignalModel
from app.ml.model_paths import DEFAULT_LSTM_MODEL_PATH
from app.ml.model_status import is_model_enabled
from app.ml.online_model import DEFAULT_ONLINE_MODEL_PATH, OnlineSignalModel
from app.ml.regime import DEFAULT_REGIME_MODEL_PATH
from app.ml.train import (
    _ENSEMBLE_LABELING,
    train_lstm_signal_model,
    train_meta_label_model,
    train_online_signal_model,
    train_signal_model_validated,
    train_signal_models_by_regime,
)
from app.openinterest.service import refresh_all_configured_symbols as refresh_open_interest_symbols
from app.orderbook.service import refresh_all_configured_symbols
from app.portfolio.shared import get_portfolio
from app.screener.service import refresh as refresh_screener
from app.security.kill_switch import KillSwitchActive

from . import status

logger = logging.getLogger(__name__)

SCREENER_REFRESH_JOB_ID = "screener_refresh"
ENGINE_CYCLE_JOB_ID = "engine_cycle"
MACRO_REFRESH_JOB_ID = "macro_refresh"
ORDERBOOK_REFRESH_JOB_ID = "orderbook_refresh"
OPEN_INTEREST_REFRESH_JOB_ID = "open_interest_refresh"
AUTO_RETRAIN_JOB_ID = "auto_retrain"
AUTO_RETRAIN_LSTM_JOB_ID = "auto_retrain_lstm"
AUTO_RETRAIN_ONLINE_JOB_ID = "auto_retrain_online"
AUTO_RETRAIN_REGIME_JOB_ID = "auto_retrain_regime"
PERIODIC_OPTIMIZATION_JOB_ID = "periodic_optimization"


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


def job_refresh_open_interest() -> None:
    """Periyodik iş: `feature_snapshot_symbols` ayarındaki sembollerin
    açık pozisyonunun (open interest) bir anlık görüntüsünü alıp kaydeder
    (bkz. `app.openinterest`). Geçmişe dönük open interest verisi yoktur —
    bu tablo yalnızca bugünden itibaren birikir (`job_refresh_orderbook`
    ile AYNI desen)."""
    try:
        exchange = get_exchange(settings.exchange_id)
        results = refresh_open_interest_symbols(exchange, settings.feature_snapshot_symbols_list)
        missing = [symbol for symbol, metrics in results.items() if metrics is None]
        detail = "tüm semboller alındı" if not missing else f"eksik semboller: {', '.join(missing)}"
        status.record(OPEN_INTEREST_REFRESH_JOB_ID, ok=True, detail=detail)
    except Exception as exc:  # noqa: BLE001 - zamanlayıcı thread'i asla çökmemeli
        logger.exception("open interest refresh job failed")
        status.record(OPEN_INTEREST_REFRESH_JOB_ID, ok=False, detail=str(exc))


def job_auto_retrain() -> None:
    """Periyodik iş (bkz. `FOURKEYS_ML_AUTO_RETRAIN_ENABLED`, varsayılan AÇIK):
    XGBoost'u (ve varsa meta-label modelini) `_resolve_symbols` (BTC-öncelikli
    seçim, bkz. `app.api.routes.ml._resolve_symbols`/`_auto_retrain_symbols`
    docstring'i) ile otomatik olarak yeniden eğitir. Aralık
    `compute_auto_retrain_interval_seconds()` ile hesaplanır (bkz.
    `app.core.config.Settings.ml_auto_retrain_seconds` docstring'i — veri
    hacmine dayalı gerekçe).

    GÜNCELLEME (kritik düzeltme, bkz. README "karlılık" — "bu şekilde
    düzeltilmesi gereken şeyler var mı?" kontrolü sırasında bulundu):
    ÖNCEDEN bu job screener'ın ham Top Long/Short çıktısını (likidite/
    korelasyon filtresi YOK) doğrudan eğitim evreni olarak kullanıyor VE
    `_ENSEMBLE_LABELING`i GEÇMİYORDU (yani `train_signal_model_validated`'ın
    KENDİ eski varsayılanlarıyla, horizon=5/1.5xATR, eğitiyordu) — bu,
    `train-all.sh` ile elle doğrulanan BTC-only + horizon=8 üretim
    konfigürasyonunu, en geç `ml_auto_retrain_max_seconds` (varsayılan 7
    gün) içinde SESSİZCE ÜZERİNE YAZARDI. Artık `_resolve_symbols` (BTC-only
    hızlı yol dahil) ve `_ENSEMBLE_LABELING` kullanılıyor — manuel
    `train-all` ile TUTARLI."""
    try:
        exchange = get_exchange(settings.exchange_id)
        symbols = _auto_retrain_symbols(exchange)
        if not symbols:
            status.record(AUTO_RETRAIN_JOB_ID, ok=True, detail="atlandı: sembol bulunamadı")
            return

        train_result = train_signal_model_validated(exchange, symbols, **_ENSEMBLE_LABELING)
        detail = (
            f"XGBoost: {train_result.rows_used} satır, "
            f"oos_balanced_acc={train_result.out_of_sample.balanced_accuracy:.3f}"
        )

        # Meta-label modeli daha önce eğitilmişse (kullanıcı bu katmanı
        # kullanıyor demektir), birincil modelle senkron kalması için o da
        # yenilenir; hiç eğitilmemişse otomatik olarak BAŞLATILMAZ (bu,
        # kullanıcının bilinçli bir tercihi olmalı, bkz. `/ml/train-meta`).
        # KRİTİK: burada da `_ENSEMBLE_LABELING` geçilir — aksi halde
        # meta-label, birincilin ÖĞRENMEDİĞİ bir soruya göre "doğru/yanlış"
        # damgası vurur (bkz. `_ENSEMBLE_LABELING` "KRİTİK" notu).
        if DEFAULT_META_MODEL_PATH.exists():
            try:
                primary_model = SignalModel.load_from()
                _, meta_rows = train_meta_label_model(exchange, symbols, primary_model, **_ENSEMBLE_LABELING)
                detail += f"; meta-label: {meta_rows} satır"
            except ValueError as exc:
                detail += f"; meta-label atlandı: {exc}"

        status.record(AUTO_RETRAIN_JOB_ID, ok=True, detail=detail)
    except Exception as exc:  # noqa: BLE001 - zamanlayıcı thread'i asla çökmemeli
        logger.exception("auto retrain job failed")
        status.record(AUTO_RETRAIN_JOB_ID, ok=False, detail=str(exc))


def _auto_retrain_symbols(exchange) -> list[str] | None:
    """`job_auto_retrain` ile AYNI sembol seçimi — LSTM/online/regime
    otomatik yenileme job'ları da bunu paylaşır, böylece tüm modeller AYNI
    evrenle senkron kalır.

    GÜNCELLEME (kritik düzeltme — bkz. `job_auto_retrain` docstring'i):
    ÖNCEDEN screener'ın ham Top Long/Short çıktısını (likidite/korelasyon
    filtresi YOK, `ml_train_max_symbols` ayarını HİÇ dikkate almıyordu)
    kullanıyordu. Artık `_resolve_symbols` (manuel `train-all`/`POST
    /ml/train-all` ile AYNI fonksiyon — BTC-only hızlı yol dahil, likidite/
    korelasyon filtreli `select_training_symbols`) kullanılıyor."""
    from app.api.routes.ml import _resolve_symbols

    return _resolve_symbols(exchange, None) or None


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
        symbols = _auto_retrain_symbols(exchange)
        if symbols is None:
            status.record(AUTO_RETRAIN_LSTM_JOB_ID, ok=True, detail="atlandı: sembol bulunamadı")
            return

        result = train_lstm_signal_model(exchange, symbols, **_ENSEMBLE_LABELING)
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
        symbols = _auto_retrain_symbols(exchange)
        if symbols is None:
            status.record(AUTO_RETRAIN_ONLINE_JOB_ID, ok=True, detail="atlandı: sembol bulunamadı")
            return

        _, report = train_online_signal_model(exchange, symbols, **_ENSEMBLE_LABELING)
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
        symbols = _auto_retrain_symbols(exchange)
        if symbols is None:
            status.record(AUTO_RETRAIN_REGIME_JOB_ID, ok=True, detail="atlandı: sembol bulunamadı")
            return

        _, results = train_signal_models_by_regime(exchange, symbols, **_ENSEMBLE_LABELING)
        summary = "; ".join(
            f"rejim {r.regime}: {r.rows_used} satır" + (f" (hata: {r.error})" if r.error else "") for r in results
        )
        status.record(AUTO_RETRAIN_REGIME_JOB_ID, ok=True, detail=summary or "sonuç yok")
    except Exception as exc:  # noqa: BLE001 - zamanlayıcı thread'i asla çökmemeli
        logger.exception("auto retrain (regime) job failed")
        status.record(AUTO_RETRAIN_REGIME_JOB_ID, ok=False, detail=str(exc))


def _dampened_step(current: float, recommended: float, fraction: float) -> float:
    """Mevcut ile önerilen arasındaki mesafenin yalnızca `fraction`'ını
    uygular — bkz. `Settings.ml_periodic_optimization_max_step_fraction`
    docstring'i: tek bir gürültülü haftanın parametreleri uçtan uca
    sıçratmasını önler."""
    return current + fraction * (recommended - current)


def job_periodic_optimization() -> None:
    """Periyodik iş (bkz. `Settings.ml_periodic_optimization_enabled`,
    varsayılan AÇIK, haftalık): `app.backtest.system_runner.run_periodic_optimization`
    ile güncel modelin güven eşiği + Kelly boyutlandırma parametrelerini
    tarayıp SONUCU `optimization_runs` tablosuna kaydeder (`GET
    /backtest/system/optimization-history`). YALNIZCA birincil (XGBoost)
    model daha önce eğitilmişse çalışır.

    `Settings.ml_periodic_optimization_auto_apply_enabled` AÇIKSA (kullanıcı
    isteği: "şimdi kur"), öneri iki güvenlik kapısından GEÇERSE CANLI
    ayarlara (`settings.live_open_confidence`/`live_close_confidence`,
    `get_portfolio().rules.kelly_min_trades`/`.kelly_multiplier`) — TAM
    değil, KADEMELİ (bkz. `_dampened_step`) olarak — uygulanır: (1) öneri
    yeterli örneklemli olmalı (bkz. `run_periodic_optimization`'ın kendi
    güvenilirlik filtresi — filtrelenmişse zaten `recommended == current`
    döner, bu kapı fiilen otomatik sağlanır), (2) mevcut PnL'den en az
    `ml_periodic_optimization_min_improvement_pct` kadar İYİ olmalı (aksi
    halde gürültü farkı yüzünden gereksiz churn olur)."""
    if not DEFAULT_MODEL_PATH.exists():
        status.record(PERIODIC_OPTIMIZATION_JOB_ID, ok=True, detail="atlandı: birincil model hiç eğitilmemiş")
        return
    try:
        exchange = get_exchange(settings.exchange_id)
        model = SignalModel.load_from()
        meta_model = MetaLabelModel.load_from() if DEFAULT_META_MODEL_PATH.exists() else None
        lstm_model = None
        if is_model_enabled(DEFAULT_LSTM_MODEL_PATH):
            from app.ml.lstm_model import LSTMSignalModel  # lazy — bkz. app.ml.model_paths docstring'i

            lstm_model = LSTMSignalModel.load_from()
        online_model = OnlineSignalModel.load_from() if is_model_enabled(DEFAULT_ONLINE_MODEL_PATH) else None

        portfolio_rules = get_portfolio().rules
        # ŞEMANIN KENDİ varsayılanları DEĞİL — o an CANLIDA GERÇEKTEN
        # kullanılan değerler (bkz. yukarıdaki docstring: auto-apply
        # zaten bunları çalışma zamanında değiştirmiş olabilir, bir
        # sonraki haftanın "mevcut"u DOĞRU raporlanmalı).
        base_request = SystemBacktestRequest(
            symbol=settings.ml_primary_symbol,
            open_confidence=settings.live_open_confidence,
            close_confidence=settings.live_close_confidence,
            kelly_min_trades=portfolio_rules.kelly_min_trades,
            kelly_multiplier=portfolio_rules.kelly_multiplier,
        )
        result = run_periodic_optimization(
            exchange, model, meta_model, base_request, lstm_model=lstm_model, online_model=online_model
        )

        applied = False
        if settings.ml_periodic_optimization_auto_apply_enabled and (
            result.recommended_total_pnl_pct
            >= result.current_total_pnl_pct + settings.ml_periodic_optimization_min_improvement_pct
        ):
            fraction = settings.ml_periodic_optimization_max_step_fraction
            settings.live_open_confidence = _dampened_step(result.current_open_confidence, result.recommended_open_confidence, fraction)
            settings.live_close_confidence = _dampened_step(result.current_close_confidence, result.recommended_close_confidence, fraction)
            portfolio_rules.kelly_multiplier = _dampened_step(result.current_kelly_multiplier, result.recommended_kelly_multiplier, fraction)
            portfolio_rules.kelly_min_trades = max(5, round(_dampened_step(result.current_kelly_min_trades, result.recommended_kelly_min_trades, fraction)))
            applied = True

        db.record_optimization_run(
            {
                "symbol": result.symbol,
                "recommended_open_confidence": result.recommended_open_confidence,
                "recommended_close_confidence": result.recommended_close_confidence,
                "recommended_kelly_min_trades": result.recommended_kelly_min_trades,
                "recommended_kelly_multiplier": result.recommended_kelly_multiplier,
                "recommended_trades_closed": result.recommended_trades_closed,
                "recommended_win_rate_pct": result.recommended_win_rate_pct,
                "recommended_total_pnl_pct": result.recommended_total_pnl_pct,
                "recommended_max_drawdown_pct": result.recommended_max_drawdown_pct,
                "current_open_confidence": result.current_open_confidence,
                "current_close_confidence": result.current_close_confidence,
                "current_kelly_min_trades": result.current_kelly_min_trades,
                "current_kelly_multiplier": result.current_kelly_multiplier,
                "current_trades_closed": result.current_trades_closed,
                "current_win_rate_pct": result.current_win_rate_pct,
                "current_total_pnl_pct": result.current_total_pnl_pct,
                "current_max_drawdown_pct": result.current_max_drawdown_pct,
                "applied": applied,
            }
        )
        applied_note = (
            f"UYGULANDI (kademeli, {settings.ml_periodic_optimization_max_step_fraction:.0%} adım) -> "
            f"yeni canlı: eşik={settings.live_open_confidence:.3f}/{settings.live_close_confidence:.3f}, "
            f"kelly={portfolio_rules.kelly_min_trades}/{portfolio_rules.kelly_multiplier:.3f}"
            if applied
            else "CANLI AYARLAR DEĞİŞTİRİLMEDİ (iyileşme eşiği geçilmedi veya auto-apply kapalı)"
        )
        status.record(
            PERIODIC_OPTIMIZATION_JOB_ID,
            ok=True,
            detail=(
                f"mevcut: eşik={result.current_open_confidence}/{result.current_close_confidence}, "
                f"kelly={result.current_kelly_min_trades}/{result.current_kelly_multiplier}, "
                f"PnL=%{result.current_total_pnl_pct:.2f} | önerilen: "
                f"eşik={result.recommended_open_confidence}/{result.recommended_close_confidence}, "
                f"kelly={result.recommended_kelly_min_trades}/{result.recommended_kelly_multiplier}, "
                f"PnL=%{result.recommended_total_pnl_pct:.2f} ({result.recommended_trades_closed} işlem) "
                f"— {applied_note}"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - zamanlayıcı thread'i asla çökmemeli
        logger.exception("periodic optimization job failed")
        status.record(PERIODIC_OPTIMIZATION_JOB_ID, ok=False, detail=str(exc))
