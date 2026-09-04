"""İş mantığına özgü (business) Prometheus metrikleri — HTTP istek
metrikleri zaten `starlette-exporter` tarafından otomatik toplanıyor
(bkz. `app.main`); bunlar `/metrics`'te AYNI (varsayılan) registry'ye
otomatik eklenir, ekstra kablolama gerekmez.

Bu metrikler asla ana iş akışını bozmamalı — `.set()`/`.inc()` çağrıları
sarmalanmaz çünkü prometheus_client'ın kendisi pratikte hata fırlatmaz
(sabit, önceden tanımlı metrik nesneleri üzerinde çalışırlar)."""

from prometheus_client import Counter, Gauge

portfolio_equity = Gauge("fourkeys_portfolio_equity", "Paper-trading portföyünün güncel değeri (quote para birimi)")
portfolio_open_positions = Gauge("fourkeys_portfolio_open_positions", "Açık pozisyon sayısı")
portfolio_realized_pnl_session = Gauge("fourkeys_portfolio_realized_pnl_session", "Oturum içi gerçekleşen PNL")

trades_closed_total = Counter(
    "fourkeys_trades_closed_total", "Kapanan işlem sayısı (kademeli dilimler dahil, her dilim ayrı sayılır)", ["direction", "result"]
)

ml_prediction_confidence = Gauge(
    "fourkeys_ml_prediction_confidence", "Son ML tahmininin güven skoru", ["symbol", "direction"]
)

scheduler_job_success = Gauge(
    "fourkeys_scheduler_job_last_run_ok", "Zamanlayıcı işinin son çalışmasının başarılı olup olmadığı (1/0)", ["job_id"]
)
scheduler_job_last_run_timestamp = Gauge(
    "fourkeys_scheduler_job_last_run_timestamp_seconds", "Zamanlayıcı işinin son çalıştığı Unix zaman damgası", ["job_id"]
)


def record_trade_closed(direction: str, pnl_pct: float) -> None:
    result = "win" if pnl_pct > 0 else "loss" if pnl_pct < 0 else "flat"
    trades_closed_total.labels(direction=direction, result=result).inc()


def record_ml_prediction(symbol: str, direction: str, confidence: float) -> None:
    ml_prediction_confidence.labels(symbol=symbol, direction=direction).set(confidence)


def record_scheduler_job(job_id: str, ok: bool) -> None:
    import time

    scheduler_job_success.labels(job_id=job_id).set(1.0 if ok else 0.0)
    scheduler_job_last_run_timestamp.labels(job_id=job_id).set(time.time())
