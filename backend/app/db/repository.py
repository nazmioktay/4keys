import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.ml.features import FEATURE_COLUMNS

from .models import (
    BacktestRun,
    BacktestTradeRow,
    FeatureSnapshot,
    MacroSnapshot,
    OHLCVRaw,
    OrderbookSnapshot,
    SignalRecord,
    TradeRecord,
)
from .session import is_enabled, session_scope

# `feature_snapshots` tablosunun kolonları, ML modelinin kullandığı
# özellik listesiyle (bkz. app.ml.features.FEATURE_COLUMNS) birebir
# senkron tutulur — "close" ayrıca ele alınır, feature listesinde değildir.
FEATURE_SNAPSHOT_COLUMNS = list(FEATURE_COLUMNS)

MACRO_SNAPSHOT_COLUMNS = [
    "total_market_cap",
    "btc_dominance",
    "funding_rate_btc",
    "vix",
    "gold_price",
    "sp500",
    "nasdaq",
    "nikkei",
    "dax",
    "fed_funds_rate",
    "ecb_deposit_rate",
]

ORDERBOOK_SNAPSHOT_COLUMNS = ["bid_volume", "ask_volume", "imbalance", "spread_pct"]

logger = logging.getLogger(__name__)


def _to_pydatetime(value) -> datetime:
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def record_latest_candle(symbol: str, timeframe: str, row: pd.Series) -> None:
    """Bir sembolün en son mumunu `ohlcv_raw`'a kaydeder.

    Kalıcılık bir yan etkidir, ana işlem akışının bir ön koşulu değildir —
    veritabanı kapalı/erişilemez olsa bile screener/motor çalışmaya devam
    etmelidir. Bu yüzden burada hiçbir hata dışarı fırlatılmaz, sadece loglanır.
    """
    if not is_enabled():
        return
    try:
        with session_scope() as db:
            db.add(
                OHLCVRaw(
                    time=_to_pydatetime(row["timestamp"]),
                    symbol=symbol,
                    timeframe=timeframe,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    except IntegrityError:
        pass  # bu mum zaten kaydedilmiş (aynı time/symbol/timeframe)
    except SQLAlchemyError:
        logger.exception("ohlcv persist failed for %s", symbol)


def record_signal(symbol: str, source: str, direction: str, confidence: float, price: float) -> None:
    if not is_enabled():
        return
    try:
        with session_scope() as db:
            db.add(
                SignalRecord(
                    symbol=symbol, source=source, direction=direction, confidence=float(confidence), price=float(price)
                )
            )
    except SQLAlchemyError:
        logger.exception("signal persist failed for %s", symbol)


def record_trade(trade: dict) -> None:
    """`PortfolioManager.close()`'un ürettiği kayıt sözlüğünü doğrudan kabul eder."""
    if not is_enabled():
        return
    try:
        with session_scope() as db:
            db.add(
                TradeRecord(
                    symbol=trade["symbol"],
                    direction=trade["direction"],
                    entry_price=float(trade["entry_price"]),
                    exit_price=float(trade["exit_price"]),
                    size_quote=float(trade.get("size_quote", 0.0)),
                    pnl_pct=float(trade["pnl_pct"]),
                    pnl_quote=float(trade.get("pnl_quote", 0.0)),
                    opened_at=datetime.fromisoformat(trade["opened_at"]),
                    closed_at=datetime.fromisoformat(trade["closed_at"]),
                    source=trade.get("source", "engine"),
                )
            )
    except SQLAlchemyError:
        logger.exception("trade persist failed for %s", trade.get("symbol"))


def record_feature_snapshot(symbol: str, timeframe: str, time, feature_row: pd.Series) -> None:
    """Bir sembolün en son (tam dolu) ML özellik vektörünü `feature_snapshots`
    tablosuna kaydeder — bkz. `app.ml.features.FEATURE_COLUMNS`.

    `time`, bu özellik vektörünün hesaplandığı mumun zaman damgasıdır
    (çağıran taraf, kendi ham OHLCV'sinden geçirir).

    Kalıcılık burada da bir yan etkidir: DB kapalı/erişilemez olsa bile
    screener çalışmaya devam eder, hata sadece loglanır.
    """
    if not is_enabled():
        return
    try:
        with session_scope() as db:
            db.add(
                FeatureSnapshot(
                    time=_to_pydatetime(time),
                    symbol=symbol,
                    timeframe=timeframe,
                    close=float(feature_row["close"]),
                    **{col: float(feature_row[col]) for col in FEATURE_SNAPSHOT_COLUMNS},
                )
            )
    except IntegrityError:
        pass  # bu zaman damgası için zaten kaydedilmiş
    except SQLAlchemyError:
        logger.exception("feature snapshot persist failed for %s", symbol)


def record_feature_snapshots_bulk(symbol: str, timeframe: str, frame: pd.DataFrame) -> int:
    """`frame`'deki (bkz. `app.ml.features.build_features` çıktısı — "timestamp"
    (int64 ns epoch) ve "close" ile FEATURE_SNAPSHOT_COLUMNS kolonlarını
    içermeli) TÜM satırları tek seferde `feature_snapshots`'a yazar.

    ML eğitimi zaten her seferinde Binance'ten geniş bir geçmiş (bkz.
    `Settings.ml_train_lookback`) çektiği için, bu geçmişi ayrıca
    `feature_snapshots`'a da yazmak LSTM/RL için gereken uzun/kesintisiz
    zaman serisinin aylarca sürecek periyodik birikim yerine TEK SEFERDE
    "backfill" edilmesini sağlar.

    Zaten kayıtlı (time, symbol, timeframe) satırları sessizce atlanır
    (ON CONFLICT DO NOTHING) — aynı geçmişi tekrar tekrar eğitmek/çağırmak
    güvenlidir, yalnızca gerçekten yeni barlar eklenir. Kalıcılık bir yan
    etkidir: DB kapalı/erişilemez olsa da eğitim akışını bozmaz."""
    if not is_enabled() or frame.empty:
        return 0
    required = {"timestamp", "close", *FEATURE_SNAPSHOT_COLUMNS}
    if not required.issubset(frame.columns):
        return 0

    rows = []
    for record in frame[["timestamp", "close", *FEATURE_SNAPSHOT_COLUMNS]].to_dict("records"):
        try:
            rows.append(
                {
                    "time": pd.Timestamp(record["timestamp"], unit="ns").to_pydatetime(),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "close": float(record["close"]),
                    **{col: float(record[col]) for col in FEATURE_SNAPSHOT_COLUMNS},
                }
            )
        except (TypeError, ValueError):
            continue
    if not rows:
        return 0

    try:
        with session_scope() as db:
            table = FeatureSnapshot.__table__
            dialect = db.bind.dialect.name if db.bind is not None else ""
            if dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as upsert_insert
            elif dialect == "sqlite":
                from sqlalchemy.dialects.sqlite import insert as upsert_insert
            else:
                upsert_insert = None

            if upsert_insert is not None:
                stmt = upsert_insert(table).values(rows)
                stmt = stmt.on_conflict_do_nothing(index_elements=["time", "symbol", "timeframe"])
                db.execute(stmt)
            else:
                for row in rows:
                    try:
                        db.add(FeatureSnapshot(**row))
                        db.flush()
                    except IntegrityError:
                        db.rollback()
        return len(rows)
    except SQLAlchemyError:
        logger.exception("bulk feature snapshot persist failed for %s", symbol)
        return 0


def get_feature_snapshots(symbol: str, timeframe: str, limit: int = 5000) -> pd.DataFrame:
    """Bir sembol için biriken ML özellik vektörlerini kronolojik sırayla
    (en eskiden en yeniye) döner — LSTM/RL eğitiminde kullanılacak zaman
    serisi veri setinin kaynağı. DB kapalıysa veya kayıt yoksa boş DataFrame.
    """
    if not is_enabled():
        return pd.DataFrame(columns=["time", "symbol", "close", *FEATURE_SNAPSHOT_COLUMNS])
    try:
        with session_scope() as db:
            query = (
                select(FeatureSnapshot)
                .where(FeatureSnapshot.symbol == symbol, FeatureSnapshot.timeframe == timeframe)
                .order_by(FeatureSnapshot.time.desc())
                .limit(limit)
            )
            rows = db.execute(query).scalars().all()
            records = [
                {
                    "time": r.time,
                    "symbol": r.symbol,
                    "close": r.close,
                    **{col: getattr(r, col) for col in FEATURE_SNAPSHOT_COLUMNS},
                }
                for r in reversed(rows)
            ]
            return pd.DataFrame(records)
    except SQLAlchemyError:
        logger.exception("failed to read feature snapshots for %s", symbol)
        return pd.DataFrame(columns=["time", "symbol", "close", *FEATURE_SNAPSHOT_COLUMNS])


def record_macro_snapshot(values: dict) -> None:
    """Ücretsiz makro veri kaynaklarının (bkz. `app.macro.data`) bir anlık
    görüntüsünü `macro_snapshots` tablosuna kaydeder. Kaynakların bir kısmı
    None olabilir (o kaynak o an erişilemedi) — bu normaldir, satır yine de
    kaydedilir; eksik alanlar NULL kalır."""
    if not is_enabled():
        return
    try:
        with session_scope() as db:
            db.add(MacroSnapshot(**{col: values.get(col) for col in MACRO_SNAPSHOT_COLUMNS}))
    except SQLAlchemyError:
        logger.exception("macro snapshot persist failed")


def get_latest_macro_snapshot() -> dict | None:
    if not is_enabled():
        return None
    try:
        with session_scope() as db:
            row = db.execute(select(MacroSnapshot).order_by(MacroSnapshot.time.desc()).limit(1)).scalar_one_or_none()
            if row is None:
                return None
            return {"time": row.time.isoformat(), **{col: getattr(row, col) for col in MACRO_SNAPSHOT_COLUMNS}}
    except SQLAlchemyError:
        logger.exception("failed to read latest macro snapshot")
        return None


def get_macro_snapshots(limit: int = 500) -> pd.DataFrame:
    if not is_enabled():
        return pd.DataFrame(columns=["time", *MACRO_SNAPSHOT_COLUMNS])
    try:
        with session_scope() as db:
            rows = db.execute(select(MacroSnapshot).order_by(MacroSnapshot.time.desc()).limit(limit)).scalars().all()
            records = [{"time": r.time, **{col: getattr(r, col) for col in MACRO_SNAPSHOT_COLUMNS}} for r in reversed(rows)]
            return pd.DataFrame(records)
    except SQLAlchemyError:
        logger.exception("failed to read macro snapshots")
        return pd.DataFrame(columns=["time", *MACRO_SNAPSHOT_COLUMNS])


def record_orderbook_snapshot(symbol: str, values: dict) -> None:
    """Bir sembolün emir defteri özetinin (bkz. `app.orderbook.data`) bir
    anlık görüntüsünü `orderbook_snapshots` tablosuna kaydeder. Geçmişe
    dönük emir defteri verisi yoktur — bu tablo yalnızca bugünden itibaren
    periyodik toplamayla birikir (`macro_snapshots` ile aynı desen, ama
    sembol bazında)."""
    if not is_enabled():
        return
    try:
        with session_scope() as db:
            db.add(OrderbookSnapshot(symbol=symbol, **{col: values.get(col) for col in ORDERBOOK_SNAPSHOT_COLUMNS}))
    except SQLAlchemyError:
        logger.exception("orderbook snapshot persist failed for %s", symbol)


def get_latest_orderbook_snapshot(symbol: str) -> dict | None:
    if not is_enabled():
        return None
    try:
        with session_scope() as db:
            row = db.execute(
                select(OrderbookSnapshot)
                .where(OrderbookSnapshot.symbol == symbol)
                .order_by(OrderbookSnapshot.time.desc())
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            return {"time": row.time.isoformat(), "symbol": row.symbol, **{col: getattr(row, col) for col in ORDERBOOK_SNAPSHOT_COLUMNS}}
    except SQLAlchemyError:
        logger.exception("failed to read latest orderbook snapshot for %s", symbol)
        return None


def get_orderbook_snapshots(symbol: str, limit: int = 5000) -> pd.DataFrame:
    if not is_enabled():
        return pd.DataFrame(columns=["time", "symbol", *ORDERBOOK_SNAPSHOT_COLUMNS])
    try:
        with session_scope() as db:
            rows = db.execute(
                select(OrderbookSnapshot)
                .where(OrderbookSnapshot.symbol == symbol)
                .order_by(OrderbookSnapshot.time.desc())
                .limit(limit)
            ).scalars().all()
            records = [
                {"time": r.time, "symbol": r.symbol, **{col: getattr(r, col) for col in ORDERBOOK_SNAPSHOT_COLUMNS}}
                for r in reversed(rows)
            ]
            return pd.DataFrame(records)
    except SQLAlchemyError:
        logger.exception("failed to read orderbook snapshots for %s", symbol)
        return pd.DataFrame(columns=["time", "symbol", *ORDERBOOK_SNAPSHOT_COLUMNS])


def get_all_orderbook_snapshots(limit: int = 200_000) -> pd.DataFrame:
    """Tüm sembollerin emir defteri geçmişini (as-of merge için) döner."""
    if not is_enabled():
        return pd.DataFrame(columns=["time", "symbol", *ORDERBOOK_SNAPSHOT_COLUMNS])
    try:
        with session_scope() as db:
            rows = db.execute(select(OrderbookSnapshot).order_by(OrderbookSnapshot.time.desc()).limit(limit)).scalars().all()
            records = [
                {"time": r.time, "symbol": r.symbol, **{col: getattr(r, col) for col in ORDERBOOK_SNAPSHOT_COLUMNS}}
                for r in reversed(rows)
            ]
            return pd.DataFrame(records)
    except SQLAlchemyError:
        logger.exception("failed to read all orderbook snapshots")
        return pd.DataFrame(columns=["time", "symbol", *ORDERBOOK_SNAPSHOT_COLUMNS])


def get_recent_trades(limit: int = 50) -> list[dict]:
    if not is_enabled():
        return []
    try:
        with session_scope() as db:
            rows = db.execute(select(TradeRecord).order_by(TradeRecord.closed_at.desc()).limit(limit)).scalars().all()
            return [
                {
                    "symbol": r.symbol,
                    "direction": r.direction,
                    "entry_price": r.entry_price,
                    "exit_price": r.exit_price,
                    "size_quote": r.size_quote,
                    "pnl_pct": r.pnl_pct,
                    "pnl_quote": r.pnl_quote,
                    "opened_at": r.opened_at.isoformat(),
                    "closed_at": r.closed_at.isoformat(),
                    "source": r.source,
                }
                for r in rows
            ]
    except SQLAlchemyError:
        logger.exception("failed to read recent trades")
        return []


def get_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """`ohlcv_raw`'dan bir sembolün en son `limit` mumunu kronolojik sırayla
    (eskiden yeniye) döner — bkz. `app.exchanges.cache.fetch_ohlcv_cached`:
    borsadan HER seferinde tam geçmişi (ör. 10.000 mum) tekrar tekrar
    çekmek yerine, önceden kaydedilmiş geçmiş buradan okunur; yalnızca
    EKSİK/YENİ kuyruk borsadan çekilir. DB kapalı/erişilemezse boş
    DataFrame (çağıran taraf bu durumda doğrudan borsaya düşer)."""
    if not is_enabled():
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    try:
        with session_scope() as db:
            rows = (
                db.execute(
                    select(OHLCVRaw)
                    .where(OHLCVRaw.symbol == symbol, OHLCVRaw.timeframe == timeframe)
                    .order_by(OHLCVRaw.time.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            records = [
                {
                    # `OHLCVRaw.time` (DateTime(timezone=True)) tz-aware bir
                    # datetime döner — borsadan gelen (`Exchange.fetch_ohlcv`)
                    # zaman damgaları HER ZAMAN tz-naive'dir; bu uyumsuzluk
                    # `build_features()`'ın `.astype("datetime64[ns]")`
                    # dönüşümünü patlatıyordu (bkz. session notu — bu, ilk
                    # önbellek-sıcak okumada TÜM sembollerin sessizce
                    # "yeterli veri yok"a düşmesine yol açan gerçek bir
                    # regresyondu). Değer zaten UTC anlık değeri olduğundan
                    # `tzinfo`'yu YALNIZCA düşürmek (dönüştürmeden) doğru.
                    "timestamp": r.time.replace(tzinfo=None) if r.time.tzinfo is not None else r.time,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                }
                for r in reversed(rows)
            ]
            return pd.DataFrame(records, columns=["timestamp", "open", "high", "low", "close", "volume"])
    except SQLAlchemyError:
        logger.exception("failed to read ohlcv history for %s", symbol)
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def save_ohlcv_bulk(symbol: str, timeframe: str, ohlcv: pd.DataFrame) -> int:
    """`ohlcv_raw`'a bir OHLCV DataFrame'ini (timestamp/open/high/low/close/
    volume kolonları) tek seferde yazar — `app.backtest.system_runner`,
    Grafana'nın candlestick panelinin okuyabilmesi için backtest'te
    kullanılan geçmişi buraya "backfill" eder (bkz.
    `record_feature_snapshots_bulk` aynı ON CONFLICT DO NOTHING deseni).
    Zaten kayıtlı (time, symbol, timeframe) satırlar sessizce atlanır."""
    if not is_enabled() or ohlcv.empty:
        return 0

    rows = [
        {
            "time": _to_pydatetime(pd.Timestamp(record["timestamp"])),
            "symbol": symbol,
            "timeframe": timeframe,
            "open": float(record["open"]),
            "high": float(record["high"]),
            "low": float(record["low"]),
            "close": float(record["close"]),
            "volume": float(record["volume"]),
        }
        for record in ohlcv[["timestamp", "open", "high", "low", "close", "volume"]].to_dict("records")
    ]
    if not rows:
        return 0

    try:
        with session_scope() as db:
            table = OHLCVRaw.__table__
            dialect = db.bind.dialect.name if db.bind is not None else ""
            if dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as upsert_insert
            elif dialect == "sqlite":
                from sqlalchemy.dialects.sqlite import insert as upsert_insert
            else:
                upsert_insert = None

            if upsert_insert is not None:
                stmt = upsert_insert(table).values(rows)
                stmt = stmt.on_conflict_do_nothing(index_elements=["time", "symbol", "timeframe"])
                db.execute(stmt)
            else:
                for row in rows:
                    try:
                        db.add(OHLCVRaw(**row))
                        db.flush()
                    except IntegrityError:
                        db.rollback()
        return len(rows)
    except SQLAlchemyError:
        logger.exception("bulk ohlcv persist failed for %s", symbol)
        return 0


def save_backtest_run(run: dict, trades: list[dict]) -> int | None:
    """Bir sistem backtest çalıştırmasının özetini + işlem listesini
    kaydeder, oluşturulan `BacktestRun.id`'yi döner (DB kapalıysa None).
    Frontend'in `GET /backtest/system/latest` ile geri okuması VE
    Grafana'nın candlestick/PnL panelleri bu tabloları kullanır."""
    if not is_enabled():
        return None
    try:
        with session_scope() as db:
            run_row = BacktestRun(**run)
            db.add(run_row)
            db.flush()  # id'yi almak için
            run_id = run_row.id
            for trade in trades:
                db.add(BacktestTradeRow(run_id=run_id, **trade))
        return run_id
    except SQLAlchemyError:
        logger.exception("backtest run persist failed")
        return None


def get_latest_backtest_run(symbol: str | None = None) -> dict | None:
    if not is_enabled():
        return None
    try:
        with session_scope() as db:
            query = select(BacktestRun).order_by(BacktestRun.created_at.desc())
            if symbol:
                query = query.where(BacktestRun.symbol == symbol)
            run = db.execute(query.limit(1)).scalar_one_or_none()
            if run is None:
                return None
            trades = db.execute(
                select(BacktestTradeRow)
                .where(BacktestTradeRow.run_id == run.id)
                .order_by(BacktestTradeRow.exit_time.asc())
            ).scalars().all()
            return {
                "id": run.id,
                "created_at": run.created_at.isoformat(),
                "symbol": run.symbol,
                "timeframe": run.timeframe,
                "candles_used": run.candles_used,
                "period_start": run.period_start.isoformat(),
                "period_end": run.period_end.isoformat(),
                "initial_balance": run.initial_balance,
                "final_equity": run.final_equity,
                "trades_closed": run.trades_closed,
                "win_rate_pct": run.win_rate_pct,
                "total_pnl_quote": run.total_pnl_quote,
                "total_pnl_pct": run.total_pnl_pct,
                "daily_pnl_quote": run.daily_pnl_quote,
                "daily_pnl_pct": run.daily_pnl_pct,
                "monthly_pnl_quote": run.monthly_pnl_quote,
                "monthly_pnl_pct": run.monthly_pnl_pct,
                "max_drawdown_pct": run.max_drawdown_pct,
                "trades": [
                    {
                        "direction": t.direction,
                        "entry_time": t.entry_time.isoformat(),
                        "exit_time": t.exit_time.isoformat(),
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "pnl_pct": t.pnl_pct,
                        "pnl_quote": t.pnl_quote,
                        "equity_after": t.equity_after,
                        "exit_reason": t.exit_reason,
                        "duration_candles": t.duration_candles,
                    }
                    for t in trades
                ],
            }
    except SQLAlchemyError:
        logger.exception("failed to read latest backtest run")
        return None


def get_recent_signals(limit: int = 50, symbol: str | None = None, source: str | None = None) -> list[dict]:
    if not is_enabled():
        return []
    try:
        with session_scope() as db:
            query = select(SignalRecord).order_by(SignalRecord.time.desc())
            if symbol:
                query = query.where(SignalRecord.symbol == symbol)
            if source:
                query = query.where(SignalRecord.source == source)
            rows = db.execute(query.limit(limit)).scalars().all()
            return [
                {
                    "time": r.time.isoformat(),
                    "symbol": r.symbol,
                    "source": r.source,
                    "direction": r.direction,
                    "confidence": r.confidence,
                    "price": r.price,
                }
                for r in rows
            ]
    except SQLAlchemyError:
        logger.exception("failed to read recent signals")
        return []
