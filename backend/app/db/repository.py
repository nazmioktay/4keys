import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from .models import OHLCVRaw, SignalRecord, TradeRecord
from .session import is_enabled, session_scope

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
            db.add(SignalRecord(symbol=symbol, source=source, direction=direction, confidence=confidence, price=price))
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
                    entry_price=trade["entry_price"],
                    exit_price=trade["exit_price"],
                    size_quote=trade.get("size_quote", 0.0),
                    pnl_pct=trade["pnl_pct"],
                    pnl_quote=trade.get("pnl_quote", 0.0),
                    opened_at=datetime.fromisoformat(trade["opened_at"]),
                    closed_at=datetime.fromisoformat(trade["closed_at"]),
                    source=trade.get("source", "engine"),
                )
            )
    except SQLAlchemyError:
        logger.exception("trade persist failed for %s", trade.get("symbol"))


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
