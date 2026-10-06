"""Zorunlu tasfiye akışı (`wss://fstream.binance.com/ws/!forceOrder@arr`): arka plan thread'inde asyncio döngüsü, belirli
aralıklarla toplu DB yazımı, üstel geri çekilmeli yeniden bağlanma, kapanış bayrağı.

Test edilebilirlik: bağlantı fabrikası (`connect`) ve yazıcı (`flush_fn`) enjekte edilir; `process_stream` bir mesaj
akışını thread/ağ olmadan işler."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable

from app.db import repository as db

logger = logging.getLogger(__name__)

URL = "wss://fstream.binance.com/ws/!forceOrder@arr"


def parse_force_order(message: str | bytes | dict) -> dict | None:
    """`{"e":"forceOrder","E":..., "o":{"s","S","q","p","ap","X","z","T"}}` -> satır sözlüğü; geçersizse None."""
    try:
        msg = json.loads(message) if isinstance(message, (str, bytes)) else message
        if msg.get("e") != "forceOrder":
            return None
        o = msg["o"]
        filled = float(o.get("z") or 0.0)
        qty = filled if filled > 0 else float(o["q"])
        avg = float(o.get("ap") or 0.0) or None
        price = float(o["p"])
        return {
            "time": datetime.fromtimestamp(int(o["T"]) / 1000.0, tz=timezone.utc),
            "symbol": str(o["s"]),
            "side": str(o["S"]),
            "price": price,
            "avg_price": avg,
            "quantity": qty,
            "quote_value": qty * (avg or price),
            "status": o.get("X"),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


class LiquidationCollector:
    def __init__(
        self,
        symbols: list[str] | None = None,
        flush_seconds: float = 30.0,
        connect: Callable[[], Any] | None = None,
        flush_fn: Callable[[list[dict]], int] = db.record_liquidation_events,
        sleep: Callable[[float], Any] | None = None,
        max_backoff: float = 60.0,
    ) -> None:
        self.symbols = {s.upper() for s in symbols} if symbols else None  # None = tüm semboller
        self.flush_seconds = flush_seconds
        self._connect = connect
        self._flush_fn = flush_fn
        self._sleep = sleep or asyncio.sleep
        self.max_backoff = max_backoff
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = {"messages": 0, "parsed": 0, "flushed": 0, "reconnects": 0}

    # ---- çekirdek (test edilebilir) --------------------------------------------------------
    def ingest(self, message) -> bool:
        self.stats["messages"] += 1
        row = parse_force_order(message)
        if row is None or (self.symbols is not None and row["symbol"] not in self.symbols):
            return False
        with self._lock:
            self._buffer.append(row)
        self.stats["parsed"] += 1
        return True

    def flush(self) -> int:
        with self._lock:
            rows, self._buffer = self._buffer, []
        if not rows:
            return 0
        try:
            self._flush_fn(rows)
        except Exception:  # noqa: BLE001 - yazım hatası toplayıcıyı durdurmamalı; satırlar geri konur
            logger.exception("liquidation flush başarısız; %d satır yeniden denenecek", len(rows))
            with self._lock:
                self._buffer = rows + self._buffer
            return 0
        self.stats["flushed"] += len(rows)
        return len(rows)

    async def process_stream(self, ws: AsyncIterator) -> None:
        """Bir bağlantının mesajlarını bitene kadar işler; her `flush_seconds`'ta tamponu yazar."""
        last_flush = time.monotonic()
        async for message in ws:
            self.ingest(message)
            if time.monotonic() - last_flush >= self.flush_seconds:
                self.flush()
                last_flush = time.monotonic()
        self.flush()

    async def run_forever(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with self._connect() as ws:
                    backoff = 1.0
                    await self.process_stream(ws)
            except Exception:  # noqa: BLE001 - ağ kopması vb.: geri çekilmeyle yeniden bağlan
                logger.warning("tasfiye akışı koptu; %.0f sn sonra yeniden bağlanılacak", backoff, exc_info=True)
            if self._stop.is_set():
                break
            self.stats["reconnects"] += 1
            await self._sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff)
        self.flush()

    # ---- thread yaşam döngüsü ---------------------------------------------------------------
    def _default_connect(self):
        import websockets

        return websockets.connect(URL, ping_interval=20, ping_timeout=20)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if self._connect is None:
            self._connect = self._default_connect
        self._stop.clear()

        def _run() -> None:
            asyncio.run(self.run_forever())

        self._thread = threading.Thread(target=_run, name="liquidation-collector", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self.flush()


_collector: LiquidationCollector | None = None


def start_liquidation_collector(symbols: list[str], flush_seconds: float) -> LiquidationCollector:
    global _collector
    if _collector is None:
        _collector = LiquidationCollector(symbols, flush_seconds)
    _collector.start()
    return _collector


def stop_liquidation_collector() -> None:
    global _collector
    if _collector is not None:
        _collector.stop()
        _collector = None
