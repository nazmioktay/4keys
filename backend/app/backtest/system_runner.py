import numpy as np
import pandas as pd

from app.db import repository as db
from app.exchanges.base import Exchange
from app.ml.features import build_features
from app.ml.meta_label import MetaLabelModel
from app.ml.model import SignalModel

from .data import fetch_full_history
from .schemas import SystemBacktestReport, SystemBacktestRequest

_DIRECTION_MAP = {1: "long", -1: "short", 0: "neutral"}

# Göstergelerin (Ichimoku ~52, Hurst penceresi 100 vb.) ısınması için
# gereken minimum bar sayısı — bu kadar veri olmadan anlamlı bir backtest
# çalıştırılamaz (bkz. app.backtest.runner._discover_sufficient_history
# aynı `len(ohlcv) < 30` kontrolünün, burada göstergelerin ısınma
# gereksinimine göre büyütülmüş hali).
_MIN_CANDLES = 250


def run_system_backtest(
    exchange: Exchange,
    model: SignalModel,
    meta_model: MetaLabelModel | None,
    request: SystemBacktestRequest,
) -> SystemBacktestReport:
    """Canlı karar motorunun kullandığı AYNI birincil modeli (+ varsa
    meta-label filtresi), `request.symbol` için gerçek geçmiş mumlar
    üzerinde bar-bar tekrar oynatır.

    Bilinen basitleştirmeler (dürüstçe belgelenir, `warnings` alanına da
    yansır):
    - Makro/order-book/taker-flow özellikleri (bkz. `ALL_FEATURE_COLUMNS`)
      burada hesaplanmaz — bunlar yalnızca "ANLIK" değerlerdir, geçmiş
      barlar için tarihsel bir zaman serisi YOKTUR. `_select_features`
      eksik kolonları zaten 0.0 (nötr) ile dolduruyor — bu, eğitim
      setindeki makro geçmişi kısa/yok olan ESKİ barlara UYGULANAN AYNI
      davranış (bkz. README), yani burada YENİ bir yanlılık eklenmiyor.
    - LSTM/online model ensemble'ı (bkz. `DecisionEngine`) burada
      DAHİL DEĞİL — yalnızca birincil (XGBoost) model + meta-label.
      Canlıda bu modeller otomatik olarak aktifse (bkz. `app.ml.model_status`)
      davranış burada tam yansıtılmaz.
    - Kademeli alım/satım (tranche), Kelly boyutlandırma, VIX rejim
      filtresi YOK — basitlik için her sinyalde TÜM equity ile tek
      giriş/tek çıkış simüle edilir (bkz. `PortfolioManager` gerçek
      canlı/paper motorunda bunlar var, ama orası ayrı bir katman).
    """
    timeframe = request.timeframe or "1h"
    ohlcv = fetch_full_history(exchange, request.symbol, timeframe, max_candles=request.candles)

    if len(ohlcv) < _MIN_CANDLES:
        raise ValueError(
            f"Yeterli geçmiş veri yok: {len(ohlcv)} mum bulundu, gösterge ısınması için en az "
            f"{_MIN_CANDLES} mum gerekli."
        )

    features = build_features(ohlcv).dropna().reset_index(drop=True)
    predictions, confidences = model.predict_batch(features)
    directions = np.array([_DIRECTION_MAP[int(p)] for p in predictions])

    cost_pct_roundtrip = (request.commission_pct + request.slippage_pct) * 2

    position: dict | None = None
    trades: list[dict] = []
    equity = request.initial_balance
    equity_curve = [equity]

    for i in range(len(features)):
        row = features.iloc[i]
        price = float(row["close"])
        ts = pd.Timestamp(int(row["timestamp"]))
        direction = str(directions[i])
        confidence = float(confidences[i])

        if position is not None:
            stop_hit = (
                request.stop_loss_pct is not None
                and (
                    (position["direction"] == "long" and price <= position["stop_loss_price"])
                    or (position["direction"] == "short" and price >= position["stop_loss_price"])
                )
            )
            opposing = (position["direction"] == "long" and direction == "short") or (
                position["direction"] == "short" and direction == "long"
            )
            should_close = stop_hit or (confidence >= request.close_confidence and (opposing or direction == "neutral"))

            if should_close:
                change_pct = (price / position["entry_price"] - 1) * 100
                gross_pct = change_pct if position["direction"] == "long" else -change_pct
                net_pct = gross_pct - cost_pct_roundtrip
                pnl_quote = position["size_quote"] * net_pct / 100
                equity += pnl_quote
                trades.append(
                    {
                        "direction": position["direction"],
                        "entry_time": position["entry_time"],
                        "exit_time": ts,
                        "entry_price": position["entry_price"],
                        "exit_price": price,
                        "pnl_pct": round(net_pct, 3),
                        "pnl_quote": round(pnl_quote, 4),
                        "equity_after": round(equity, 4),
                        "exit_reason": "stop_loss" if stop_hit else "signal",
                        "duration_candles": i - position["entry_index"],
                    }
                )
                equity_curve.append(equity)
                position = None
                continue  # aynı barda hemen yeniden pozisyon açılmaz

        if position is None and direction in ("long", "short") and confidence >= request.open_confidence:
            if meta_model is not None and request.use_meta_label:
                decision = meta_model.decide(row, confidence)
                if not decision.act:
                    continue
            stop_loss_price = None
            if request.stop_loss_pct is not None:
                stop_loss_price = (
                    price * (1 - request.stop_loss_pct / 100)
                    if direction == "long"
                    else price * (1 + request.stop_loss_pct / 100)
                )
            position = {
                "direction": direction,
                "entry_price": price,
                "entry_time": ts,
                "entry_index": i,
                "stop_loss_price": stop_loss_price,
                "size_quote": equity,
            }

    trades_closed = len(trades)
    wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    win_rate_pct = round(wins / trades_closed * 100, 2) if trades_closed else 0.0

    peak = equity_curve[0]
    max_drawdown_pct = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown_pct = max(max_drawdown_pct, (peak - value) / peak * 100)

    period_start = pd.Timestamp(int(features.iloc[0]["timestamp"]))
    period_end = pd.Timestamp(int(features.iloc[-1]["timestamp"]))
    period_days = max((period_end - period_start).total_seconds() / 86400, 1e-6)

    total_pnl_quote = equity - request.initial_balance
    total_pnl_pct = total_pnl_quote / request.initial_balance * 100
    daily_pnl_quote = total_pnl_quote / period_days
    daily_pnl_pct = total_pnl_pct / period_days
    monthly_pnl_quote = daily_pnl_quote * 30
    monthly_pnl_pct = daily_pnl_pct * 30

    warnings: list[str] = [
        "Makro/order-book/taker-flow özellikleri bu backtest'te hesaplanmaz (yalnızca anlık değerleri var, "
        "geçmişi yok) — eğitim setindeki eski barlarla AYNI şekilde nötr (0.0) kabul edilir.",
        "LSTM/online model ensemble'ı bu backtest'e DAHİL DEĞİL — yalnızca birincil (XGBoost) model"
        + (" + meta-label filtresi" if meta_model is not None and request.use_meta_label else "") + " kullanılır.",
        "Kademeli alım/satım, Kelly boyutlandırma ve VIX rejim filtresi burada simüle edilmez — her sinyalde "
        "tüm equity ile tek giriş/tek çıkış varsayılır.",
    ]
    if trades_closed < 10:
        warnings.append(f"Yalnızca {trades_closed} işlem kapandı — istatistiksel güvenilirlik düşük.")

    run_row = {
        "symbol": request.symbol,
        "timeframe": timeframe,
        "candles_used": len(ohlcv),
        "period_start": period_start.to_pydatetime(),
        "period_end": period_end.to_pydatetime(),
        "initial_balance": request.initial_balance,
        "final_equity": round(equity, 4),
        "trades_closed": trades_closed,
        "win_rate_pct": win_rate_pct,
        "total_pnl_quote": round(total_pnl_quote, 4),
        "total_pnl_pct": round(total_pnl_pct, 3),
        "daily_pnl_quote": round(daily_pnl_quote, 4),
        "daily_pnl_pct": round(daily_pnl_pct, 4),
        "monthly_pnl_quote": round(monthly_pnl_quote, 4),
        "monthly_pnl_pct": round(monthly_pnl_pct, 3),
        "max_drawdown_pct": round(max_drawdown_pct, 3),
    }
    trade_rows = [
        {
            "direction": t["direction"],
            "entry_time": t["entry_time"].to_pydatetime(),
            "exit_time": t["exit_time"].to_pydatetime(),
            "entry_price": t["entry_price"],
            "exit_price": t["exit_price"],
            "pnl_pct": t["pnl_pct"],
            "pnl_quote": t["pnl_quote"],
            "equity_after": t["equity_after"],
            "exit_reason": t["exit_reason"],
            "duration_candles": t["duration_candles"],
        }
        for t in trades
    ]

    # Grafana'nın candlestick paneli için kullanılan geçmişi `ohlcv_raw`'a
    # backfill eder (DB kapalıysa no-op). Kalıcılık bir yan etkidir; backtest
    # sonucu DB olmadan da hesaplanıp dönebilir.
    db.save_ohlcv_bulk(request.symbol, timeframe, ohlcv)
    run_id = db.save_backtest_run(run_row, trade_rows)

    return SystemBacktestReport(
        id=run_id,
        created_at=None,
        symbol=request.symbol,
        timeframe=timeframe,
        candles_used=len(ohlcv),
        period_start=period_start.isoformat(),
        period_end=period_end.isoformat(),
        initial_balance=request.initial_balance,
        final_equity=round(equity, 4),
        trades_closed=trades_closed,
        win_rate_pct=win_rate_pct,
        total_pnl_quote=round(total_pnl_quote, 4),
        total_pnl_pct=round(total_pnl_pct, 3),
        daily_pnl_quote=round(daily_pnl_quote, 4),
        daily_pnl_pct=round(daily_pnl_pct, 4),
        monthly_pnl_quote=round(monthly_pnl_quote, 4),
        monthly_pnl_pct=round(monthly_pnl_pct, 3),
        max_drawdown_pct=round(max_drawdown_pct, 3),
        trades=[
            {
                "direction": t["direction"],
                "entry_time": t["entry_time"].isoformat(),
                "exit_time": t["exit_time"].isoformat(),
                "entry_price": t["entry_price"],
                "exit_price": t["exit_price"],
                "pnl_pct": t["pnl_pct"],
                "pnl_quote": t["pnl_quote"],
                "equity_after": t["equity_after"],
                "exit_reason": t["exit_reason"],
                "duration_candles": t["duration_candles"],
            }
            for t in trades
        ],
        warnings=warnings,
    )
