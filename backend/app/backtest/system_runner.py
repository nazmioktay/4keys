import numpy as np
import pandas as pd

from app.db import repository as db
from app.engine.decision import DecisionEngine
from app.exchanges.base import Exchange
from app.ml.advanced_indicators import average_true_range
from app.ml.features import FEATURE_COLUMNS, build_features
from app.ml.lstm_model import LSTMSignalModel
from app.ml.meta_label import MetaLabelModel
from app.ml.model import Prediction, SignalModel
from app.ml.online_model import OnlineSignalModel

from app.exchanges.cache import fetch_ohlcv_cached

from .schemas import SystemBacktestReport, SystemBacktestRequest

_DIRECTION_MAP = {1: "long", -1: "short", 0: "neutral"}

# Göstergelerin (Ichimoku ~52, Hurst penceresi 100 vb.) ısınması için
# gereken minimum bar sayısı — bu kadar veri olmadan anlamlı bir backtest
# çalıştırılamaz (bkz. app.backtest.runner._discover_sufficient_history
# aynı `len(ohlcv) < 30` kontrolünün, burada göstergelerin ısınma
# gereksinimine göre büyütülmüş hali).
_MIN_CANDLES = 250

def _fmt_mult(mult: float | None) -> str:
    return f"{mult}xATR" if mult is not None else "kapalı"


_SIZE_EXPLANATION = (
    "Kademeli alım/satım, Kelly boyutlandırma ve VIX rejim filtresi burada simüle edilmez — her işlemde "
    "o anki equity'nin TAMAMI ile tek giriş/tek çıkış yapılır (gerçek canlı/paper motorunda "
    "PortfolioManager farklı, riske göre kademeli boyutlandırma uygular)."
)


def _lstm_predictions_for_series(
    lstm_model: LSTMSignalModel,
    raw_features: pd.DataFrame,
    valid_positions: np.ndarray,
) -> tuple[list[str | None], list[float | None]]:
    """`raw_features` (henüz `dropna` uygulanmamış, `ohlcv` ile AYNI index'e
    sahip) üzerinde `sliding_window_view` ile vektörize LSTM tahmini üretir
    (bkz. `app.ml.sequence_dataset` — aynı desen). Makro/order-book gibi
    yalnızca ANLIK değeri olan kolonlar (geçmiş zaman serisi yok) 0.0
    (nötr) ile doldurulur — `app.engine.decision._predict`'in canlıda
    yaptığıyla AYNI basitleştirme, burada yeni bir yanlılık eklenmez."""
    columns = lstm_model.feature_columns or FEATURE_COLUMNS
    seq_len = lstm_model.seq_len

    frame = pd.DataFrame(index=raw_features.index)
    for col in columns:
        frame[col] = raw_features[col] if col in raw_features.columns else 0.0
    arr = frame.to_numpy(dtype="float32")
    n = len(arr)

    directions: list[str | None] = [None] * len(valid_positions)
    confidences: list[float | None] = [None] * len(valid_positions)
    if n < seq_len:
        return directions, confidences

    windows = np.lib.stride_tricks.sliding_window_view(arr, seq_len, axis=0)
    windows = np.moveaxis(windows, -1, 1)  # (n - seq_len + 1, seq_len, n_özellik)
    end_position_to_window_idx = {int(pos): k for k, pos in enumerate(range(seq_len - 1, n))}

    # DİKKAT: `valid_positions` (XGBoost özelliklerinin `dropna` sonrası kalan
    # orijinal konumları) ısınma bölgesinin SONRASI olsa da, birkaç gösterge
    # formülü (ör. `linreg_zscore`, `nwe_position`, `bb_percent_b`,
    # `di_diff_norm`, `ichimoku_cloud_position`, mum fitil oranları) bir
    # rolling payda/aralık tam SIFIR olduğunda (ör. ADX'te plus_di+minus_di=0,
    # ya da bir doji mumda high=low) ISINMA SONRASI da ARA SIRA NaN üretebilir
    # — bu durumda `raw_features.dropna()` yalnızca O satırı düşürür,
    # `valid_positions`'ta ARDIŞIK OLMAYAN bir boşluk bırakır. Bu satır,
    # PENCERENİN İÇİNDE (bitiş noktası değil) yer alsa bile LSTM'in
    # rekürrent ileri geçişini baştan sona NaN'a bulaştırır (softmax NaN
    # olur, argmax NaN'larda deterministik olarak ilk sınıfa düşer —
    # yani yön rastgele/anlamsız görünür ama güven NaN olur). Bu yüzden
    # her pencere gerçekten sonlu mu diye AÇIKÇA kontrol edilir; NaN içeren
    # pencereler (nadir) LSTM'siz (None) bırakılır.
    usable_positions = [p for p in valid_positions if int(p) in end_position_to_window_idx]
    if not usable_positions:
        return directions, confidences

    idxs = np.array([end_position_to_window_idx[int(p)] for p in usable_positions])
    candidate_windows = windows[idxs]
    finite_mask = np.isfinite(candidate_windows).all(axis=(1, 2))
    if not finite_mask.any():
        return directions, confidences

    batch_preds, batch_confs = lstm_model.predict_batch(candidate_windows[finite_mask])

    pos_to_row = {int(p): i for i, p in enumerate(valid_positions)}
    finite_positions = [p for p, keep in zip(usable_positions, finite_mask) if keep]
    for orig_pos, pred, conf in zip(finite_positions, batch_preds, batch_confs):
        row_idx = pos_to_row[int(orig_pos)]
        directions[row_idx] = _DIRECTION_MAP[int(pred)]
        confidences[row_idx] = float(conf)
    return directions, confidences


def run_system_backtest(
    exchange: Exchange,
    model: SignalModel,
    meta_model: MetaLabelModel | None,
    request: SystemBacktestRequest,
    lstm_model: LSTMSignalModel | None = None,
    online_model: OnlineSignalModel | None = None,
) -> SystemBacktestReport:
    """Canlı karar motorunun kullandığı AYNI modelleri (XGBoost birincil +
    varsa meta-label filtresi + varsa LSTM/online ensemble) `request.symbol`
    için gerçek geçmiş mumlar üzerinde bar-bar tekrar oynatır.

    Bilinen basitleştirmeler (dürüstçe belgelenir, `warnings` alanına da
    yansır):
    - Makro/order-book/taker-flow özellikleri (bkz. `ALL_FEATURE_COLUMNS`)
      burada hesaplanmaz — bunlar yalnızca "ANLIK" değerlerdir, geçmiş
      barlar için tarihsel bir zaman serisi YOKTUR. Eksik kolonlar 0.0
      (nötr) ile doldurulur — eğitim setindeki makro geçmişi kısa/yok olan
      ESKİ barlara UYGULANAN AYNI davranış (bkz. README), yani burada
      YENİ bir yanlılık eklenmiyor.
    - Kademeli alım/satım (tranche) ve Kelly boyutlandırma YOK — her
      sinyalde TÜM equity ile tek giriş/tek çıkış simüle edilir (bkz.
      `PortfolioManager` gerçek canlı/paper motorunda bunlar var, ama
      orası ayrı bir katman) — bkz. her işlemin `size_explanation` alanı.
    - Risk yönetimi: sabit yüzdelik YERİNE ATR (Average True Range)
      tabanlı stop-loss/kâr-alma/trailing-stop (kullanıcı isteği) —
      bkz. `SystemBacktestRequest.atr_*` alanları.
    """
    timeframe = request.timeframe or "1h"
    # `fetch_full_history` (DCA/JSON-strateji backtest'inde kullanılan)
    # BİLEREK 2017'den İLERİYE doğru sayfalar (o motorların amacı piyasa
    # döngülerinin en başından itibaren geniş bir test istemesi) — bu
    # sistem backtest'i için YANLIŞ: model GÜNCEL veriyle eğitiliyor, o
    # yüzden test de GÜNCEL (en son `request.candles` mum) olmalı, yoksa
    # (bu bug'da olduğu gibi) 2019-2020 gibi alakasız bir dönem test
    # edilir. `fetch_ohlcv_cached` (eğitimle AYNI yol) en son mumları
    # döner ve DB önbelleğinden faydalanır.
    ohlcv = fetch_ohlcv_cached(exchange, request.symbol, timeframe, request.candles)

    if len(ohlcv) < _MIN_CANDLES:
        raise ValueError(
            f"Yeterli geçmiş veri yok: {len(ohlcv)} mum bulundu, gösterge ısınması için en az "
            f"{_MIN_CANDLES} mum gerekli."
        )

    raw_features = build_features(ohlcv)
    raw_features["atr"] = average_true_range(ohlcv, length=request.atr_period)

    features = raw_features.dropna()
    valid_positions = features.index.to_numpy()
    features = features.reset_index(drop=True)

    predictions, confidences = model.predict_batch(features)
    xgb_directions = np.array([_DIRECTION_MAP[int(p)] for p in predictions])

    use_lstm = lstm_model is not None and request.use_ensemble
    use_online = online_model is not None and request.use_ensemble
    lstm_directions: list[str | None]
    lstm_confidences: list[float | None]
    if use_lstm:
        lstm_directions, lstm_confidences = _lstm_predictions_for_series(lstm_model, raw_features, valid_positions)
    else:
        lstm_directions = [None] * len(features)
        lstm_confidences = [None] * len(features)

    cost_pct_roundtrip = (request.commission_pct + request.slippage_pct) * 2

    position: dict | None = None
    trades: list[dict] = []
    equity = request.initial_balance
    equity_curve = [equity]

    for i in range(len(features)):
        row = features.iloc[i]
        price = float(row["close"])
        ts = pd.Timestamp(int(row["timestamp"]))
        atr_now = float(row["atr"])

        xgb_pred = Prediction(direction=str(xgb_directions[i]), confidence=float(confidences[i]))
        decision_parts = [f"XGBoost={xgb_pred.direction}({xgb_pred.confidence:.2f})"]
        combined = xgb_pred

        lstm_dir = lstm_directions[i]
        lstm_conf = lstm_confidences[i]
        if lstm_dir is not None:
            lstm_pred = Prediction(direction=lstm_dir, confidence=float(lstm_conf))
            combined = DecisionEngine._combine_predictions(combined, lstm_pred)
            decision_parts.append(f"LSTM={lstm_pred.direction}({lstm_pred.confidence:.2f})")

        online_dir: str | None = None
        online_conf: float | None = None
        if use_online:
            online_pred = online_model.predict(row)
            online_dir, online_conf = online_pred.direction, online_pred.confidence
            combined = DecisionEngine._combine_predictions(combined, online_pred)
            decision_parts.append(f"Online={online_pred.direction}({online_pred.confidence:.2f})")

        decision_reason = " + ".join(decision_parts) + f" -> karar={combined.direction}({combined.confidence:.2f})"
        direction = combined.direction
        confidence = combined.confidence

        if position is not None:
            if position["direction"] == "long":
                position["best_price"] = max(position["best_price"], price)
                if request.atr_trailing_mult is not None:
                    candidate = position["best_price"] - request.atr_trailing_mult * atr_now
                    position["trailing_stop_price"] = max(position["trailing_stop_price"], candidate)
                stop_hit = request.atr_stop_loss_mult is not None and price <= position["trailing_stop_price"]
                take_profit_hit = request.atr_take_profit_mult is not None and price >= position["take_profit_price"]
            else:
                position["best_price"] = min(position["best_price"], price)
                if request.atr_trailing_mult is not None:
                    candidate = position["best_price"] + request.atr_trailing_mult * atr_now
                    position["trailing_stop_price"] = min(position["trailing_stop_price"], candidate)
                stop_hit = request.atr_stop_loss_mult is not None and price >= position["trailing_stop_price"]
                take_profit_hit = request.atr_take_profit_mult is not None and price <= position["take_profit_price"]

            opposing = (position["direction"] == "long" and direction == "short") or (
                position["direction"] == "short" and direction == "long"
            )
            signal_close = confidence >= request.close_confidence and (opposing or direction == "neutral")
            should_close = stop_hit or take_profit_hit or signal_close

            if should_close:
                if stop_hit:
                    exit_reason = (
                        "trailing_stop" if position["trailing_stop_price"] != position["initial_stop_loss_price"] else "stop_loss"
                    )
                elif take_profit_hit:
                    exit_reason = "take_profit"
                else:
                    exit_reason = "signal"

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
                        "exit_reason": exit_reason,
                        "duration_candles": i - position["entry_index"],
                        "size_quote": position["size_quote"],
                        "size_explanation": _SIZE_EXPLANATION,
                        "xgboost_direction": position["decision"]["xgboost_direction"],
                        "xgboost_confidence": position["decision"]["xgboost_confidence"],
                        "lstm_direction": position["decision"]["lstm_direction"],
                        "lstm_confidence": position["decision"]["lstm_confidence"],
                        "online_direction": position["decision"]["online_direction"],
                        "online_confidence": position["decision"]["online_confidence"],
                        "decision_reason": position["decision"]["decision_reason"],
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
            initial_stop_loss_price = None
            take_profit_price = None
            if request.atr_stop_loss_mult is not None:
                initial_stop_loss_price = (
                    price - request.atr_stop_loss_mult * atr_now
                    if direction == "long"
                    else price + request.atr_stop_loss_mult * atr_now
                )
            if request.atr_take_profit_mult is not None:
                take_profit_price = (
                    price + request.atr_take_profit_mult * atr_now
                    if direction == "long"
                    else price - request.atr_take_profit_mult * atr_now
                )
            position = {
                "direction": direction,
                "entry_price": price,
                "entry_time": ts,
                "entry_index": i,
                "initial_stop_loss_price": initial_stop_loss_price,
                "trailing_stop_price": initial_stop_loss_price,
                "take_profit_price": take_profit_price,
                "best_price": price,
                "size_quote": equity,
                "decision": {
                    "xgboost_direction": xgb_pred.direction,
                    "xgboost_confidence": round(xgb_pred.confidence, 4),
                    "lstm_direction": lstm_dir,
                    "lstm_confidence": round(lstm_conf, 4) if lstm_conf is not None else None,
                    "online_direction": online_dir,
                    "online_confidence": round(online_conf, 4) if online_conf is not None else None,
                    "decision_reason": decision_reason,
                },
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

    ensemble_note = []
    if use_lstm:
        ensemble_note.append("LSTM")
    if use_online:
        ensemble_note.append("online")
    warnings: list[str] = [
        "Makro/order-book/taker-flow özellikleri bu backtest'te hesaplanmaz (yalnızca anlık değerleri var, "
        "geçmişi yok) — eğitim setindeki eski barlarla AYNI şekilde nötr (0.0) kabul edilir.",
        (
            "Ensemble'a dahil edilen modeller: XGBoost" + (" + meta-label filtresi" if meta_model is not None and request.use_meta_label else "")
            + ("".join(f" + {name}" for name in ensemble_note))
            + "."
            if ensemble_note
            else "LSTM/online model bu backtest'e DAHİL DEĞİL (kullanılabilir/etkin değil veya use_ensemble=false) — yalnızca "
            "birincil (XGBoost) model" + (" + meta-label filtresi" if meta_model is not None and request.use_meta_label else "") + " kullanılır."
        ),
        _SIZE_EXPLANATION,
        (
            f"Risk yönetimi: ATR({request.atr_period}) tabanlı — "
            f"stop-loss={_fmt_mult(request.atr_stop_loss_mult)}, kâr-alma={_fmt_mult(request.atr_take_profit_mult)}, "
            f"trailing={_fmt_mult(request.atr_trailing_mult)}. Kâr-alma/trailing varsayılan olarak KAPALI — çıkış "
            "birincil olarak dinamik model sinyaline (close_confidence) bağlı kalır, sabit bir mesafede kesilmez."
        ),
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
            "size_quote": t["size_quote"],
            "size_explanation": t["size_explanation"],
            "xgboost_direction": t["xgboost_direction"],
            "xgboost_confidence": t["xgboost_confidence"],
            "lstm_direction": t["lstm_direction"],
            "lstm_confidence": t["lstm_confidence"],
            "online_direction": t["online_direction"],
            "online_confidence": t["online_confidence"],
            "decision_reason": t["decision_reason"],
        }
        for t in trades
    ]

    # `fetch_ohlcv_cached` kullanılan geçmişi zaten `ohlcv_raw`'a yazdı
    # (Grafana'nın candlestick paneli buradan okur) — burada ayrıca
    # yazmaya gerek yok.
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
                "size_quote": t["size_quote"],
                "size_explanation": t["size_explanation"],
                "xgboost_direction": t["xgboost_direction"],
                "xgboost_confidence": t["xgboost_confidence"],
                "lstm_direction": t["lstm_direction"],
                "lstm_confidence": t["lstm_confidence"],
                "online_direction": t["online_direction"],
                "online_confidence": t["online_confidence"],
                "decision_reason": t["decision_reason"],
            }
            for t in trades
        ],
        warnings=warnings,
    )
