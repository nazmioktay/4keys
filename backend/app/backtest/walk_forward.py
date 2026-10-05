"""Walk-forward sistem backtest'i (plan 2.6).

Tek holdout penceresinde ölçmek yerine zaman ekseni ardışık test
katmanlarına bölünür; her katmanda birincil model (ve istenirse meta-label)
YALNIZCA o katmandan önceki verilerle sıfırdan eğitilir, sonra
`run_system_backtest` aynı yürütme kurallarıyla yalnızca o katmanı oynatır.
Böylece hiçbir sonuç, modelin eğitimde gördüğü bir dönemden gelmez ve tek
bir şanslı/şanssız pencereye bağlı kalmaz.

Serinin son `final_test_frac` dilimi hiçbir katmana verilmez: parametre
taramaları (sweep'ler, haftalık optimizasyon) bu raporu kullanır, ayrılan son
dilim yalnızca nihai doğrulama içindir (`include_final_test=True`).

Katman modelleri yalnızca veriye ve etiketlemeye bağlıdır; güven eşiği,
boyutlandırma gibi yürütme parametrelerine bağlı değildir. Bu yüzden
`prepare_walk_forward` modelleri bir kez eğitir, `evaluate_walk_forward`
aynı modellerle çok sayıda parametre adayını ucuzca değerlendirir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.exchanges.base import Exchange
from app.exchanges.cache import fetch_ohlcv_cached, timeframe_minutes
from app.ml.dataset import build_symbol_frame
from app.ml.features import ALL_FEATURE_COLUMNS
from app.ml.meta_label import MetaLabelModel, build_meta_dataset_out_of_fold
from app.ml.model import SignalModel

from .metrics import monte_carlo_bootstrap
from .schemas import MonteCarloReport, SystemBacktestRequest
from .system_runner import run_system_backtest

logger = logging.getLogger(__name__)

# Calmar benzeri skorda paydanın alt sınırı (%): neredeyse sıfır drawdown'lu
# birkaç işlemlik bir sonucun skoru sonsuza kaçmasın.
_MIN_DRAWDOWN_FOR_SCORE = 1.0


class WalkForwardFold(BaseModel):
    fold: int
    train_rows: int
    test_start: str
    test_end: str
    trades_closed: int
    win_rate_pct: float
    total_pnl_pct: float
    max_drawdown_pct: float
    error: str | None = None


class WalkForwardSystemReport(BaseModel):
    symbol: str
    timeframe: str
    folds: list[WalkForwardFold]
    trades_closed: int
    win_rate_pct: float
    # Katmanların equity getirileri bileşik olarak çarpılır.
    total_pnl_pct: float
    # Katmanlardaki en kötü maks. drawdown.
    max_drawdown_pct: float
    # PnL / maks. drawdown (Calmar benzeri) — sweep'ler ve optimizasyon bunu sıralar.
    score: float
    profitable_folds: int
    final_test_reserved_from: str | None
    includes_final_test: bool
    monte_carlo: MonteCarloReport | None = None
    warnings: list[str] = Field(default_factory=list)


@dataclass
class PreparedFold:
    fold: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_rows: int
    model: SignalModel | None = None
    meta_model: MetaLabelModel | None = None
    error: str | None = None


@dataclass
class PreparedWalkForward:
    symbol: str
    timeframe: str
    ohlcv: pd.DataFrame
    folds: list[PreparedFold]
    use_meta_label: bool
    final_test_reserved_from: str | None
    includes_final_test: bool


def score_result(total_pnl_pct: float, max_drawdown_pct: float) -> float:
    return total_pnl_pct / max(max_drawdown_pct, _MIN_DRAWDOWN_FOR_SCORE)


def _fold_boundaries(n_bars: int, n_folds: int, min_train_frac: float, final_test_frac: float, include_final_test: bool) -> list[tuple[int, int]]:
    usable_end = n_bars if include_final_test else int(n_bars * (1 - final_test_frac))
    first_test = int(n_bars * min_train_frac)
    if usable_end - first_test < n_folds * 30:
        raise ValueError(
            f"Walk-forward için yeterli bar yok: test bölgesi {usable_end - first_test} bar, "
            f"{n_folds} katman için en az {n_folds * 30} gerekli."
        )
    edges = np.linspace(first_test, usable_end, n_folds + 1).astype(int)
    return [(int(edges[k]), int(edges[k + 1])) for k in range(n_folds)]


def _naive_utc(values: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(values)
    return stamps.dt.tz_convert(None) if stamps.dt.tz is not None else stamps


def prepare_walk_forward(
    exchange: Exchange,
    request: SystemBacktestRequest,
    primary_template: SignalModel | None = None,
    n_folds: int = 4,
    min_train_frac: float = 0.4,
    final_test_frac: float = 0.1,
    include_final_test: bool = False,
    use_meta_label: bool = True,
    labeling: dict | None = None,
    embargo_bars: int = 24,
    ohlcv: pd.DataFrame | None = None,
    frame: pd.DataFrame | None = None,
) -> PreparedWalkForward:
    """Veriyi çeker, özellik/etiket çerçevesini kurar ve her katman için
    birincil modeli (+ istenirse out-of-fold meta-label'ı) YALNIZCA o
    katmandan önceki satırlarla eğitir.

    `primary_template`: katman modellerinin hiperparametre kaynağı (genelde
    üretimdeki model); verilmezse varsayılan `SignalModel`.
    `labeling`: eğitim etiketlemesi — varsayılan ensemble etiketlemesi
    (`app.ml.train._ENSEMBLE_LABELING`).
    `ohlcv`/`frame`: hazır veri/özellik çerçevesi (testler, tekrar kullanım).
    """
    from app.ml.train import _ENSEMBLE_LABELING  # local import: train -> backtest döngüsünü önler

    labeling = dict(labeling or _ENSEMBLE_LABELING)
    timeframe = request.timeframe or "1h"
    bar = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    horizon = int(labeling.get("horizon", 8))

    if ohlcv is None:
        ohlcv = fetch_ohlcv_cached(exchange, request.symbol, timeframe, request.candles)
    if frame is None:
        frame = build_symbol_frame(
            exchange,
            request.symbol,
            timeframe,
            ohlcv,
            horizon,
            labeling.get("threshold_pct", 1.0),
            labeling.get("labeling_method", "atr_triple_barrier"),
            labeling.get("take_profit_pct", 1.0),
            labeling.get("stop_loss_pct", 1.0),
            tie_neutral=bool(labeling.get("tie_neutral", False)),
            label_cost_pct=float(labeling.get("label_cost_pct", 0.0)),
        )

    stamps = _naive_utc(ohlcv["timestamp"])
    frame_stamps = _naive_utc(frame["bar_timestamp"])
    boundaries = _fold_boundaries(len(ohlcv), n_folds, min_train_frac, final_test_frac, include_final_test)

    def _factory() -> SignalModel:
        if primary_template is None:
            return SignalModel()
        return SignalModel(
            algorithm=primary_template.algorithm,
            calibrate=primary_template._calibrate,
            calibration_method=primary_template._calibration_method,
            xgb_params=primary_template.xgb_params,
        )

    folds: list[PreparedFold] = []
    for k, (start_idx, end_idx) in enumerate(boundaries):
        test_start = stamps.iloc[start_idx]
        test_end = stamps.iloc[end_idx] if end_idx < len(stamps) else stamps.iloc[-1] + bar
        # Etiket `horizon` bar ileriye bakar: test başlangıcına bu kadar (+ embargo)
        # yakın eğitim satırlarının etiketi test penceresinden bilgi taşır, atılır.
        train_cutoff = test_start - bar * (horizon + embargo_bars)
        train_rows = frame[frame_stamps < train_cutoff]
        prepared = PreparedFold(fold=k, test_start=test_start, test_end=test_end, train_rows=len(train_rows))
        if len(train_rows) < 200 or train_rows["label"].nunique() < 2:
            prepared.error = f"katman eğitimi için yetersiz veri ({len(train_rows)} satır)"
            folds.append(prepared)
            continue
        X_train = train_rows[ALL_FEATURE_COLUMNS].reset_index(drop=True)
        y_train = train_rows["label"].reset_index(drop=True)
        prepared.model = _factory()
        prepared.model.fit(X_train, y_train)
        if use_meta_label:
            time_frac = pd.Series(np.linspace(0.0, 1.0, len(X_train)))
            meta_X, meta_y = build_meta_dataset_out_of_fold(X_train, y_train, time_frac, _factory)
            if len(meta_y) >= 30 and meta_y.nunique() == 2:
                prepared.meta_model = MetaLabelModel()
                prepared.meta_model.fit(meta_X, meta_y)
        folds.append(prepared)

    reserved_from = None
    if not include_final_test:
        reserved_from = stamps.iloc[int(len(ohlcv) * (1 - final_test_frac))].isoformat()
    return PreparedWalkForward(
        symbol=request.symbol,
        timeframe=timeframe,
        ohlcv=ohlcv,
        folds=folds,
        use_meta_label=use_meta_label,
        final_test_reserved_from=reserved_from,
        includes_final_test=include_final_test,
    )


def evaluate_walk_forward(exchange: Exchange, prepared: PreparedWalkForward, request: SystemBacktestRequest) -> WalkForwardSystemReport:
    """Hazırlanmış katman modelleriyle `request`'in yürütme/boyutlandırma
    ayarlarını her test katmanında oynatır ve sonuçları birleştirir."""
    if request.use_dynamic_exit:
        raise ValueError("Walk-forward backtest use_dynamic_exit ile desteklenmiyor (dinamik çıkış modeli katman başına eğitilmiyor).")
    fold_request = request.model_copy(
        update={"use_ensemble": False, "restrict_to_holdout": False, "use_meta_label": prepared.use_meta_label}
    )

    folds: list[WalkForwardFold] = []
    equity_returns: list[float] = []
    growth = 1.0
    worst_drawdown = 0.0
    total_trades = 0
    total_wins = 0
    for p in prepared.folds:
        fold = WalkForwardFold(
            fold=p.fold,
            train_rows=p.train_rows,
            test_start=p.test_start.isoformat(),
            test_end=p.test_end.isoformat(),
            trades_closed=0,
            win_rate_pct=0.0,
            total_pnl_pct=0.0,
            max_drawdown_pct=0.0,
            error=p.error,
        )
        if p.model is not None:
            try:
                report = run_system_backtest(
                    exchange,
                    p.model,
                    p.meta_model,
                    fold_request,
                    persist=False,
                    ohlcv=prepared.ohlcv,
                    eval_window=(p.test_start, p.test_end),
                )
                fold.trades_closed = report.trades_closed
                fold.win_rate_pct = report.win_rate_pct
                fold.total_pnl_pct = report.total_pnl_pct
                fold.max_drawdown_pct = report.max_drawdown_pct
                growth *= 1 + report.total_pnl_pct / 100
                worst_drawdown = max(worst_drawdown, report.max_drawdown_pct)
                total_trades += report.trades_closed
                total_wins += sum(1 for t in report.trades if t.pnl_pct > 0)
                for t in report.trades:
                    equity_before = t.equity_after - t.pnl_quote
                    if equity_before > 0:
                        equity_returns.append(t.pnl_quote / equity_before * 100)
            except ValueError as exc:
                fold.error = str(exc)
        if fold.error:
            logger.warning("walk-forward katman %d atlandı: %s", fold.fold, fold.error)
        folds.append(fold)

    total_pnl_pct = (growth - 1) * 100
    warnings = [
        "Yalnızca birincil model" + (" + katman başına yeniden eğitilen meta-label" if prepared.use_meta_label else "")
        + " kullanıldı; LSTM/online ensemble walk-forward'da yer almaz.",
        "Her katman Kelly geçmişi olmadan başlar (ilk işlemler fixed_risk ile boyutlanır).",
    ]
    if any(f.error for f in folds):
        warnings.append(f"{sum(1 for f in folds if f.error)} katman atlandı (yetersiz veri).")

    return WalkForwardSystemReport(
        symbol=prepared.symbol,
        timeframe=prepared.timeframe,
        folds=folds,
        trades_closed=total_trades,
        win_rate_pct=round(total_wins / total_trades * 100, 2) if total_trades else 0.0,
        total_pnl_pct=round(total_pnl_pct, 4),
        max_drawdown_pct=round(worst_drawdown, 4),
        score=round(score_result(total_pnl_pct, worst_drawdown), 4),
        profitable_folds=sum(1 for f in folds if f.error is None and f.total_pnl_pct > 0),
        final_test_reserved_from=prepared.final_test_reserved_from,
        includes_final_test=prepared.includes_final_test,
        monte_carlo=monte_carlo_bootstrap(equity_returns, seed=42),
        warnings=warnings,
    )


def run_walk_forward_system_backtest(
    exchange: Exchange,
    request: SystemBacktestRequest,
    primary_template: SignalModel | None = None,
    n_folds: int = 4,
    min_train_frac: float = 0.4,
    final_test_frac: float = 0.1,
    include_final_test: bool = False,
    use_meta_label: bool = True,
    labeling: dict | None = None,
    embargo_bars: int = 24,
    ohlcv: pd.DataFrame | None = None,
    frame: pd.DataFrame | None = None,
) -> WalkForwardSystemReport:
    """`request`'in yürütme/boyutlandırma ayarlarıyla walk-forward sistem
    backtest'i (`prepare_walk_forward` + `evaluate_walk_forward`). Yalnızca
    birincil model (+ istenirse katman başına yeniden eğitilen meta-label)
    kullanılır; LSTM/online ensemble katman başına yeniden eğitilmediği için
    dışarıda bırakılır (aksi halde geleceği görmüş modeller karışırdı)."""
    if request.use_dynamic_exit:
        raise ValueError("Walk-forward backtest use_dynamic_exit ile desteklenmiyor (dinamik çıkış modeli katman başına eğitilmiyor).")
    prepared = prepare_walk_forward(
        exchange,
        request,
        primary_template=primary_template,
        n_folds=n_folds,
        min_train_frac=min_train_frac,
        final_test_frac=final_test_frac,
        include_final_test=include_final_test,
        use_meta_label=use_meta_label,
        labeling=labeling,
        embargo_bars=embargo_bars,
        ohlcv=ohlcv,
        frame=frame,
    )
    return evaluate_walk_forward(exchange, prepared, request)


def _is_reliable(report: WalkForwardSystemReport, min_trades: int) -> bool:
    """Öneri olarak seçilebilmek için: yeterli toplam işlem VE geçerli
    katmanların en az yarısında kâr — tek bir şanslı katmanın taşıdığı
    sonuç seçilmez."""
    valid = [f for f in report.folds if f.error is None]
    return bool(valid) and report.trades_closed >= min_trades and report.profitable_folds * 2 >= len(valid)


def run_walk_forward_optimization(
    exchange: Exchange,
    prepared: PreparedWalkForward,
    base_request: SystemBacktestRequest,
    open_confidence_values: list[float] | None = None,
    close_confidence_gap: float = 0.05,
    kelly_min_trades_values: list[int] | None = None,
    kelly_multiplier_values: list[float] | None = None,
    meta_label_threshold_values: list[float] | None = None,
    min_trades: int | None = None,
):
    """Haftalık optimizasyonun walk-forward sürümü (plan Faz 4).
    `run_periodic_optimization` ile AYNI eksenleri sırayla tarar (güven
    eşiği -> Kelly boyutlandırma -> meta-label eşiği), ama:
    - her aday walk-forward raporuyla ölçülür (tek holdout değil),
    - sıralama ölçütü PnL değil `score` (PnL / maks. drawdown),
    - aday yalnızca `_is_reliable` ise seçilebilir.
    Ayrılmış son test dilimi (`final_test_frac`) hiçbir adaya gösterilmez."""
    from .system_runner import MIN_RELIABLE_TRADES_FOR_OPTIMIZATION, OptimizationRunResult

    min_trades = MIN_RELIABLE_TRADES_FOR_OPTIMIZATION if min_trades is None else min_trades
    open_confidence_values = open_confidence_values or [0.45, 0.5, 0.55, 0.6]
    kelly_min_trades_values = kelly_min_trades_values or [20, 40]
    kelly_multiplier_values = kelly_multiplier_values or [0.5, 0.75, 1.0]
    meta_label_threshold_values = meta_label_threshold_values or [0.5, 0.55, 0.6, 0.65]

    def _best(candidates: list[tuple[SystemBacktestRequest, WalkForwardSystemReport]]):
        reliable = [c for c in candidates if _is_reliable(c[1], min_trades)]
        return max(reliable, key=lambda c: c[1].score) if reliable else None

    def _evaluate(request: SystemBacktestRequest) -> tuple[SystemBacktestRequest, WalkForwardSystemReport]:
        return request, evaluate_walk_forward(exchange, prepared, request)

    current = _evaluate(base_request)
    best = current if _is_reliable(current[1], min_trades) else None

    def _search(start: SystemBacktestRequest, variants: list[dict]) -> SystemBacktestRequest:
        nonlocal best
        winner = _best([_evaluate(start.model_copy(update=v)) for v in variants] + ([best] if best else []))
        if winner is not None:
            best = winner
            return winner[0]
        return start

    request = _search(
        base_request,
        [{"open_confidence": oc, "close_confidence": max(0.34, oc - close_confidence_gap)} for oc in open_confidence_values],
    )
    request = _search(
        request,
        [{"kelly_min_trades": m, "kelly_multiplier": k} for m in kelly_min_trades_values for k in kelly_multiplier_values],
    )
    if prepared.use_meta_label:
        request = _search(request, [{"meta_label_act_threshold": t} for t in meta_label_threshold_values])

    recommended_request, recommended_report = best if best is not None else current
    current_request, current_report = current
    return OptimizationRunResult(
        symbol=base_request.symbol,
        recommended_open_confidence=recommended_request.open_confidence,
        recommended_close_confidence=recommended_request.close_confidence,
        recommended_kelly_min_trades=recommended_request.kelly_min_trades,
        recommended_kelly_multiplier=recommended_request.kelly_multiplier,
        recommended_meta_label_act_threshold=recommended_request.meta_label_act_threshold,
        recommended_trades_closed=recommended_report.trades_closed,
        recommended_win_rate_pct=recommended_report.win_rate_pct,
        recommended_total_pnl_pct=recommended_report.total_pnl_pct,
        recommended_max_drawdown_pct=recommended_report.max_drawdown_pct,
        current_open_confidence=current_request.open_confidence,
        current_close_confidence=current_request.close_confidence,
        current_kelly_min_trades=current_request.kelly_min_trades,
        current_kelly_multiplier=current_request.kelly_multiplier,
        current_meta_label_act_threshold=current_request.meta_label_act_threshold,
        current_trades_closed=current_report.trades_closed,
        current_win_rate_pct=current_report.win_rate_pct,
        current_total_pnl_pct=current_report.total_pnl_pct,
        current_max_drawdown_pct=current_report.max_drawdown_pct,
        current_score=current_report.score,
        recommended_score=recommended_report.score,
        method="walk_forward",
    )
