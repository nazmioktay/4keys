import pytest

from app.backtest import walk_forward as wf
from app.backtest.schemas import SystemBacktestRequest
from app.backtest.system_runner import OptimizationRunResult
from app.core import live_overrides
from app.core.config import settings
from app.db.session import init_db, reset_for_tests
from app.scheduler import jobs


def _report(pnl: float, dd: float, trades: int, profitable_folds: int, folds: int = 4) -> wf.WalkForwardSystemReport:
    return wf.WalkForwardSystemReport(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        folds=[
            wf.WalkForwardFold(fold=i, train_rows=100, test_start="a", test_end="b", trades_closed=1,
                               win_rate_pct=50.0, total_pnl_pct=1.0 if i < profitable_folds else -1.0, max_drawdown_pct=dd)
            for i in range(folds)
        ],
        trades_closed=trades,
        win_rate_pct=50.0,
        total_pnl_pct=pnl,
        max_drawdown_pct=dd,
        score=wf.score_result(pnl, dd),
        profitable_folds=profitable_folds,
        final_test_reserved_from=None,
        includes_final_test=False,
    )


def test_optimization_ranks_by_score_and_skips_unreliable_candidates(monkeypatch):
    outcomes = {
        0.5: _report(pnl=4.0, dd=4.0, trades=60, profitable_folds=3),   # skor 1.0 (mevcut)
        0.45: _report(pnl=10.0, dd=10.0, trades=80, profitable_folds=3),  # en yüksek PnL ama skor 1.0
        0.55: _report(pnl=6.0, dd=2.0, trades=50, profitable_folds=3),   # skor 3.0 — seçilmeli
        0.6: _report(pnl=9.0, dd=1.0, trades=5, profitable_folds=4),     # skor 9 ama çok az işlem
    }
    monkeypatch.setattr(wf, "evaluate_walk_forward", lambda ex, prep, req: outcomes[round(req.open_confidence, 2)])
    prepared = wf.PreparedWalkForward("BTC/USDT:USDT", "1h", None, [], False, None, False)

    result = wf.run_walk_forward_optimization(
        None, prepared, SystemBacktestRequest(open_confidence=0.5),
        open_confidence_values=[0.45, 0.55, 0.6], kelly_min_trades_values=[40], kelly_multiplier_values=[1.0],
    )

    assert result.recommended_open_confidence == 0.55
    assert result.recommended_score == pytest.approx(3.0)
    assert result.current_score == pytest.approx(1.0)
    assert result.method == "walk_forward"


def test_candidate_profitable_in_minority_of_folds_is_not_recommended(monkeypatch):
    outcomes = {
        0.5: _report(pnl=2.0, dd=2.0, trades=60, profitable_folds=3),
        0.55: _report(pnl=20.0, dd=2.0, trades=60, profitable_folds=1),  # tek şanslı katman
    }
    monkeypatch.setattr(wf, "evaluate_walk_forward", lambda ex, prep, req: outcomes[round(req.open_confidence, 2)])
    prepared = wf.PreparedWalkForward("BTC/USDT:USDT", "1h", None, [], False, None, False)
    result = wf.run_walk_forward_optimization(
        None, prepared, SystemBacktestRequest(open_confidence=0.5),
        open_confidence_values=[0.55], kelly_min_trades_values=[40], kelly_multiplier_values=[1.0],
    )
    assert result.recommended_open_confidence == 0.5


def _result(current_score, recommended_score, current_pnl=1.0, recommended_pnl=1.0):
    return OptimizationRunResult(
        symbol="BTC", recommended_open_confidence=0.55, recommended_close_confidence=0.5, recommended_kelly_min_trades=40,
        recommended_kelly_multiplier=1.0, recommended_meta_label_act_threshold=0.6, recommended_trades_closed=50,
        recommended_win_rate_pct=55.0, recommended_total_pnl_pct=recommended_pnl, recommended_max_drawdown_pct=1.0,
        current_open_confidence=0.5, current_close_confidence=0.45, current_kelly_min_trades=40, current_kelly_multiplier=1.0,
        current_meta_label_act_threshold=0.6, current_trades_closed=50, current_win_rate_pct=55.0,
        current_total_pnl_pct=current_pnl, current_max_drawdown_pct=1.0,
        current_score=current_score, recommended_score=recommended_score, method="walk_forward",
    )


def test_auto_apply_gate_uses_score_when_available(monkeypatch):
    monkeypatch.setattr(settings, "ml_periodic_optimization_min_score_improvement", 0.25)
    # PnL çok daha yüksek ama skor iyileşmesi eşiğin altında -> uygulanmaz
    assert jobs._improvement_is_enough(_result(1.0, 1.1, current_pnl=1.0, recommended_pnl=50.0)) is False
    assert jobs._improvement_is_enough(_result(1.0, 1.3)) is True


@pytest.fixture
def _sqlite_db(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "sqlite:///:memory:")
    reset_for_tests()
    init_db()
    yield
    reset_for_tests()


def test_live_overrides_survive_restart(_sqlite_db, monkeypatch):
    monkeypatch.setattr(settings, "live_open_confidence", 0.57)
    monkeypatch.setattr(settings, "live_close_confidence", 0.51)
    monkeypatch.setattr(settings, "live_meta_label_act_threshold", 0.62)
    live_overrides.persist_live_overrides()

    monkeypatch.setattr(settings, "live_open_confidence", 0.5)  # restart: varsayılana döndü
    monkeypatch.setattr(settings, "live_close_confidence", 0.45)
    monkeypatch.setattr(settings, "live_meta_label_act_threshold", 0.6)
    applied = live_overrides.apply_persisted_live_overrides()

    assert applied["live_open_confidence"] == pytest.approx(0.57)
    assert settings.live_open_confidence == pytest.approx(0.57)
    assert settings.live_close_confidence == pytest.approx(0.51)
    assert settings.live_meta_label_act_threshold == pytest.approx(0.62)
