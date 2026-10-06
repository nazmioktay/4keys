"""Koşucu uçtan uca (sentetik piyasa): kayıt, sayaç, smoke muafiyeti, bütçe, config doğrulama."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from research import budget, config, registry, runner
from research.panel import ForwardOnlyTooShortError, build_panel
from research.sources.base import Source
from research.sources.registry import register_source
from tests.research.synth import make_market, write_market


@pytest.fixture
def market(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    write_market(make_market())
    return tmp_path


def _cfg(**over):
    cfg = {
        "id": "t_vol_001", "question": "vol_tahmini", "smoke": False, "sources": ["ohlcv_core"], "universe": {"n": 10},
        "period": {"start": "2021-07-01", "end": "2023-08-31"}, "target": {"name": "next_rv", "h": 1},
        "models": ["har_rv", "ridge"], "cv": {"n_splits": 3, "embargo_days": 1, "min_train_days": 200},
        "signal": {"adapter": "vol_target", "params": {"target_vol_annual": 0.4, "cap": 1.0}}, "seed": 0,
    }
    cfg.update(over)
    return cfg


def _paths(tmp_path):
    return dict(results_dir=tmp_path / "results", log_path=tmp_path / "deneyler.md", registry_file=tmp_path / "results" / "_registry.json")


def _log(tmp_path):
    (tmp_path / "deneyler.md").write_text("# Deney günlüğü\n\n**Toplam deneme: 0**\n\n| t |\n|---|\n", encoding="utf-8")


def test_config_validation_defaults_and_embargo_rule():
    cfg = runner.load_config(_cfg())
    assert cfg["primary_model"] == "har_rv" and cfg["checks"]["reproducibility"] is True and cfg["models"][1]["id"] == "ridge"
    with pytest.raises(runner.ConfigError, match="embargo_days"):
        runner.load_config(_cfg(target={"name": "next_rv", "h": 5}, cv={"embargo_days": 2}))
    with pytest.raises(runner.ConfigError, match="zorunlu"):
        runner.load_config({"id": "x"})
    with pytest.raises(runner.ConfigError, match="benzersiz"):
        runner.load_config(_cfg(models=["ridge", "ridge"]))
    with pytest.raises(runner.ConfigError, match="primary_model"):
        runner.load_config(_cfg(primary_model="yok"))


def test_end_to_end_run_registers_counts_trials_and_writes_artifacts(market):
    _log(market)
    r = runner.run_experiment(_cfg(), **_paths(market))
    assert r.valid, r.checks.failures
    assert r.registered and (r.out_dir / "oof.parquet").exists() and (r.out_dir / "report.md").exists() and (r.out_dir / "config.yaml").exists()
    oof = pd.read_parquet(r.oof_path)
    assert {"har_rv", "ridge", "y"} <= set(oof.columns)
    # volatilite kümelenmesi olan sentetik piyasada HAR gerçek beceri gösterir (IC belirgin pozitif), ama 1'e yakın değildir
    ic = r.metrics["variants"]["har_rv"]["pred"]
    assert 0.2 < ic["ic_mean"] < 0.95 and ic["ic_t"] > 5 and ic["qlike"] > 0
    assert registry.current_trial_count(_paths(market)["registry_file"]) == 2  # 2 model = 2 varyant = 2 deneme
    metrics = json.loads((r.out_dir / "metrics.json").read_text())
    assert metrics["trials_after"] == 2 and 0 <= metrics["deflated_sharpe"] <= 1 and metrics["valid"] is True
    assert "t_vol_001" in (market / "deneyler.md").read_text(encoding="utf-8")
    for name in ("shuffled_target", "availability", "extra_lag", "reproducibility"):
        assert r.checks.results[name]["passed"]


def test_smoke_run_never_counts_or_logs(market):
    _log(market)
    r = runner.run_experiment(_cfg(smoke=True, id="t_smoke"), **_paths(market))
    assert r.smoke and not r.registered and r.out_dir.parts[-2] == "_smoke"
    assert (r.out_dir / "oof.parquet").exists() and (r.out_dir / "report.png").exists()
    assert registry.current_trial_count(_paths(market)["registry_file"]) == 0
    assert "t_smoke" not in (market / "deneyler.md").read_text(encoding="utf-8")


def test_budget_blocks_new_experiments_when_the_question_is_full(market, monkeypatch):
    _log(market)
    monkeypatch.setattr(registry, "MAX_CONFIGS_PER_QUESTION", 1)
    no_checks = {k: False for k in runner.DEFAULT_CHECKS}
    runner.run_experiment(_cfg(id="b1", checks=no_checks), **_paths(market))
    with pytest.raises(budget.BudgetExceededError, match="bütçesi dolu"):
        runner.run_experiment(_cfg(id="b2", checks=no_checks), **_paths(market))
    assert registry.current_trial_count(_paths(market)["registry_file"]) == 2  # ikinci koşu hiç başlamadı


def test_smoke_runs_are_tallied_per_question_and_capped(market, monkeypatch):
    _log(market)
    monkeypatch.setattr(registry, "MAX_SMOKE_PER_QUESTION", 2)
    no_checks = {k: False for k in runner.DEFAULT_CHECKS}
    paths = _paths(market)
    for i in range(2):
        r = runner.run_experiment(_cfg(smoke=True, id=f"s{i}", checks=no_checks), **paths)
        assert r.metrics["smoke"] is True
    with pytest.raises(budget.BudgetExceededError, match="smoke koşu sınırı"):
        runner.run_experiment(_cfg(smoke=True, id="s3", checks=no_checks), **paths)
    data = json.loads(paths["registry_file"].read_text())
    assert data["smoke_runs"] == {"vol_tahmini": 2} and data["total_trials"] == 0  # deneme sayacı ASLA artmadı
    assert "KULLANILAMAZ" in (market / "results" / "_smoke" / "s0" / "report.md").read_text(encoding="utf-8")


def test_question_text_is_normalized_so_it_cannot_be_used_to_dodge_the_budget(market, monkeypatch):
    assert runner.load_config(_cfg(question="  Vol_Tahmini "))["question"] == "vol_tahmini"
    with pytest.raises(runner.ConfigError, match="question"):
        runner.load_config(_cfg(question="boşluk var!"))
    _log(market)
    monkeypatch.setattr(registry, "MAX_CONFIGS_PER_QUESTION", 1)
    no_checks = {k: False for k in runner.DEFAULT_CHECKS}
    runner.run_experiment(_cfg(id="n1", question="Vol_Tahmini", checks=no_checks), **_paths(market))
    with pytest.raises(budget.BudgetExceededError):  # "vol_tahmini " yazarak yeni bütçe AÇILAMAZ
        runner.run_experiment(_cfg(id="n2", question="vol_tahmini ", checks=no_checks), **_paths(market))


def test_budget_is_enforced_under_the_registry_lock_for_concurrent_runs(tmp_path, monkeypatch):
    import threading

    monkeypatch.setattr(registry, "MAX_CONFIGS_PER_QUESTION", 3)
    paths = dict(results_dir=tmp_path / "res", log_path=tmp_path / "deneyler.md", registry_file=tmp_path / "res" / "_registry.json")
    (tmp_path / "deneyler.md").write_text("# g\n\n**Toplam deneme: 0**\n\n| t |\n|---|\n", encoding="utf-8")
    returns = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 200), index=pd.date_range("2021-01-01", periods=200))
    ok, blocked = [], []

    def go(i):
        try:
            registry.register_experiment(f"c{i}", {}, returns, {"sharpe": 1.0, "max_drawdown": -0.1}, question="eszamanli", **paths)
            ok.append(i)
        except budget.BudgetExceededError:
            blocked.append(i)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(ok) == 3 and len(blocked) == 7  # 10 eşzamanlı koşudan tam 3'ü sığar (40'ı aşma yarışı yok)
    assert registry.current_trial_count(paths["registry_file"]) == 3


def test_run_rejects_final_test_window_and_forward_only_sources_without_history(market):
    with pytest.raises(Exception, match="nihai test"):
        runner.run_experiment(_cfg(period={"start": "2021-07-01", "end": "2025-10-15"}), **_paths(market))

    @register_source
    class FwdToy(Source):
        name = "fwd_toy"
        forward_only = True

        def accumulated_days(self):
            return 100.0

        def fetch(self, start, end, symbols=None): ...
        def to_panel(self, universe, dates): ...

    try:
        with pytest.raises(ForwardOnlyTooShortError, match="forward_only"):
            build_panel(runner.load_config(_cfg(sources=["fwd_toy"])))
    finally:
        from research.sources.registry import SOURCE_REGISTRY

        SOURCE_REGISTRY.pop("fwd_toy", None)


def test_example_config_is_valid_and_smoke(tmp_path):
    cfg = runner.load_config(config.BACKEND_ROOT / "research" / "configs" / "ornek_vol.yaml")
    assert cfg["smoke"] is True and cfg["target"] == {"name": "next_rv", "h": 1} and [m["name"] for m in cfg["models"]] == ["har_rv", "lightgbm_reg"]
    assert yaml.safe_load((config.BACKEND_ROOT / "research" / "configs" / "ornek_vol.yaml").read_text(encoding="utf-8"))["period"]["end"] is not None


def test_forward_only_source_without_overlap_with_the_research_period_fails_loudly(market, monkeypatch):
    """Çelişki (denetim bulgusu): forward_only veri nihai pencereden SONRA birikir; period.end < 2025-10-01 ile hiç kesişmez.
    Sessizce boş/NaN özellik üretmek yerine YÜKSEK SESLE reddedilmeli."""
    from datetime import datetime, timezone

    from app.core.config import settings
    from app.db import repository as dbrepo
    from app.db.models import OIDetailSnapshot
    from app.db.session import init_db, reset_for_tests, session_scope
    from research.panel import ForwardOnlyNoOverlapError

    monkeypatch.setattr(settings, "database_url", "sqlite:///:memory:")
    reset_for_tests()
    init_db()
    try:
        for _ in range(2):
            dbrepo.record_oi_detail_snapshot("S00USDT", {"open_interest": 1.0, "open_interest_value": 1e6})
        with session_scope() as sess:
            rows = sess.query(OIDetailSnapshot).order_by(OIDetailSnapshot.id).all()
            rows[0].time = datetime(2025, 10, 5, tzinfo=timezone.utc)
            rows[1].time = datetime(2026, 10, 5, tzinfo=timezone.utc)  # 12 ayı aşkın birikim -> ForwardOnlyTooShort geçer
        with pytest.raises(ForwardOnlyNoOverlapError, match="nihai test penceresinden SONRA"):
            build_panel(runner.load_config(_cfg(sources=["ohlcv_core", "oi_detail"])))
    finally:
        reset_for_tests()
