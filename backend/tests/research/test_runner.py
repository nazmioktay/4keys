"""Koşucu uçtan uca (sentetik piyasa): kayıt, sayaç, smoke muafiyeti, bütçe, config doğrulama."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from research import budget, config, registry, runner
from research.data import store
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
    monkeypatch.setattr(budget, "MAX_CONFIGS_PER_QUESTION", 1)
    # runner.assert_budget varsayılan limiti import anında bağladığı için doğrudan limit geçen sürümü kullanır
    monkeypatch.setattr(runner, "assert_budget", lambda q, n, rf: budget.assert_budget(q, n, rf, limit=1))
    runner.run_experiment(_cfg(id="b1", checks={k: False for k in runner.DEFAULT_CHECKS}), **_paths(market))
    with pytest.raises(budget.BudgetExceededError, match="bütçesi dolu"):
        runner.run_experiment(_cfg(id="b2", checks={k: False for k in runner.DEFAULT_CHECKS}), **_paths(market))
    assert registry.current_trial_count(_paths(market)["registry_file"]) == 2  # ikinci koşu hiç başlamadı


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
