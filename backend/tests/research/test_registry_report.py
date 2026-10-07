import json

import numpy as np
import pandas as pd
import pytest

from research import registry, report
from research.guard import FinalTestError


def _returns(n=400, seed=0, start="2021-01-01", mu=0.001):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, 0.01, n), index=pd.date_range(start, periods=n, freq="D"), name="return")


@pytest.fixture
def paths(tmp_path):
    log = tmp_path / "deneyler.md"
    log.write_text("# Deney günlüğü\n\n**Toplam deneme: 0**\n\n| Tarih | Deney |\n|---|---|\n\n<!-- Nihai pencere açma kayıtları (yalnızca Aşama 5): \"FINAL-TEST-ACILDI <deney_id>\" -->\n", encoding="utf-8")
    return {"results_dir": tmp_path / "results", "log_path": log, "registry_file": tmp_path / "results" / "_registry.json"}


def test_smoke_registration_writes_nothing_and_does_not_count(paths):
    out = registry.register_experiment("smoke1", {"a": 1}, _returns(), {"sharpe": 1.0}, smoke=True, **paths)
    assert out["registered"] is False
    assert not paths["results_dir"].exists()
    assert registry.current_trial_count(paths["registry_file"]) == 0
    assert "Toplam deneme: 0" in paths["log_path"].read_text(encoding="utf-8")


def test_registration_writes_files_increments_counter_and_updates_log(paths):
    r = _returns()
    out = registry.register_experiment(
        "e001", {"strategy": "ema", "span": 200}, r, {"sharpe": 1.2, "max_drawdown": -0.25},
        n_variants=3, hypothesis="trend | test", decision="devam", data_hash="abc123", **paths,
    )
    d = paths["results_dir"] / "e001"
    assert {p.name for p in d.iterdir()} == {"config.yaml", "metrics.json", "daily_returns.parquet"}
    metrics = json.loads((d / "metrics.json").read_text())
    assert metrics["trials_before"] == 0 and metrics["trials_after"] == 3 and metrics["n_variants"] == 3
    assert metrics["data_snapshot_hash"] == "abc123" and metrics["git_commit"]
    assert 0.0 <= metrics["deflated_sharpe"] <= 1.0
    assert pd.read_parquet(d / "daily_returns.parquet").shape[0] == len(r)
    assert registry.current_trial_count(paths["registry_file"]) == 3
    log = paths["log_path"].read_text(encoding="utf-8")
    assert "**Toplam deneme: 3**" in log and "| e001 |" in log and "trend / test" in log  # '|' kaçışı
    assert out["trials_after"] == 3

    registry.register_experiment("e002", {"x": 1}, r, {"sharpe": 0.5, "max_drawdown": -0.1}, n_variants=2, **paths)
    assert registry.current_trial_count(paths["registry_file"]) == 5
    # DSR, artan deneme sayısıyla düşer (aynı getiri serisi)
    d2 = json.loads((paths["results_dir"] / "e002" / "metrics.json").read_text())["deflated_sharpe"]
    assert d2 <= metrics["deflated_sharpe"]


def test_registration_refuses_overwrite_and_bad_ids(paths):
    registry.register_experiment("e1", {}, _returns(), {"sharpe": 1.0, "max_drawdown": -0.1}, **paths)
    with pytest.raises(registry.ExperimentExistsError):
        registry.register_experiment("e1", {}, _returns(), {}, **paths)
    with pytest.raises(ValueError):
        registry.register_experiment("../kotu", {}, _returns(), {}, **paths)
    with pytest.raises(ValueError):
        registry.register_experiment("e9", {}, _returns(), {}, n_variants=0, **paths)
    assert registry.current_trial_count(paths["registry_file"]) == 1


def test_variants_dataframe_is_stored_for_pbo(paths):
    variants = pd.concat([_returns(seed=i).rename(f"v{i}") for i in range(4)], axis=1)
    registry.register_experiment("e5", {}, variants, {"sharpe": 1.0, "max_drawdown": -0.1}, n_variants=4, **paths)
    stored = pd.read_parquet(paths["results_dir"] / "e5" / "daily_returns.parquet")
    assert list(stored.columns) == ["v0", "v1", "v2", "v3"]


def test_render_report_writes_markdown_and_png_with_benchmark(tmp_path):
    r = _returns(seed=1)
    bench = _returns(seed=2, mu=0.0005)
    out = report.render_report(r, bench, tmp_path / "rep", "Deneme stratejisi", extra={"not": "altyapı"})
    md = (tmp_path / "rep" / "report.md").read_text(encoding="utf-8")
    png = tmp_path / "rep" / "report.png"
    assert png.exists() and png.stat().st_size > 5000
    assert "BTC al-tut" in md and "Sharpe" in md and "Takvim yılı" in md and "altyapı" in md
    assert out["metrics"]["n_days"] == len(r) and "beta_vs_btc" in out["metrics"]


def test_render_report_refuses_final_test_dates(tmp_path):
    r = _returns(n=40, start="2025-09-01")  # 09-01 .. 10-10
    with pytest.raises(FinalTestError):
        report.render_report(r, None, tmp_path / "rep", "x")


def test_git_commit_prefers_env_then_git_and_marks_uncommitted_backend_changes(tmp_path, monkeypatch):
    import shutil
    import subprocess

    from research import config as rconfig
    from research import registry as reg

    monkeypatch.setenv("GIT_COMMIT", "abc1234")
    assert reg.git_commit() == "abc1234"  # ortam değişkeni önceliklidir
    monkeypatch.delenv("GIT_COMMIT")
    if shutil.which("git") is None:
        pytest.skip("git yok")
    repo = tmp_path / "repo"
    (repo / "backend").mkdir(parents=True)
    (repo / "backend" / "x.py").write_text("a = 1\n", encoding="utf-8")
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "ilk")
    monkeypatch.setattr(rconfig, "REPO_ROOT", repo)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert reg.git_commit() == head
    (repo / "backend" / "x.py").write_text("a = 2\n", encoding="utf-8")
    assert reg.git_commit() == head + "+dirty"  # backend'de commit edilmemiş değişiklik
