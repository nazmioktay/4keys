"""Deney kaydı + GLOBAL deneme sayacı (KURALLAR.md §6).

Her çalıştırma: `research/results/<id>/` (config.yaml, metrics.json, daily_returns.parquet),
`docs/research/deneyler.md` özet satırı, `research/results/_registry.json` sayacı.
Her varyant bir denemedir; Deflated Sharpe bu sayıyla hesaplanır. `smoke=True` HİÇBİR şey kaydetmez."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from . import config, stats
from .data import store

_LOCK = threading.Lock()

MAX_CONFIGS_PER_QUESTION = 40  # KURALLAR.md §9: soru başına en fazla 40 config (ablasyonlar dahil)
MAX_SMOKE_PER_QUESTION = 10  # smoke sayılmaz AMA sınırsız deneme-yanılma kaçağı olmasın: soru başına en fazla 10 smoke koşusu


class BudgetExceededError(RuntimeError):
    pass


class RegistryCorruptError(RuntimeError):
    """`_registry.json` var ama okunamıyor: SESSİZCE boş sayıp üzerine yazmak sayaç/geçmişi silerdi."""


def normalize_question(question: str | None) -> str:
    """'q', 'q ', 'Q' tek bir bütçe sayılsın: strip + lower; yalnızca harf/rakam/_.- ."""
    q = (question or "genel").strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]+", q):
        raise ValueError(f"question yalnızca harf/rakam/_.- içerebilir: {question!r}")
    return q


class ExperimentExistsError(RuntimeError):
    pass


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.REPO_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _read_registry(path: Path) -> dict:
    if not path.exists():
        return {"total_trials": 0, "experiments": [], "smoke_runs": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # dosya VAR ama bozuk: hata ver, üzerine yazma
        raise RegistryCorruptError(f"{path} okunamadı ({exc}); elle onarın veya yedekten geri yükleyin (üzerine yazılmadı)") from exc


def _write_registry(path: Path, registry: dict) -> None:
    """Atomik yazma: geçici dosya + os.replace (yarım yazılmış/bozuk kayıt dosyası oluşmaz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(registry, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _loose_question(question) -> str:
    """ESKİ kayıtlar için TOLERANSLI normalizasyon (eski load_config question'ı doğrulamıyordu): hata fırlatmaz."""
    return (str(question).strip().lower() if question else "") or "genel"


def _configs_in(registry: dict, question: str) -> int:
    return sum(1 for e in registry["experiments"] if _loose_question(e.get("question")) == question)


def configs_used(question: str, registry_file: Path | None = None) -> int:
    return _configs_in(_read_registry(Path(registry_file or config.REGISTRY_FILE)), normalize_question(question))


def assert_budget(question: str, n_new: int = 1, registry_file: Path | None = None, limit: int | None = None) -> int:
    """Yeni `n_new` config sığıyor mu? Sığmıyorsa `BudgetExceededError`. Döner: kalan bütçe. (Erken/hızlı başarısızlık için;
    asıl zorlama `register_experiment` içinde KİLİT ALTINDADIR.)"""
    q = normalize_question(question)
    limit = MAX_CONFIGS_PER_QUESTION if limit is None else limit
    used = configs_used(q, registry_file)
    if used + n_new > limit:
        raise BudgetExceededError(
            f"'{q}' sorusu için deneme bütçesi dolu ({used}/{limit} config; {n_new} yeni istendi). "
            "Yeni deneme yapılmaz: mevcut sonuçlardan karar verin (docs/research/deneyler.md)."
        )
    return limit - used - n_new


def record_smoke_run(question: str, registry_file: Path | None = None, limit: int | None = None) -> int:
    """Smoke koşusunu SAYAR (deneme sayacına DEĞİL, ayrı `smoke_runs` tablosuna); soru başına sınırı aşarsa reddeder.
    Smoke sonuçları karar için kullanılamaz (yalnızca çerçeve kontrolü)."""
    q = normalize_question(question)
    path = Path(registry_file or config.REGISTRY_FILE)
    limit = MAX_SMOKE_PER_QUESTION if limit is None else limit
    with _LOCK:
        reg = _read_registry(path)
        runs = reg.setdefault("smoke_runs", {})
        if runs.get(q, 0) + 1 > limit:
            raise BudgetExceededError(
                f"'{q}' sorusu için smoke koşu sınırı doldu ({runs.get(q, 0)}/{limit}). Smoke çıktıları karar için kullanılamaz; "
                "gerçek (kayıtlı) deneme yapın veya soruyu değiştirmeyin."
            )
        runs[q] = runs.get(q, 0) + 1
        _write_registry(path, reg)
        return runs[q]


def is_registered(experiment_id: str, registry_file: Path | None = None, results_dir: Path | None = None) -> bool:
    reg = _read_registry(Path(registry_file or config.REGISTRY_FILE))
    return any(e["id"] == experiment_id for e in reg["experiments"]) or (Path(results_dir or config.RESULTS_DIR) / experiment_id / "metrics.json").exists()


def current_trial_count(registry_file: Path | None = None) -> int:
    return int(_read_registry(Path(registry_file or config.REGISTRY_FILE))["total_trials"])


def _update_log(log_path: Path, row: str, total_trials: int) -> None:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Deney günlüğü\n\n**Toplam deneme: 0**\n\n"
    text = re.sub(r"\*\*Toplam deneme: \d+\*\*", f"**Toplam deneme: {total_trials}**", text)
    marker = "<!-- Nihai pencere açma kayıtları"
    if marker in text:
        head, tail = text.split(marker, 1)
        text = head.rstrip("\n") + "\n" + row + "\n\n" + marker + tail
    else:
        text = text.rstrip("\n") + "\n" + row + "\n"
    log_path.write_text(text, encoding="utf-8")


def register_experiment(
    experiment_id: str,
    experiment_config: dict,
    daily_returns: pd.Series | pd.DataFrame,
    metrics: dict,
    *,
    n_variants: int = 1,
    hypothesis: str = "",
    decision: str = "",
    smoke: bool = False,
    question: str | None = None,
    results_dir: Path | None = None,
    log_path: Path | None = None,
    registry_file: Path | None = None,
    data_hash: str | None = None,
) -> dict:
    """Deneyi kaydeder ve (smoke değilse) deneme sayacını `n_variants` kadar artırır.

    `daily_returns`: Series (seçilen varyant) veya DataFrame (tüm varyantlar, PBO için). Deflated Sharpe
    güncel toplam sayıyla otomatik hesaplanıp `metrics['deflated_sharpe']`a yazılır."""
    if smoke:
        return {"experiment_id": experiment_id, "registered": False, "reason": "smoke: kayıt ve sayaç yok"}
    results_dir = Path(results_dir or config.RESULTS_DIR)
    log_path = Path(log_path or config.EXPERIMENT_LOG)
    registry_file = Path(registry_file or config.REGISTRY_FILE)
    question = normalize_question(question) if question is not None else None
    if n_variants < 1:
        raise ValueError("n_variants >= 1 olmalı")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", experiment_id):
        raise ValueError("experiment_id yalnızca harf/rakam/_.- içerebilir")

    with _LOCK:
        out_dir = results_dir / experiment_id
        registry = _read_registry(registry_file)
        if (out_dir / "metrics.json").exists() or any(e["id"] == experiment_id for e in registry["experiments"]):
            raise ExperimentExistsError(f"{experiment_id} zaten kayıtlı (kayıtlar değiştirilmez)")
        if question is not None and _configs_in(registry, question) + 1 > MAX_CONFIGS_PER_QUESTION:  # eşzamanlı koşular dahil: KİLİT ALTINDA
            raise BudgetExceededError(
                f"'{question}' sorusu için deneme bütçesi dolu ({_configs_in(registry, question)}/{MAX_CONFIGS_PER_QUESTION} config). "
                "Yeni deneme yapılmaz: mevcut sonuçlardan karar verin."
            )
        before = int(registry["total_trials"])
        after = before + n_variants

        primary = daily_returns if isinstance(daily_returns, pd.Series) else daily_returns.iloc[:, 0]
        full_metrics = dict(metrics)
        if "deflated_sharpe" not in full_metrics:
            full_metrics["deflated_sharpe"] = stats.deflated_sharpe(primary, n_trials=after)
        data_hash = data_hash or store.snapshot_hash()
        commit = git_commit()
        full_metrics.update({"trials_before": before, "trials_after": after, "n_variants": n_variants,
                             "data_snapshot_hash": data_hash, "git_commit": commit})

        out_dir.mkdir(parents=True, exist_ok=True)  # koşucu oof.parquet'i önceden yazmış olabilir
        (out_dir / "config.yaml").write_text(yaml.safe_dump(experiment_config, allow_unicode=True, sort_keys=True), encoding="utf-8")
        (out_dir / "metrics.json").write_text(json.dumps(full_metrics, indent=1, default=str), encoding="utf-8")
        frame = daily_returns.to_frame("return") if isinstance(daily_returns, pd.Series) else daily_returns
        frame.to_parquet(out_dir / "daily_returns.parquet")

        registry["total_trials"] = after
        registry["experiments"].append(
            {"id": experiment_id, "question": question, "n_variants": n_variants, "trials_before": before, "trials_after": after,
             "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": commit, "data_hash": data_hash}
        )
        _write_registry(registry_file, registry)

        sharpe = full_metrics.get("sharpe", float("nan"))
        mdd = full_metrics.get("max_drawdown", float("nan"))
        row = (
            f"| {datetime.now(timezone.utc):%Y-%m-%d %H:%M} | {experiment_id} | {hypothesis.replace('|', '/')} | {n_variants} | "
            f"{sharpe:.2f} | {abs(mdd) * 100:.1f}% | {decision.replace('|', '/')} | {data_hash} | {commit} |"
        )
        _update_log(log_path, row, after)
    return {"experiment_id": experiment_id, "registered": True, "trials_after": after, "dir": str(out_dir), "metrics": full_metrics}
