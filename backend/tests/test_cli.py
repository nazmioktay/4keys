import ast
import inspect
import json

import app.cli as cli_module
from app.cli import main
from app.ml.train import TrainAllStepResult


def test_train_all_cli_prints_json_and_returns_zero_on_success(monkeypatch, capsys):
    def fake_resolve_symbols(exchange, symbols):
        return symbols or ["BTC/USDT:USDT"]

    def fake_train_all_models(exchange, symbols):
        return [TrainAllStepResult("xgboost", True, "detail")]

    monkeypatch.setattr("app.api.routes.ml._resolve_symbols", fake_resolve_symbols)
    monkeypatch.setattr("app.ml.train.train_all_models", fake_train_all_models)
    monkeypatch.setattr(cli_module, "init_db", lambda: None)
    monkeypatch.setattr(cli_module, "get_exchange", lambda exchange_id: object())

    exit_code = main(["train-all", "--symbols", "BTC/USDT:USDT"])
    assert exit_code == 0

    output = json.loads(capsys.readouterr().out)
    assert output["symbols_used"] == 1
    assert output["steps"] == [{"step": "xgboost", "ok": True, "detail": "detail"}]


def test_train_all_cli_returns_nonzero_when_no_symbols_found(monkeypatch, capsys):
    monkeypatch.setattr("app.api.routes.ml._resolve_symbols", lambda exchange, symbols: [])
    monkeypatch.setattr(cli_module, "init_db", lambda: None)
    monkeypatch.setattr(cli_module, "get_exchange", lambda exchange_id: object())

    exit_code = main(["train-all"])
    assert exit_code == 1
    assert "sembol bulunamadı" in capsys.readouterr().out


def test_cli_never_imports_app_main_or_starts_scheduler():
    """Bu CLI'nin TEK amacı: ağır eğitimi, canlı API'nin (ve onun arka plan
    zamanlayıcısının) çalıştığı process'ten AYRI tutmak (bkz. modül
    docstring'i — üretimde 3 kez OOM ile canlı API öldü). `app.main`
    (FastAPI app + lifespan + `start_scheduler()`) buradan İTHAL EDİLİRSE
    veya zamanlayıcı doğrudan başlatılırsa bu izolasyon BOZULUR — kaynağı
    tarayıp bunu bir regresyon olarak yakalar (docstring/yorumlardaki
    AÇIKLAYICI bahisler değil, GERÇEK import/çağrıları hedefler)."""
    tree = ast.parse(inspect.getsource(cli_module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(alias.name.startswith("app.main") for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "app.main" and not (node.module or "").startswith("app.main.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "start_scheduler"
