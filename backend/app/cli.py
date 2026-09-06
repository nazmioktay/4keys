"""Ağır ML işlemlerini (şimdilik: train-all), canlı API'yi çalıştıran
uzun-ömürlü `uvicorn` process'inden AYRI, tek seferlik bir process olarak
çalıştırmak için komut satırı girişi.

Neden: `POST /ml/train-all` önceden AYNI process içinde çalışıyordu — o
process canlı API'yi ve arka plan zamanlayıcısını (`app.scheduler`) da
barındırıyor. Ağır eğitim (XGBoost + meta-label + LSTM + online + regime×3,
tek çağrıda) bu process'in bellek kullanımını, kutunun fiziksel RAM'ini
(üretim sunucusunda 3.7GB, swap eklenene kadar YOK) aşacak kadar
şişirebiliyor; kernel OOM-killer'ı devreye girip canlı API'yi öldürüyor —
üretimde 3 kez gözlendi (bkz. README "OOM üretim olayı").

Bu CLI, `deploy/train-all.sh` tarafından TAMAMEN AYRI bir Docker
konteynerinde (`docker run --rm`, canlı `4keys-backend` konteynerinin
DIŞINDA) çalıştırılır — bir eğitim isteği ne kadar bellek yerse yesin,
canlı API process'i asla etkilenmez (ayrı process = ayrı cgroup; ayrıca
konteynere bir `--memory` üst sınırı da verilir).

`app.main`/FastAPI lifespan'ı KASITLI OLARAK tetiklenmez (`start_scheduler()`
çağrılmaz) — bu, tek seferlik bir komut, ikinci bir zamanlayıcı örneği
istemiyoruz. `init_db()` çağrılır (tablo oluşturma tamamen idempotent,
zamanlayıcıyla ilgisi yok).

Kullanım: `python -m app.cli train-all [--symbols BTC/USDT:USDT ...]`
"""

from __future__ import annotations

import argparse
import json
import sys

from app.core.config import settings
from app.db.session import init_db
from app.exchanges import get_exchange


def _cmd_train_all(args: argparse.Namespace) -> int:
    from app.api.routes.ml import _resolve_symbols
    from app.ml.train import train_all_models

    init_db()
    exchange = get_exchange(settings.exchange_id)
    symbols = _resolve_symbols(exchange, args.symbols)
    if not symbols:
        print(json.dumps({"error": "Eğitim için sembol bulunamadı."}, ensure_ascii=False))
        return 1

    results = train_all_models(exchange, symbols)
    output = {
        "symbols_used": len(symbols),
        "steps": [{"step": r.step, "ok": r.ok, "detail": r.detail} for r in results],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="4keys ağır ML işlemleri — canlı API process'inden AYRI çalıştırmak için.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_all_parser = subparsers.add_parser(
        "train-all",
        help="XGBoost -> meta-label -> LSTM -> online -> regime, sırayla (bkz. POST /ml/train-all ile AYNI mantık).",
    )
    train_all_parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Boş bırakılırsa screener + BTC-öncelikli seçim (select_training_symbols) kullanılır.",
    )
    train_all_parser.set_defaults(func=_cmd_train_all)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
