import subprocess
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]


def test_importing_app_main_does_not_load_torch():
    """Regresyon: `torch` (yalnızca LSTM/PatchTST için kullanılır) TEK
    BAŞINA process RSS'ine ~460MB ekliyor (ölçüldü — bkz. README "backend
    baştaki bellek" araştırması) — üretim sunucusunda (3.7GB RAM, swap
    yoktu) bu, canlı API'nin boştaki bellek kullanımının önemli bir
    kısmını oluşturuyordu ve tekrarlanan OOM olaylarına katkıda
    bulunuyordu. `app.main` (canlı `uvicorn` process'inin başlangıçta
    yüklediği HER ŞEY) artık torch'u yalnızca GERÇEKTEN bir LSTM/PatchTST
    modeli yüklenecek/eğitilecek/tahmin edilecek fonksiyonun İÇİNDE (lazy)
    import ediyor — bkz. `app.ml.model_paths` (torch'suz `DEFAULT_LSTM_
    MODEL_PATH`/`DEFAULT_PATCHTST_MODEL_PATH`) ve `TYPE_CHECKING` korumalı
    tip belirteçleri.

    Bu testin AYRI bir subprocess'te çalışması ZORUNLU: aynı pytest
    process'inde çalışan DİĞER testler (LSTM/PatchTST testleri) torch'u
    zaten import etmiş olabilir — bu da testi (aynı process'te) anlamsız
    kılardı, çünkü `sys.modules` tüm test dosyaları arasında paylaşılır."""
    result = subprocess.run(
        [sys.executable, "-c", "import app.main; import sys; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
        cwd=str(_BACKEND_DIR),
        timeout=60,
    )
    assert result.returncode == 0, f"app.main import edilemedi: {result.stderr}"
    assert result.stdout.strip() == "False", (
        f"app.main import etmek torch'u da yükledi (regresyon) — stdout={result.stdout!r}, "
        f"stderr(son 2000 karakter)={result.stderr[-2000:]!r}"
    )
