"""Haftalık (auto_retrain/periodic_optimization) job'ların "son ne zaman
GERÇEKTEN çalıştı" bilgisini süreç yeniden başlasa da KALICI tutar.

Neden gerekli: `BackgroundScheduler`'a `next_run_time` verilmeden eklenen
bir "interval" job'ı, ilk çalışmasını job'un EKLENDİĞİ andan (yani süreç
BAŞLADIĞI andan) bir tam interval sonra planlar (bkz. `scheduler.py`daki
bilinçli "uygulama her açılışta ağır eğitimleri hemen tetiklemesin" kararı).
Sürekli açık kalan bir sunucuda bu sorun değildir. Ama sık kapanıp açılan
BİR MAKİNEDE (bu proje burada geliştiriliyor), her yeniden başlatma sayacı
SIFIRLAR — 7 günlük aralık, süreç 7 günden daha sık yeniden başladığı
sürece HİÇBİR ZAMAN dolmaz, iş asla çalışmaz.

Çözüm: her haftalık job'un son (denenen) çalışma zamanı, kalıcı
`fourkeys_ml_artifacts` volume'üne (model dosyalarıyla AYNI, zaten
container yeniden oluşturulsa da KORUNAN volume) küçük bir JSON dosyası
olarak yazılır. Açılışta bu dosya okunur; eğer son çalışmadan bu yana
job'un aralığından FAZLA zaman geçmişse (ör. bilgisayar günlerce kapalı
kaldıysa), job normal 7 gün beklemek yerine YAKINDA (bkz.
`scheduler.py`daki kademeli gecikmeler) tetiklenir — kaçırılan çalışma
"yakalanır"."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_PATH = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "scheduler_last_success.json"


def read_last_run(job_id: str) -> datetime | None:
    try:
        data = json.loads(STATE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    raw = data.get(job_id)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def record_run(job_id: str) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {}
        if STATE_PATH.exists():
            try:
                data = json.loads(STATE_PATH.read_text())
            except json.JSONDecodeError:
                data = {}
        data[job_id] = datetime.now(timezone.utc).isoformat()
        STATE_PATH.write_text(json.dumps(data))
    except OSError:
        logger.exception("scheduler son-çalışma kaydı yazılamadı: %s", job_id)
