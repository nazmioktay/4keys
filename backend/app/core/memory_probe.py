"""Tek seferlik bellek (RSS) tanılama yardımcısı — bkz. README "OOM üretim
olayı": eğitim, mum sayısından (`lookback`) BAĞIMSIZ olarak sabit bir
~1.2-1.4GB seviyesinde ölüyor (kanıt: LOOKBACK=1000 ve LOOKBACK=5000 hemen
hemen AYNI anon-rss'te ölüyor). Bu sabit maliyetin NEREDE oluştuğunu
(import mı, sembol-başı işleme mi) bulmak için `train-all` akışının
belirli noktalarına yerleştirilir.

Yalnızca `MEMORY_PROBE=1` ortam değişkeni ayarlıyken bir şey YAPAR —
varsayılan (kapalı) durumda üretimde sıfır ek maliyet/gürültü."""

import os


def log_rss(label: str) -> None:
    if os.environ.get("MEMORY_PROBE") != "1":
        return
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    print(f"[memory_probe] {label}: {line.split()[1]} kB", flush=True)
                    return
    except OSError:
        pass
