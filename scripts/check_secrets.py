#!/usr/bin/env python3
"""Git tarafından takip edilen dosyalarda sızmış API anahtarı/sır olup
olmadığını tarar (Güvenlik Protokolü Bölüm 9.5). CI'da veya bir pre-commit
hook olarak çalıştırılabilir:

    python scripts/check_secrets.py

Sıfır olmayan çıkış kodu = şüpheli bulgu var (veya .env git'e eklenmiş).
"""

import re
import subprocess
import sys

SUSPICIOUS_PATTERNS = [
    (re.compile(r"FOURKEYS_\w*(API_KEY|API_SECRET|CLIENT_SECRET)\s*=\s*[\"']?[A-Za-z0-9_\-]{8,}"), "dolu bir gizli anahtar değeri"),
    (re.compile(r"\b[A-Za-z0-9]{64}\b"), "64 karakterlik yüksek entropili dize (Binance API secret uzunluğu)"),
    (re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"), "özel anahtar bloğu"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key deseni"),
]

# Bu dosyaların kendisi örnek/şablon/test olduğu için tarama dışı bırakılır.
EXCLUDED_PATHS = {"backend/.env.example", "scripts/check_secrets.py"}


def tracked_files() -> list[str]:
    result = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    files = tracked_files()
    findings: list[str] = []

    for path in files:
        if path in EXCLUDED_PATHS:
            continue
        if path == ".env" or path.endswith("/.env"):
            findings.append(f"{path}: .env dosyası git'e eklenmiş — ASLA commit edilmemeli, .gitignore kontrol edin.")
            continue
        try:
            with open(path, "r", errors="ignore") as f:
                content = f.read()
        except (OSError, IsADirectoryError):
            continue

        for pattern, description in SUSPICIOUS_PATTERNS:
            if pattern.search(content):
                findings.append(f"{path}: {description} tespit edildi.")
                break  # bir dosya için bir bulgu yeterli, gürültüyü azaltır

    if findings:
        print("UYARI: Aşağıdaki şüpheli bulgular tespit edildi:\n")
        for f in findings:
            print(f"  - {f}")
        print("\nBunlar gerçek bir sır ise HEMEN o anahtarı iptal edip yenisini oluşturun,")
        print("ardından git geçmişinden temizleyin (örn. git filter-repo).")
        return 1

    print(f"OK: {len(files)} dosya tarandı, şüpheli bir sır bulunamadı.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
