# Araştırma/test imajı: backend imajı + araştırma bağımlılıkları (prod imajına GİRMEZ).
# Derleme (depo kökünden):  docker build -t 4keys-research-test -f backend/research/Dockerfile backend
# Çalıştırma:               docker run --rm -v "$PWD:/repo" -w /repo/backend 4keys-research-test python -m pytest -q tests/research
FROM 4keys-backend-test:latest
# LightGBM OpenMP çalışma zamanı ister (python:slim'de yok); git: deney kaydına commit hash'i yazılsın (registry.git_commit)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git && rm -rf /var/lib/apt/lists/*     && git config --system --add safe.directory '*' && git config --system core.autocrlf input
# safe.directory: depo konteynere host kullanıcısıyla bağlanır (-v); sahiplik farkı yüzünden git aksi halde çalışmayı reddeder.
# core.autocrlf input: Windows host'ta dosyalar CRLF'tir (autocrlf=true); normalizasyon olmadan konteynerdeki git her dosyayı
# "değişmiş" sayar ve kayıttaki hash yanlışlıkla "+dirty" olur.
COPY requirements-research.txt /tmp/requirements-research.txt
RUN pip install --no-cache-dir -r /tmp/requirements-research.txt
