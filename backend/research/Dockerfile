# Araştırma/test imajı: backend imajı + araştırma bağımlılıkları (prod imajına GİRMEZ).
# Derleme (depo kökünden):  docker build -t 4keys-research-test -f backend/research/Dockerfile backend
# Çalıştırma:               docker run --rm -v "$PWD:/repo" -w /repo/backend 4keys-research-test python -m pytest -q tests/research
FROM 4keys-backend-test:latest
# LightGBM OpenMP çalışma zamanı ister (python:slim'de yok)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-research.txt /tmp/requirements-research.txt
RUN pip install --no-cache-dir -r /tmp/requirements-research.txt
