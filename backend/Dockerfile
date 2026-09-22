FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
ENV PIP_DEFAULT_TIMEOUT=200
# torch'un varsayılan PyPI paketi GPU/CUDA sürümüdür (~3GB, bu container'da
# kullanılmayan nvidia_* bağımlılıkları indirir, build'i dramatik yavaşlatır).
# CPU-only resmi PyTorch indeksinden önce kurulur (~200MB); requirements.txt
# işlenirken pip `torch>=2.2` kısıtını zaten karşılandığı için CUDA sürümüyle
# DEĞİŞTİRMEZ.
RUN pip install --no-cache-dir --retries 10 torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --retries 10 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
