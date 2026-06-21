FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir --timeout 600 --retries 5 \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --timeout 600 --retries 5 \
    -r requirements-api.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY helmet_v3_best.pt .
COPY api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
