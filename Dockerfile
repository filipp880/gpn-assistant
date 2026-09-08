FROM python:3.10-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === PRE-DOWNLOAD ML МОДЕЛЕЙ (ГЛАВНЫЙ ФИКС) ===
# Скачиваем модели в кэш HuggingFace прямо при сборке образа
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python ingest.py && uvicorn app:app --host 0.0.0.0 --port 8000"]