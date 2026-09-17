# syntax=docker/dockerfile:1

FROM python:3.10-slim

WORKDIR /app

# Системные зависимости (нужны, если для какого-то пакета нет готового wheel)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

# Python-зависимости: BuildKit-кэш pip ускоряет пересборку (колёса не качаются
# заново при изменении исходников проекта). Версии зафиксированы в requirements.txt.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# === PRE-DOWNLOAD ML ИМА (единоразовое скачивание, запекаются в образ) ===
# BGE-M3 (~2.2 ГБ) + bge-reranker-v2-m3 (~2.2 ГБ). Слой пересобирается только когда
# меняется что-то выше (база/requirements); обычные изменения кода его не трогают,
# поэтому перый контейнер не качает модели из интернета.
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"

COPY . .

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=0

EXPOSE 8000

CMD ["sh", "-c", "python ingest.py && uvicorn app:app --host 0.0.0.0 --port 8000"]