from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from tracing import flush as trace_flush

# Русский текст в консоли (логи uvicorn/прогрева) не должен превращаться в '?'
config.init_console_utf8()

# --- Настройка логирования ---
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
logger = logging.getLogger("gpn_assistant_api")


def _warmup_models() -> None:
    """Прогрев ML-моделей в фоне (не блокирует старт сервера):
    первый запрос жюри не висит, а /health тем временем показывает degraded."""
    logger.info("Начинаем прогрев ML-моделей...")
    try:
        # 1. Прогреваем BGE-M3 (dense + sparse)
        from retrieval import get_bge_m3, get_reranker, store_ready, reload_store
        model = get_bge_m3()
        model.encode(["прогрев"], return_dense=True, return_sparse=True)
        logger.info("BGE-M3 прогрет")

        # 2. Прогреваем реранкер
        get_reranker().predict([["запрос", "документ"]])
        logger.info("Реранкер прогрет")

        # 3. Инициализируем словарь агента
        from agent import get_agent
        get_agent()._resolve_slang("ГПН")
        logger.info("Словарь прогрет")

        # 3а. Проверяем базу знаний: сервис НЕ падает, если индекс ещё не построен
        reload_store()
        if store_ready():
            logger.info("База знаний загружена (kbase)")
        else:
            logger.warning("База знаний не найдена — выполните: python ingest.py, либо POST /reindex")

        # 4. Прогреваем саму LLM (Ollama): загружаем модель при старте,
        #    чтобы первый запрос жюри не тратил время на загрузку весов
        agent = get_agent()
        agent._client.chat(
            model=agent.model_name,
            messages=[{'role': 'user', 'content': 'прогрев'}],
            options={'num_predict': 1, 'num_ctx': agent.num_ctx},
            keep_alive=-1,
        )
        logger.info("LLM прогрета: %s", agent.model_name)
        logger.info("Сервер полностью готов к приёму запросов!")
    except Exception as e:
        logger.error("Ошибка прогрева: %s", e)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Прогрев моделей в фоне при старте, сброс трейсов при выходе."""
    logger.info("Сервер GPN Assistant запущен. Swagger UI: http://localhost:8000/docs")
    # Инициализация Langfuse tracing (lazy, если ключи заданы)
    from tracing import get_langfuse
    get_langfuse()
    if config.WARMUP_MODELS:
        import threading
        threading.Thread(target=_warmup_models, name="model-warmup", daemon=True).start()
    try:
        yield
    finally:
        trace_flush()


# --- Инициализация FastAPI ---
app = FastAPI(
    title="GPN Corporate AI Assistant API",
    description="""
    Корпоративный AI-ассистент для ПАО «Газпром нефть».
    Понимает внутренний сленг, исправляет опечатки в аббревиатурах (ГПНР -> Газпромнефть-Развитие)
    и использует гибридный поиск (dense + sparse от BGE-M3) с реранкингом.
    """,
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Core", "description": "Эндпоинты для автотестов жюри и фронтенда"},
        {"name": "System", "description": "Мониторинг и управление сессиями"}
    ]
)

# CORS (чтобы можно было подключить простой UI на Streamlit/React для демо).
# В .env.example: CORS_ORIGINS=* — на проде заменить на конкретные домены.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in config.CORS_ORIGINS.split(",") if o.strip()] or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Роутеры (поделены на модули пакета routes/) ---
from routes import admin, chat, dictionary, sessions

app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(dictionary.router)
app.include_router(admin.router)