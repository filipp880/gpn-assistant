from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid
import logging
import asyncio
import os
import re
from datetime import datetime
from io import BytesIO
import httpx
from tracing import trace_query, flush as trace_flush

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
logger = logging.getLogger("gpn_assistant_api")

# --- Инициализация FastAPI ---
app = FastAPI(
    title="GPN Corporate AI Assistant API",
    description="""
    Корпоративный AI-ассистент для ПАО «Газпром нефть».
    Понимает внутренний сленг, исправляет опечатки в аббревиатурах (ГПНР -> Газпромнефть-Развитие)
    и использует гибридный поиск (dense + sparse от BGE-M3) с реранкингом.
    """,
    version="1.0.0",
    openapi_tags=[
        {"name": "Core", "description": "Эндпоинты для автотестов жюри и фронтенда"},
        {"name": "System", "description": "Мониторинг и управление сессиями"}
    ]
)

# CORS (чтобы можно было подключить простой UI на Streamlit/React для демо)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Схемы (OpenAPI спецификация) ---
class ChatRequest(BaseModel):
    query: str = Field(..., description="Запрос пользователя", example="Какая EBITDA у ГПНР за 2025 год?")
    session_id: Optional[str] = Field(None, description="ID сессии для мульти-тернового диалога")

class ResolvedTerm(BaseModel):
    original: str = Field(..., description="Исходный токен с опечаткой или сленгом")
    canonical: str = Field(..., description="Нормализованное значение из словаря")
    score: float = Field(..., description="Уверенность Fuzzy Matching (0-100)")

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[str] = Field(default_factory=list, description="Источники из ChromaDB")
    resolved_terms: List[ResolvedTerm] = Field(default_factory=list, description="Лог распознавания аббревиатур")
    raw_context: str = Field("", description="Сырой контекст, ушедший в промт LLM")
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    ollama_status: str
    db_status: str
    models_ready: bool
    message: str

class DictionaryItem(BaseModel):
    key: str = Field(..., description="Аббревиатура/термин", example="КРС")
    value: str = Field(..., description="Расшифровка/значение", example="Капитальный ремонт скважин")

class DictionaryResponse(BaseModel):
    count: int
    dictionary: Dict[str, str]

# --- Persistent Session Storage (Redis + in-memory fallback) ---
from sessions import SessionStore
store = SessionStore(os.getenv("REDIS_URL"))

# --- АДАПТЕР К ТВОЕМУ КОДУ ---
def _run_sync_agent(query: str, session_id: str, history: list, trace=None) -> dict:
    from agent import run_my_agent_logic
    return run_my_agent_logic(query, session_id, history, trace=trace)


async def execute_agent(query: str, session_id: str, trace=None) -> dict:
    """Запускает синхронный код агента в отдельном потоке, не блокируя FastAPI"""
    history = store.get_history(session_id)
    # asyncio.to_thread - спасение для синхронных LLM/ChromaDB вызовов в FastAPI
    result = await asyncio.to_thread(_run_sync_agent, query, session_id, history, trace=trace)
    return result


# --- HTTP ENDPOINTS ---
@app.post("/chat", response_model=ChatResponse, tags=["Core"])
async def chat_endpoint(request: ChatRequest):
    start_time = datetime.now()
    session_id = request.session_id or str(uuid.uuid4())
    
    store.append_message(session_id, "user", request.query)
    logger.info(f"[{session_id}] Запрос: {request.query}")
    
    with trace_query(request.query, session_id) as trace:
        try:
            agent_result = await execute_agent(request.query, session_id, trace=trace)
            trace.update(output={
                "answer": agent_result.get("answer", ""),
                "sources": agent_result.get("sources", []),
                "resolved_terms": agent_result.get("resolved_terms", []),
            })
        except Exception as e:
            logger.error(f"[{session_id}] Ошибка: {str(e)}")
            trace.update(metadata={"error": str(e)})
            from agent import OllamaUnavailableError
            if isinstance(e, OllamaUnavailableError):
                model = os.getenv("LLM_MODEL", "gemma4:e2b-it-qat")
                detail = f"Ollama недоступна. Проверьте, что сервис запущен и модель {model} загружена."
                raise HTTPException(status_code=503, detail=detail)
            raise HTTPException(status_code=500, detail=str(e))
            
    end_time = datetime.now()
    latency_ms = (end_time - start_time).total_seconds() * 1000
    logger.info("[%s] ответ за %.0f мс | источники: %s | терминов: %d",
                session_id, latency_ms,
                agent_result.get("sources", []),
                len(agent_result.get("resolved_terms", [])))

    store.append_message(session_id, "assistant", agent_result.get("answer", ""))
    
    return ChatResponse(
        session_id=session_id,
        answer=agent_result.get("answer", ""),
        sources=agent_result.get("sources", []),
        resolved_terms=agent_result.get("resolved_terms", []),
        raw_context=agent_result.get("raw_context", ""),
        latency_ms=round(latency_ms, 2)
    )

@app.get("/history/{session_id}", tags=["System"])
async def get_history(session_id: str):
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return {"session_id": session_id, "history": store.get_history(session_id)}

@app.delete("/history/{session_id}", tags=["System"])
async def clear_history(session_id: str):
    store.delete(session_id)
    return {"status": "success"}

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Readiness: Ollama + база знаний + модели (жюри часто пингуют этот эндпоинт)."""
    ollama_up = False
    ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/tags"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(ollama_url, timeout=3.0)
            ollama_up = r.status_code == 200
    except Exception:
        pass

    from retrieval import status_info
    info = status_info()
    db_ready = info["db_ready"]
    models_ready = info["bge_m3_loaded"] and info["reranker_loaded"]

    if ollama_up and db_ready:
        return HealthResponse(
            status="ok", ollama_status="up", db_status="ready",
            models_ready=models_ready, message="Ready for tests",
        )
    return HealthResponse(
        status="degraded",
        ollama_status="up" if ollama_up else "down",
        db_status="ready" if db_ready else "not_ready",
        models_ready=models_ready,
        message="Initializing..." if ollama_up else "Ollama unavailable",
    )

@app.post("/reindex", tags=["System"])
async def reindex():
    """Перечитывает документы из data/ и перестраивает индекс.

    Нужно после добавления/изменения файлов в data/ — без перезапуска сервиса.
    """
    try:
        await asyncio.to_thread(_rebuild_index)
        return {"status": "success", "message": "Индекс пересобран из data/"}
    except Exception as e:
        logger.error(f"Ошибка переиндексации: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка переиндексации: {str(e)}")


# --- Корпоративный словарь (редактирование) ---

@app.get("/dictionary", response_model=DictionaryResponse, tags=["System"])
async def get_dictionary():
    """Возвращает корпоративный словарь аббревиатур."""
    from agent import agent_instance
    return {"count": len(agent_instance.dictionary), "dictionary": agent_instance.dictionary}


@app.put("/dictionary", response_model=DictionaryResponse, tags=["System"])
async def replace_dictionary(items: Dict[str, str]):
    """Полностью заменяет словарь: {аббревиатура: расшифровка}."""
    from agent import agent_instance, save_dictionary
    cleaned = {
        k.strip(): v.strip()
        for k, v in items.items()
        if k and k.strip() and v and v.strip()
    }
    agent_instance.dictionary = cleaned
    save_dictionary(cleaned)
    logger.info("Словарь заменён полностью: %d терминов", len(cleaned))
    return {"count": len(cleaned), "dictionary": cleaned}


@app.post("/dictionary", response_model=DictionaryResponse, tags=["System"])
async def upsert_dictionary_item(item: DictionaryItem):
    """Добавляет или обновляет один термин словаря."""
    from agent import agent_instance, save_dictionary
    key = item.key.strip()
    value = item.value.strip()
    if not key or not value:
        raise HTTPException(status_code=422, detail="key и value не должны быть пустыми")
    agent_instance.dictionary[key] = value
    save_dictionary(agent_instance.dictionary)
    logger.info("Словарь: добавлено/обновлено '%s' -> '%s'", key, value)
    return {"count": len(agent_instance.dictionary), "dictionary": agent_instance.dictionary}


@app.delete("/dictionary/{key}", response_model=DictionaryResponse, tags=["System"])
async def delete_dictionary_item(key: str):
    """Удаляет термин из словаря."""
    from agent import agent_instance, save_dictionary
    removed = agent_instance.dictionary.pop(key, None)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"Термин '{key}' не найден")
    save_dictionary(agent_instance.dictionary)
    logger.info("Словарь: удалён термин '%s'", key)
    return {"count": len(agent_instance.dictionary), "dictionary": agent_instance.dictionary}


# --- Загрузка документов ---

_ALLOWED_UPLOAD_EXT = {".pdf", ".txt"}


def _rebuild_index() -> None:
    """Перечитывает data/ и перестраивает индекс (общий для /reindex и /upload)."""
    import ingest
    ingest.main()
    from retrieval import reload_store
    reload_store()


@app.post("/upload", tags=["System"])
async def upload_document(file: UploadFile = File(..., description="PDF или TXT документ")):
    """Загружает документ (PDF/TXT) в базу знаний и пересобирает индекс."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _ALLOWED_UPLOAD_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Допустимые форматы: PDF, TXT (получено: '{ext or 'без расширения'}')",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Пустой файл")

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(content))
            if len(reader.pages) == 0:
                raise ValueError("в PDF нет страниц")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Не удалось прочитать PDF: {e}")

    os.makedirs("data", exist_ok=True)
    base_name = os.path.basename(file.filename or "upload")
    safe_name = re.sub(r"[^\w.()-]", "_", base_name).strip("_")
    if not os.path.splitext(safe_name)[1].lower():
        safe_name += ext
    safe_name = safe_name or f"upload{ext}"
    path = os.path.join("data", safe_name)

    with open(path, "wb") as f:
        f.write(content)
    logger.info("Загружен документ: %s (%d байт)", path, len(content))

    try:
        await asyncio.to_thread(_rebuild_index)
    except Exception as e:
        logger.error("Ошибка переиндексации после загрузки: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка переиндексации: {e}")

    return {
        "status": "success",
        "file": path,
        "message": "Документ сохранён и индекс пересобран",
    }

@app.on_event("startup")
async def startup_event():
    logger.info("Сервер GPN Assistant запущен. Swagger UI: http://localhost:8000/docs")
    # Инициализация Langfuse tracing (lazy, если ключи заданы)
    from tracing import get_langfuse
    get_langfuse()

@app.on_event("shutdown")
async def shutdown_event():
    trace_flush()

@app.on_event("startup")
async def warmup_models():
    """
    Прогрев ML-моделей при старте сервера.
    Это гарантирует, что первый запрос от жюри не будет висеть 2 минуты.
    """
    logger.info("🔥 Начинаем прогрев ML-моделей...")
    
    try:
        # 1. Прогреваем BGE-M3 (dense + sparse)
        from retrieval import get_bge_m3, get_reranker, store_ready, reload_store
        model = get_bge_m3()
        _ = model.encode(["прогрев"], return_dense=True, return_sparse=True)
        logger.info("✅ BGE-M3 прогрет")
        
        # 2. Прогреваем реранкер
        _ = get_reranker().predict([["запрос", "документ"]])
        logger.info("✅ Реранкер прогрет")
        
        # 3. Инициализируем словарь агента
        from agent import agent_instance
        agent_instance._resolve_slang("ГПН")
        logger.info("✅ Словарь прогрет")

        # 3а. Проверяем базу знаний: сервис НЕ падает, если индекс ещё не построен
        reload_store()
        if store_ready():
            logger.info("✅ База знаний загружена (kbase)")
        else:
            logger.warning("⚠️ База знаний не найдена — выполните: python ingest.py, либо POST /reindex")
        
        # 4. Прогреваем саму LLM (Ollama): загружаем модель при старте,
        #    чтобы первый запрос жюри не тратил время на загрузку весов
        from agent import agent_instance as agent
        agent._client.chat(
            model=agent.model_name,
            messages=[{'role': 'user', 'content': 'прогрев'}],
            options={'num_predict': 1, 'num_ctx': agent.num_ctx},
            keep_alive=-1,
        )
        logger.info(f"✅ LLM прогрета: {agent.model_name}")
        
        logger.info("🚀 Сервер полностью готов к приему запросов!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка прогрева: {e}")