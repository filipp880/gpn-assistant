from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
import logging
import asyncio
from datetime import datetime
import httpx

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
    и использует гибридный поиск (BM25 + Vector) с реранкингом.
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
    message: str

# --- In-Memory Storage для истории сессий ---
# Для хакатона этого достаточно. В проде заменяем на Redis/PostgreSQL.
sessions_store: Dict[str, List[Dict[str, Any]]] = {}

# --- АДАПТЕР К ТВОЕМУ КОДУ ---
def _run_sync_agent(query: str, session_id: str, history: list) -> dict:
    from agent import run_my_agent_logic
    return run_my_agent_logic(query, session_id, history)


async def execute_agent(query: str, session_id: str) -> dict:
    """Запускает синхронный код агента в отдельном потоке, не блокируя FastAPI"""
    history = sessions_store.get(session_id, [])
    # asyncio.to_thread - спасение для синхронных LLM/ChromaDB вызовов в FastAPI
    result = await asyncio.to_thread(_run_sync_agent, query, session_id, history)
    return result


# --- HTTP ENDPOINTS ---
@app.post("/chat", response_model=ChatResponse, tags=["Core"])
async def chat_endpoint(request: ChatRequest):
    start_time = datetime.now()
    session_id = request.session_id or str(uuid.uuid4())
    
    if session_id not in sessions_store:
        sessions_store[session_id] = []
        
    sessions_store[session_id].append({"role": "user", "content": request.query})
    logger.info(f"[{session_id}] Запрос: {request.query}")
    
    try:
        agent_result = await execute_agent(request.query, session_id)
    except Exception as e:
        logger.error(f"[{session_id}] Ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    end_time = datetime.now()
    latency_ms = (end_time - start_time).total_seconds() * 1000
    
    sessions_store[session_id].append({"role": "assistant", "content": agent_result.get("answer", "")})
    
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
    if session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return {"session_id": session_id, "history": sessions_store[session_id]}

@app.delete("/history/{session_id}", tags=["System"])
async def clear_history(session_id: str):
    if session_id in sessions_store:
        del sessions_store[session_id]
    return {"status": "success"}

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Проверка статуса Ollama (жюри часто пингуют этот эндпоинт)"""
    ollama_url = "http://localhost:11434/api/tags" # Или http://ollama:11434 в Docker
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(ollama_url, timeout=3.0)
            if r.status_code == 200:
                return HealthResponse(status="ok", ollama_status="up", message="Ready for tests")
    except Exception:
        pass
    return HealthResponse(status="degraded", ollama_status="down", message="Ollama unavailable")

@app.on_event("startup")
async def startup_event():
    logger.info("Сервер GPN Assistant запущен. Swagger UI: http://localhost:8000/docs")
    # Здесь можно вызвать pre-load векторной БД, чтобы первый запрос не был долгим

@app.on_event("startup")
async def warmup_models():
    """
    Прогрев ML-моделей при старте сервера.
    Это гарантирует, что первый запрос от жюри не будет висеть 2 минуты.
    """
    logger.info("🔥 Начинаем прогрев ML-моделей...")
    
    try:
        # 1. Прогреваем эмбеддер
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        _ = emb_model.encode(["прогрев"])
        logger.info("✅ Эмбеддер прогрет")
        
        # 2. Прогреваем реранкер
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
        _ = reranker.predict([["запрос", "документ"]])
        logger.info("✅ Реранкер прогрет")
        
        # 3. Инициализируем словарь агента
        from agent import agent_instance
        agent_instance._resolve_slang("ГПН")
        logger.info("✅ Словарь прогрет")
        
        logger.info("🚀 Сервер полностью готов к приему запросов!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка прогрева: {e}")