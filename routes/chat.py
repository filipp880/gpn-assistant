"""Эндпоинт /chat — основной контракт для фронтенда и автотестов жюри."""

import asyncio
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException

import config
from schemas import ChatRequest, ChatResponse
from store import store
from tracing import trace_query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Core"])


def _run_sync_agent(query: str, session_id: str, history: list, trace=None) -> dict:
    from agent import run_my_agent_logic
    return run_my_agent_logic(query, session_id, history, trace=trace)


async def _execute_agent(query: str, session_id: str, trace=None) -> dict:
    """Запускает синхронный код агента в отдельном потоке, не блокируя FastAPI."""
    history = store.get_history(session_id)
    return await asyncio.to_thread(_run_sync_agent, query, session_id, history, trace=trace)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = datetime.now()
    session_id = request.session_id or str(uuid.uuid4())

    store.append_message(session_id, "user", request.query)
    logger.info("[%s] Запрос: %s", session_id, request.query)

    with trace_query(request.query, session_id) as trace:
        try:
            agent_result = await _execute_agent(request.query, session_id, trace=trace)
            trace.update(output={
                "answer": agent_result.get("answer", ""),
                "sources": agent_result.get("sources", []),
                "resolved_terms": agent_result.get("resolved_terms", []),
            })
        except Exception as e:
            logger.error("[%s] Ошибка: %s", session_id, e)
            trace.update(metadata={"error": str(e)})
            from agent import OllamaUnavailableError
            if isinstance(e, OllamaUnavailableError):
                detail = (f"Ollama недоступна. Проверьте, что сервис запущен "
                          f"и модель {config.LLM_MODEL} загружена.")
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
        latency_ms=round(latency_ms, 2),
    )