"""Pydantic-схемы API (общие для роутеров и OpenAPI-спецификации)."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="Запрос пользователя", json_schema_extra={"example": "Какая EBITDA у ГПНР за 2025 год?"})
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
    key: str = Field(..., description="Аббревиатура/термин", json_schema_extra={"example": "КРС"})
    value: str = Field(..., description="Расшифровка/значение", json_schema_extra={"example": "Капитальный ремонт скважин"})


class DictionaryResponse(BaseModel):
    count: int
    dictionary: Dict[str, str]