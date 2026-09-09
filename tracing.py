"""Langfuse observability — tracing каждого шага RAG-пайплайна.

Необязательный модуль: если LANGFUSE_* переменные не заданы, все функции — no-op.
При включении добавляет ~0ms к latency (lazy init, HTTP-батчинг).

Env vars:
  LANGFUSE_HOST        — URL self-hosted Langfuse (default: http://langfuse:3000)
  LANGFUSE_PUBLIC_KEY  — public key из Langfuse project settings
  LANGFUSE_SECRET_KEY  — secret key из Langfuse project settings
  LANGFUSE_ENABLED     — explicit toggle: 1/true/on или 0/false/off (default: auto-detect)
"""

import os
import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_langfuse = None
_initialized = False


def _should_enable() -> bool:
    """Определяет, включено ли трейсирование."""
    enabled = os.getenv("LANGFUSE_ENABLED", "").lower()
    if enabled in ("1", "true", "yes", "on"):
        return True
    if enabled in ("0", "false", "no", "off"):
        return False
    # Auto-detect: включаем, если заданы оба ключа
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_langfuse():
    """Ленивая инициализация Langfuse клиента. Вызывается один раз."""
    global _langfuse, _initialized
    if _initialized:
        return _langfuse
    _initialized = True

    if not _should_enable():
        logger.info("Tracing отключен (LANGFUSE_* не заданы)")
        return None

    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            host=os.getenv("LANGFUSE_HOST", "http://langfuse:3000"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        )
        logger.info("Langfuse tracing инициализирован: %s", os.getenv("LANGFUSE_HOST"))
    except Exception as e:
        logger.warning("Langfuse недоступен, tracing отключен: %s", e)
        _langfuse = None

    return _langfuse


def flush():
    """Отправляет накопленные события в Langfuse. Вызывать при shutdown."""
    lf = get_langfuse()
    if lf:
        lf.flush()


@contextmanager
def trace_query(query: str, session_id: str, user_id: str | None = None):
    """Топовый трейс на весь запрос к /chat.

    Usage:
        with trace_query(query, session_id) as trace:
            result = agent.process_query(query, history)
            trace.update(output=result)
    """
    lf = get_langfuse()
    if not lf:
        yield _NoopTrace()
        return

    trace = lf.trace(
        name="gpn.chat",
        input={"query": query, "session_id": session_id},
        user_id=user_id or session_id,
        metadata={"version": "1.0.0"},
    )
    start = time.perf_counter()
    try:
        yield trace
    except Exception as e:
        trace.update(metadata={"error": str(e)})
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        trace.update(metadata={"latency_ms": round(elapsed_ms, 1)})


@contextmanager
def create_span(trace_or_parent, name: str, metadata: dict | None = None, input_data=None):
    """Создаёт child-span внутри трейса или родительского спана.

    Usage:
        with create_span(trace, "dense_search", {"top_k": 8}) as span:
            results = do_dense_search(query)
            span.update(output={"count": len(results)})
    """
    if not trace_or_parent or not get_langfuse():
        yield _NoopSpan()
        return

    span = trace_or_parent.span(
        name=name,
        input=input_data,
        metadata=metadata or {},
    )
    start = time.perf_counter()
    try:
        yield span
    except Exception as e:
        span.update(metadata={"error": str(e)})
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        current_meta = span.metadata if hasattr(span, 'metadata') else {}
        if current_meta is None:
            current_meta = {}
        span.update(metadata={**current_meta, "latency_ms": round(elapsed_ms, 1)})


def score(trace, name: str, value: float, comment: str | None = None):
    """Логирует числовую метрику (latency, score, etc.) к трейсу."""
    lf = get_langfuse()
    if lf and trace:
        lf.score(
            trace_id=trace.id,
            name=name,
            value=value,
            comment=comment,
        )


class _NoopTrace:
    """Заглушка: методы-бездейственники, чтобы код не проверял if trace."""
    id = None
    def span(self, **kw): return _NoopSpan()
    def update(self, **kw): pass
    def score(self, **kw): pass


class _NoopSpan:
    """Заглушка-спан для no-op режима."""
    metadata = {}
    def update(self, **kw): pass
