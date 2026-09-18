"""Централизованная конфигурация проекта: пути и runtime-настройки из env.

Все значения читаются из переменных окружения (см. .env.example) с дефолтами,
совместимыми с docker-compose. Модуль не импортирует тяжёлых зависимостей.
"""

import os
import sys


def init_console_utf8() -> None:
    """Переключает stdout/stderr на UTF-8: иначе русский текст в Windows-терминале
    выводится как '?' или крякозябры."""
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except (ImportError, AttributeError, OSError):
            pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name, "1" if default else "0").lower()
    return val in ("1", "true", "yes", "on")


# --- Пути (внутри контейнера совпадают с WORKDIR /app и named-томами) ---
CHROMADB_DIR = os.getenv("CHROMADB_DIR", "chromadb")
DATA_DIR = os.getenv("DATA_DIR", "data")
DICTIONARY_FILE = os.getenv("DICTIONARY_FILE", "corporate_dictionary.json")
HISTORY_FILE = os.getenv("HISTORY_FILE", "history.json")

# --- Модель / Ollama ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:e2b-it-qat")
LLM_MAX_CONTEXT = 32000
LLM_NUM_CTX = _env_int("LLM_NUM_CTX", 8192)
LLM_SELF_CHECK = _env_flag("LLM_SELF_CHECK", default=True)
MAX_HISTORY_TURNS = _env_int("MAX_HISTORY_TURNS", 6)
OLLAMA_MAX_RETRIES = _env_int("OLLAMA_MAX_RETRIES", 3)
AGENT_MAX_ITERATIONS = _env_int("AGENT_MAX_ITERATIONS", 3)

# --- Сервис ---
WARMUP_MODELS = _env_flag("WARMUP_MODELS", default=True)
UPLOAD_MAX_MB = _env_int("UPLOAD_MAX_MB", 50)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
# Ограничение параллельных тяжёлых вызовов (энкод BGE-M3 / реранкер) на CPU.
SEARCH_CONCURRENCY = _env_int("SEARCH_CONCURRENCY", 2)