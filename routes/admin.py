"""Системные эндпоинты: health, переиндексация, загрузка документов."""

import asyncio
import logging
import os
import re
from io import BytesIO

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile

import config
from schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System"])

_ALLOWED_UPLOAD_EXT = {".pdf", ".txt"}


def _rebuild_index() -> None:
    """Перечитывает data/ и перестраивает индекс (общий для /reindex и /upload)."""
    import ingest
    ingest.main()
    from retrieval import reload_store
    reload_store()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Readiness: Ollama + база знаний + модели (жюри часто пингуют этот эндпоинт)."""
    ollama_up = False
    ollama_url = config.OLLAMA_HOST + "/api/tags"
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


@router.post("/reindex")
async def reindex():
    """Перечитывает документы из data/ и перестраивает индекс.

    Нужно после добавления/изменения файлов в data/ — без перезапуска сервиса.
    """
    try:
        await asyncio.to_thread(_rebuild_index)
        return {"status": "success", "message": "Индекс пересобран из data/"}
    except Exception as e:
        logger.error("Ошибка переиндексации: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка переиндексации: {e}")


@router.post("/upload")
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
    if len(content) > config.UPLOAD_MAX_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Файл больше лимита {config.UPLOAD_MAX_MB} МБ",
        )

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(content))
            if len(reader.pages) == 0:
                raise ValueError("в PDF нет страниц")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Не удалось прочитать PDF: {e}")

    os.makedirs(config.DATA_DIR, exist_ok=True)
    base_name = os.path.basename(file.filename or "upload")
    safe_name = re.sub(r"[^\w.()-]", "_", base_name).strip("_")
    if not os.path.splitext(safe_name)[1].lower():
        safe_name += ext
    safe_name = safe_name or f"upload{ext}"
    path = os.path.join(config.DATA_DIR, safe_name)

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