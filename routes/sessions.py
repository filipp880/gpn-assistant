"""Управление сессиями: история диалога."""

from fastapi import APIRouter, HTTPException

from store import store

router = APIRouter(tags=["System"])


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return {"session_id": session_id, "history": store.get_history(session_id)}


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    store.delete(session_id)
    return {"status": "success"}