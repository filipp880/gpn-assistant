"""Корпоративный словарь аббревиатур (чтение и редактирование)."""

import logging
from typing import Dict

from fastapi import APIRouter, HTTPException

from agent import get_agent, save_dictionary
from schemas import DictionaryItem, DictionaryResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System"])


@router.get("/dictionary", response_model=DictionaryResponse)
async def get_dictionary():
    """Возвращает корпоративный словарь аббревиатур."""
    dictionary = get_agent().dictionary
    return {"count": len(dictionary), "dictionary": dictionary}


@router.put("/dictionary", response_model=DictionaryResponse)
async def replace_dictionary(items: Dict[str, str]):
    """Полностью заменяет словарь: {аббревиатура: расшифровка}."""
    cleaned = {
        k.strip(): v.strip()
        for k, v in items.items()
        if k and k.strip() and v and v.strip()
    }
    agent = get_agent()
    agent.dictionary = cleaned
    save_dictionary(cleaned)
    logger.info("Словарь заменён полностью: %d терминов", len(cleaned))
    return {"count": len(cleaned), "dictionary": cleaned}


@router.post("/dictionary", response_model=DictionaryResponse)
async def upsert_dictionary_item(item: DictionaryItem):
    """Добавляет или обновляет один термин словаря."""
    key = item.key.strip()
    value = item.value.strip()
    if not key or not value:
        raise HTTPException(status_code=422, detail="key и value не должны быть пустыми")
    agent = get_agent()
    agent.dictionary[key] = value
    save_dictionary(agent.dictionary)
    logger.info("Словарь: добавлено/обновлено '%s' -> '%s'", key, value)
    return {"count": len(agent.dictionary), "dictionary": agent.dictionary}


@router.delete("/dictionary/{key}", response_model=DictionaryResponse)
async def delete_dictionary_item(key: str):
    """Удаляет термин из словаря."""
    agent = get_agent()
    removed = agent.dictionary.pop(key, None)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"Термин '{key}' не найден")
    save_dictionary(agent.dictionary)
    logger.info("Словарь: удалён термин '%s'", key)
    return {"count": len(agent.dictionary), "dictionary": agent.dictionary}