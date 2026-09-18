"""Хранилище корпоративного словаря: JSON или CSV.

Формат выбирается по расширению файла (config.DICTIONARY_FILE):
- `.json` — привычный вид { "аббревиатура": "расшифровка" };
- `.csv`  — таблица `term,definition`, UTF-8 с BOM (корректно открывается в Excel).
Терм и расшифровка квотируются csv-модулем, поэтому могут содержать запятые.
"""

import csv
import json
import logging
import os

logger = logging.getLogger(__name__)

CSV_HEADERS = ("term", "definition")


def is_csv(path: str) -> bool:
    return bool(path) and path.lower().endswith(".csv")


def load_dictionary(path: str) -> dict:
    """Загружает словарь из JSON или CSV. Если файла нет или он битый — {}."""
    if not path or not os.path.exists(path):
        return {}
    try:
        if is_csv(path):
            return _load_csv(path)
        return _load_json(path)
    except (json.JSONDecodeError, csv.Error, OSError) as e:
        logger.error("Ошибка чтения словаря %s: %s", path, e)
        return {}


def save_dictionary(dictionary: dict, path: str) -> None:
    """Сохраняет словарь в JSON или CSV согласно расширению файла."""
    if is_csv(path):
        _save_csv(dictionary, path)
    else:
        _save_json(dictionary, path)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(dictionary: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)


def _load_csv(path: str) -> dict:
    result = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    for row in rows[1:]:  # первая строка — заголовок (term,definition)
        if len(row) < 2:
            continue
        term = (row[0] or "").strip()
        definition = (row[1] or "").strip()
        if term and not result.get(term):
            result[term] = definition
    return result


def _save_csv(dictionary: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
        for term, definition in dictionary.items():
            writer.writerow([term, definition])