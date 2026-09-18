"""Чтение корпоративных документов в текст для базы знаний.

Поддерживаемые форматы: TXT, PDF, DOCX, XLSX, PPTX.

Все функции принимают путь к файлу (str/PathLike) ИЛИ файл-подобный объект
(BytesIO) — валидация /upload и ingest используют один и тот же код. Если вход
задан путём, поток открывается и закрывается здесь; файл-подобный объект
вызывающий код владеет сам.

Тяжёлые библиотеки (python-docx, openpyxl, python-pptx, pypdf) импортируются
лениво внутри функций, поэтому сам модуль подключается без них — это удобно
для тестов и удерживает fetch от импорта пакетов документов на старте.

Особенности извлечения:
- DOCX: параграфы и таблицы в порядке следования в документе, заголовки
  помечаются markdown-`#`, таблица сериализуется построчно «ячейка | ячейка».
- XLSX: каждый лист превращается в блок "[Лист: Имя]", строки — в
  «заголовок: значение | ...» по первой строке из ≥2 непустых ячеек
  (заголовок таблицы). Так таблица извлекается «заголовок-осознанно», а не
  плоской строкой: запрос «EBITDA 2025» находит «2025: 135».
- PPTX: текст текстовых фреймов и таблиц по слайдам, перед каждым слайдом —
  markdown-заголовок «# Слайд N».
"""

import logging
import os
from datetime import date, datetime, time

logger = logging.getLogger(__name__)

# Единый список расширений базы знаний. Используется ingest.py (чтение),
# retrieval.py (fingerprint/_knowledge_files) и routes/admin.py (валидация /upload).
SUPPORTED_EXT = {".txt", ".pdf", ".docx", ".xlsx", ".pptx"}

# Заголовком таблицы считается строка из ≥2 непустых ячеек (предпочтительно с текстом).
_HEADER_MIN_CELLS = 2
# Заголовок ищется в первых строках листа (не заходим в данные при пустой шапке).
_HEADER_SEARCH_ROWS = 50


def _is_path(value):
    return isinstance(value, (str, os.PathLike))


def _open_stream(path_or_stream, mode="rb"):
    """Путь → открытый поток (вызывающий должен закрыть); объект → как есть."""
    if _is_path(path_or_stream):
        return open(path_or_stream, mode)
    return path_or_stream


def _close_stream(stream, path_or_stream) -> None:
    """Закрывает поток только если мы сами его открыли."""
    if _is_path(path_or_stream):
        stream.close()


def _value_to_text(value) -> str:
    """Ячейка таблицы/ячейка электронной таблицы → строка для поиска."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    text = str(value).strip()
    return text if text else ""


def read_txt(path_or_stream) -> str:
    """Читает UTF-8 текст. Файл-поток может быть бинарным (bytes)."""
    if _is_path(path_or_stream):
        with open(path_or_stream, "r", encoding="utf-8") as f:
            return f.read()
    data = path_or_stream.read()
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def read_pdf(path_or_stream) -> str:
    """Извлекает текст со страниц PDF (текстовые слои)."""
    from pypdf import PdfReader
    reader = PdfReader(path_or_stream)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# --- DOCX ---

def _docx_body_items(doc):
    """Перебирает содержимое документа в порядке следования: параграфы и таблицы."""
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def _docx_table_text(table) -> str:
    """Серия строк «ячейка | ячейка», пустые строки пропускаются."""
    lines = []
    for row in table.rows:
        cells = [c.strip() for c in (_cell_text(cell) for cell in row.cells) if c.strip()]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _cell_text(cell) -> str:
    try:
        return cell.text or ""
    except Exception:
        return ""


def _is_list_style(style_name: str) -> bool:
    lowered = (style_name or "").lower()
    return "list" in lowered or "bullet" in lowered or "number" in lowered


def read_docx(path_or_stream) -> str:
    """Текст DOCX: параграфы и таблицы в порядке документа; заголовки — markdown."""
    from docx import Document
    from docx.table import Table
    stream = _open_stream(path_or_stream)
    try:
        doc = Document(stream)
        lines = []
        for item in _docx_body_items(doc):
            if isinstance(item, Table):
                text = _docx_table_text(item)
                if text:
                    lines.append(text)
                continue
            text = (item.text or "").strip()
            if not text:
                continue
            style_name = ""
            try:
                style_name = item.style.name if item.style else ""
            except Exception:
                style_name = ""
            if style_name.lower().startswith("heading"):
                lines.append(f"# {text}")
            elif _is_list_style(style_name):
                lines.append(f"- {text}")
            else:
                lines.append(text)
        return "\n".join(lines)
    finally:
        _close_stream(stream, path_or_stream)


# --- XLSX ---

def _is_text_cell(cell: str) -> bool:
    """Ячейка выглядит как текст, а не как число/дата."""
    if not cell:
        return False
    try:
        float(cell.replace(",", "."))
    except ValueError:
        return True
    return False


def _find_header_row(rows: list) -> int | None:
    """Индекс строки-заголовка: первая строка с ≥2 непустыми ячейками.

    Предпочитается строка, в которой есть текстовая ячейка (а не только числа):
    строки-названия и строки без заголовка (одна непустая ячейка по ширине
    листа, чисто числовые данные) пропускаются. Если такой строки нет —
    заголовком считается первая строка из ≥2 ячеек (например, шапка
    «2024 | 2025», целиком состоящая из чисел). Пустой список — None.
    """
    first_multi: int | None = None
    for idx, row in enumerate(rows[:_HEADER_SEARCH_ROWS]):
        if len(row) < _HEADER_MIN_CELLS:
            continue
        if first_multi is None:
            first_multi = idx
        if any(_is_text_cell(cell) for cell in row):
            return idx
    return first_multi


def _serialize_row(row: list, header: list | None) -> str:
    """Строка таблицы с контекстом заголовков: «показатель: 120 | 2024: 135»."""
    if header:
        pairs = []
        for i, value in enumerate(row):
            label = header[i] if i < len(header) and header[i] else f"колонка {i + 1}"
            pairs.append(f"{label}: {value}")
        return " | ".join(pairs)
    return " | ".join(row)


def read_xlsx(path_or_stream) -> str:
    """Заголовок-осознанная сериализация всех листов .xlsx.

    С каждой строкой данных идут её заголовки («Показатель: EBITDA | 2025: 135»),
    поэтому поиск точно попадает в ячейку, а не в плоский ряд чисел.
    """
    from openpyxl import load_workbook
    stream = _open_stream(path_or_stream)
    try:
        wb = load_workbook(stream, read_only=True, data_only=True)
        parts = []
        for ws in wb.worksheets:
            raw_rows = [
                [_value_to_text(cell) for cell in row]
                for row in ws.iter_rows(values_only=True)
            ]
            # Срезаем хвостовые пустые ячейки каждой строки
            rows = []
            for row in raw_rows:
                while row and not row[-1]:
                    row.pop()
                if row:
                    rows.append(row)
            if not rows:
                continue

            header_idx = _find_header_row(rows)
            header = rows[header_idx] if header_idx is not None else None
            start = header_idx + 1 if header_idx is not None else 0

            sheet_lines = [f"[Лист: {ws.title}]"]
            for row in rows[start:]:
                sheet_lines.append(_serialize_row(row, header))
            parts.append("\n".join(sheet_lines))

        if not parts:
            logger.warning("XLSX: в книге нет заполненных листов")
            return ""
        return "\n\n".join(parts)
    finally:
        _close_stream(stream, path_or_stream)


# --- PPTX ---

def _pptx_text_frame_text(frame) -> str:
    try:
        return (frame.text or "").strip()
    except Exception:
        return ""


def _pptx_table_text(table) -> str:
    lines = []
    for row in table.rows:
        cells = [_pptx_text_frame_text(cell.text_frame) for cell in row.cells]
        cells = [c for c in cells if c]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def read_pptx(path_or_stream) -> str:
    """Текст слайдов: текстовые фреймы и таблицы, с заголовком «# Слайд N»."""
    from pptx import Presentation
    stream = _open_stream(path_or_stream)
    try:
        prs = Presentation(stream)
        parts = []
        for idx, slide in enumerate(prs.slides, start=1):
            slide_parts = [f"# Слайд {idx}"]
            for shape in slide.shapes:
                try:
                    if shape.has_text_frame:
                        text = _pptx_text_frame_text(shape.text_frame)
                        if text:
                            slide_parts.append(text)
                    elif shape.has_table:
                        text = _pptx_table_text(shape.table)
                        if text:
                            slide_parts.append(text)
                except Exception as e:
                    logger.warning("Слайд %d, фигура пропущена: %s", idx, e)
            parts.append("\n".join(slide_parts))
        return "\n\n".join(parts)
    finally:
        _close_stream(stream, path_or_stream)


# --- Диспетчер ---

def read_document(path_or_stream, ext: str | None = None) -> str:
    """Извлекает текст из документа. Расширение определяется по имени файла,
    если не задано явно (удобно при передаче BytesIO: передавайте ext)."""
    if ext is None:
        name = os.path.basename(str(path_or_stream))
        ext = os.path.splitext(name)[1]
    ext = ext.lower()
    if ext == ".txt":
        return read_txt(path_or_stream)
    if ext == ".pdf":
        return read_pdf(path_or_stream)
    if ext == ".docx":
        return read_docx(path_or_stream)
    if ext == ".xlsx":
        return read_xlsx(path_or_stream)
    if ext == ".pptx":
        return read_pptx(path_or_stream)
    raise ValueError(f"Неподдерживаемый формат документа: {ext!r}")