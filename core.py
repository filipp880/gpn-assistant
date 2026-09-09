"""Чистые функции проекта: чанкинг, лемматизация, RRF-фьюжн, распознавание сленга.

Модуль не импортирует тяжёлые ML-зависимости (torch, ChromaDB и т.п.),
поэтому его можно юнит-тестировать без скачивания моделей.
"""

import math
import re
from collections import Counter

import pymorphy3
from rapidfuzz import process, fuzz

_SEP_RE = re.compile(r"[^А-Яа-яЁёA-Za-z0-9]+")


# --- Dynamic top_k ---
_MIN_TOP_K = 2
_MAX_TOP_K = 8
_complexity_re = re.compile(
    r"(сравни|сравнение|разниц|отличи|характеристик|описani|список|перечень"
    r"|какие|каких|какой|какого|какова|сколько|когда|где|почему|зачем"
    r"|расскажи|опиши|приведи|назови|в чем|что такое|определ)",
    re.IGNORECASE,
)


def estimate_top_k(query: str, min_k: int = _MIN_TOP_K, max_k: int = _MAX_TOP_K) -> int:
    """Адаптивный top_k в зависимости от сложности запроса.

    Простые фактоидные запросы ("что такое EBITDA") → min_k (2-3).
    Сложные многоаспектные ("сравни характеристики ГПА-16 и ГПА-25") → max_k (7-8).

    Критерии:
    - Длина запроса (количество слов)
    - Наличие вопросительных/сравнительных конструкций
    - Количество сущностей (капс/аббревиатур)
    """
    words = query.split()
    word_count = len(words)

    # Базовый score по длине
    if word_count <= 3:
        score = 0.2
    elif word_count <= 6:
        score = 0.4
    elif word_count <= 10:
        score = 0.6
    else:
        score = 0.8

    # Многоаспектные запросы (сравнение, списки, несколько вопросов)
    if _complexity_re.search(query):
        score += 0.15

    # Несколько аббревиатур/сущностей (капс-слова = корпоративные термины)
    caps_count = sum(1 for w in words if w.isupper() and len(w) >= 2)
    if caps_count >= 2:
        score += 0.15
    elif caps_count >= 1:
        score += 0.05

    # Перечисления через запятую/точку с запятой
    if re.search(r"[,;]", query):
        score += 0.1

    score = max(0.0, min(1.0, score))
    return min_k + int(score * (max_k - min_k))

# Видимые омонимы латиница/кириллица (для ошибочно набранных аббревиатур «ГПH», «ГАЗПРОМ»).
# Срабатывают только на ВЕРХНЕМ регистре. Намеренно без I/L/D/R — они дают ложно-положительные
# совпадения на обычных латинских словах (IT, RAG, COVID и т.п.).
_HOMOGLYPH = {
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н",
    "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т",
    "X": "Х", "Y": "У",
    "Ё": "Е",
}
_HOMOGLYPH_TABLE = str.maketrans(_HOMOGLYPH)


def chunk_with_overlap(text: str, chunk_size: int = 500, overlap: int = 150) -> list[str]:
    """Разбивает текст на чанки фиксированного размера с перекрытием."""
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def adaptive_chunk(text: str, max_chunk: int = 500) -> list[str]:
    """Адаптивный чанкинг: режет по предложениям и абзацам, а не по символам.

    Нормальное предложение никогда не разбивается; граница абзаца — естественный
    разрыв. Предложения длиннее max_chunk режутся жёстко.

    Заголовки разделов (markdown `## ...` или короткая строка-название, стоящая
    своим абзацем) не теряются: заголовок приклеивается к первому чанку своего
    раздела, так что dense-поиск и реранкер видят контекст раздела целиком.
    """
    chunks = []
    current = []
    cur_len = 0
    heading = None

    def flush():
        nonlocal current, cur_len
        if current:
            body = " ".join(current)
            chunks.append(f"{heading}\n{body}" if heading else body)
            current, cur_len = [], 0

    lines = text.split("\n")
    for i, para in enumerate(lines):
        stripped = para.strip()
        if not stripped:
            continue
        next_blank = (i + 1 >= len(lines)) or not lines[i + 1].strip()
        new_heading = _heading_of(stripped, standalone=next_blank)
        if new_heading is not None:
            flush()
            heading = new_heading
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", para) if s.strip()]
        for sent in sentences:
            if len(sent) > max_chunk:
                flush()
                for j in range(0, len(sent), max_chunk):
                    part = sent[j : j + max_chunk].strip()
                    if part:
                        chunks.append(f"{heading}\n{part}" if heading else part)
                continue
            if current and cur_len + len(sent) + 1 > max_chunk:
                flush()
            current.append(sent)
            cur_len += len(sent) + 1
        flush()

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c]


def parent_child_chunk(
    text: str, parent_size: int = 2000, child_size: int = 500, overlap: int = 100,
) -> list[tuple[str, list[str]]]:
    """Parent-Child чанкинг: большие родительские чанки с мелкими дочерними.

    Каждый parent разбивается на overlapping children для индексации.
    Поиск идёт по children, но LLM видит parent целиком.

    Возвращает: [(parent_text, [child_1, child_2, ...]), ...]
    """
    # Step 1: разбить документ на parent chunks (по предложениям/абзацам)
    raw_parents = adaptive_chunk(text, max_chunk=parent_size)

    result = []
    for parent_text in raw_parents:
        if len(parent_text) <= child_size:
            # Parent足够小 — он сам является единственным child
            result.append((parent_text, [parent_text]))
            continue

        # Step 2: разбить parent на overlapping children
        children = []
        step = child_size - overlap
        for i in range(0, len(parent_text), step):
            child = parent_text[i:i + child_size].strip()
            if child:
                children.append(child)
        result.append((parent_text, children))

    return result


def _heading_of(line: str, standalone: bool = False) -> str | None:
    """Определяет, является ли строка заголовком раздела.

    Правила: markdown-заголовок (`# ...`) — всегда; текстовая строка — только
    если она короткая, без знаков конца предложения и стоит отдельным абзацем
    (следующая строка пустая или конец документа).
    """
    s = line.strip()
    if not s or len(s) > 80:
        return None
    if not re.search(r"[а-яА-ЯёЁa-zA-Z]", s):
        return None
    md = re.match(r"^#{1,6}\s+(.+)", s)
    if md:
        return md.group(1).strip()
    if not standalone:
        return None
    if s.endswith(('.', '!', '?', ':')):
        return None
    return s


_morph = pymorphy3.MorphAnalyzer()


def tokenize_lemmas(text: str) -> list[str]:
    """Токенизация с лемматизацией (pymorphy3)."""
    words = re.findall(r"[\wа-яё]+", text.lower())
    return [_morph.normal_forms(w)[0] for w in words]


class OkapiBM25:
    """Okapi BM25 на лемматизированных токенах (pymorphy3).

    Чистый Python, без внешних зависимостей — индекс строится в памяти при
    загрузке хранилища. Это третий канал гибридного поиска: точная лексика
    словоформ, дополняет learned-sparse от BGE-M3 (SPLADE-подобный).
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.doc_tf: list[Counter] = []
        df: Counter = Counter()
        total_len = 0
        for doc in corpus:
            tf = Counter(tokenize_lemmas(doc))
            self.doc_tf.append(tf)
            total_len += sum(tf.values())
            df.update(tf.keys())
        self.avgdl = total_len / self.N if self.N else 0.0
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for t, n in df.items()
        }

    def get_scores(self, query: str) -> list[float]:
        """Скоринг всех документов корпуса под запрос (леммы)."""
        q = Counter(tokenize_lemmas(query))
        out = []
        for tf in self.doc_tf:
            dl = sum(tf.values())
            norm = dl / self.avgdl if self.avgdl else 0.0
            score = 0.0
            for t, qtf in q.items():
                f = tf.get(t, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * norm)
                score += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / denom
            out.append(score)
        return out


def rrf_fusion(rankings: list[list[str]], k: int = 60, top_k: int = 5) -> list[str]:
    """Reciprocal Rank Fusion: объединение списков ранжированных документов."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:top_k]


def weighted_rrf_fusion(
    rankings: list[list[str]],
    weights: list[float],
    k: int = 60,
    top_k: int = 5,
) -> list[str]:
    """Взвешенный Reciprocal Rank Fusion: RRF, но каждый трекер с весом.

    Позволяет дать семантическому каналу больший/меньший приоритет, чем
    лексическому. При равных весах вырождается в обычный RRF.
    """
    if len(weights) != len(rankings):
        weights = [1.0] * len(rankings)
    scores = {}
    for weight, ranking in zip(weights, rankings):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:top_k]


def normalize_token(token: str) -> str:
    """Приводит токен к каноническому виду для сравнения со словарём.

    1) Убирает любые разделители/скобки: "ГПН-Р" -> "ГПНР", "ebitda." -> "EBITDA".
    2) Верхний регистр.
    3) Заменяет визуальные омонимы латиница/кириллица: "ГПH" (латинская H) -> "ГПН",
       "ВАЗ" с латинской A -> "ВАЗ".
    """
    upper = _SEP_RE.sub("", token).upper()
    return upper.translate(_HOMOGLYPH_TABLE)


def resolve_slang_terms(query: str, dictionary: dict, threshold: int = 80) -> tuple[str, list]:
    """Нечеткий поиск опечаток и сленга во фразе.

    Возвращает (расширенный_запрос, список распознанных терминов).

    Правила:
    - Токен, уже содержащий "(", не трогаем (он уже обогащён расшифровкой).
    - Токен из 1 символа, с цифрами или чисто цифровой пропускаем.
    - «Аббревиатурой» считается токен без строчных букв (ГПНР, EBITDA). Для неё
      порог снижается до 60% — иначе одиночная опечатка в коротком акрониме
      (75% сходства для 4 букв и 67% для 3 букв) не была бы распознана.
      Для обычных слов порог прежний — 80%.
    - Токен аннотируется, если он *не равен введённому* посимвольно слову словаря:
      так «ГПНР» (точное написание) не шумит, а «гпнр», «ГПН-Р», «КПЕ» распознаются.
    """
    if not dictionary:
        return query, []

    # Ключи словаря в канонической (нормализованной) форме -> исходный ключ
    norm_keys = {normalize_token(key): key for key in dictionary}
    norm_key_list = list(norm_keys)

    resolved_terms = []
    tokens = query.split()

    for i, token in enumerate(tokens):
        if "(" in token:
            continue
        clean_token = normalize_token(token)
        if len(clean_token) < 2 or not clean_token.isalpha():
            continue

        # Сравнение с ключом по символам исходного токена (без краевых знаков)
        raw = token.strip(".,!?;:()")
        # Абревиатура в оригинале набрана без строчных букв -> мягкий порог
        is_abbrev = not any(ch.islower() for ch in raw)
        thr = 60 if is_abbrev else threshold

        match_result = process.extractOne(clean_token, norm_key_list, scorer=fuzz.ratio)
        if not match_result:
            continue
        norm_match, score, _ = match_result
        original_key = norm_keys[norm_match]
        if score > thr and original_key != raw:
            canonical = dictionary[original_key]
            resolved_terms.append({
                "original": token,
                "canonical": canonical,
                "score": float(score)
            })
            # Обогащаем запрос для поиска и LLM, если каноничной формы ещё нет
            if canonical.lower() not in query.lower():
                tokens[i] = f"{token} ({canonical})"

    expanded = " ".join(tokens)

    # --- Обратная карта: расшифровка -> аббревиатура ---
    # Пользователь пишет «заправочная станция», а в документах только «АЗС».
    # Прямая карта (аббревиатура -> расшифровка) тут не сработает, поэтому ищем
    # расшифровки в запросе по леммам и дописываем аббревиатуру: «... (АЗС)».
    # Лемматизация покрывает «заправочной станции -> заправочная станция».
    query_lemmas = {normalize_token(t) for t in tokenize_lemmas(query) if t.isalpha() and len(t) >= 3}
    reverse_notes = []
    seen_reverse = set()
    for key, value in dictionary.items():
        value_lemmas = {normalize_token(t) for t in tokenize_lemmas(value) if t.isalpha() and len(t) >= 3}
        if len(value_lemmas) < 1:
            continue
        # Синонимичная фраза распознана, только если в запросе есть ВСЕ её значимые слова
        if not value_lemmas.issubset(query_lemmas):
            continue
        if normalize_token(key) in query_lemmas:  # аббревиатура уже упомянута пользователем
            continue
        if key in seen_reverse or f"({key}" in expanded:
            continue
        seen_reverse.add(key)
        reverse_notes.append(key)
        resolved_terms.append({
            "original": value,
            "canonical": key,
            "score": 100.0
        })

    if reverse_notes:
        expanded = f"{expanded} ({' '.join(reverse_notes)})"

    return expanded, resolved_terms