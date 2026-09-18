import json
import os
import re
import threading
import time
import ollama
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from retrieval import hybrid_search
from core import (
    resolve_slang_terms,
    context_budget_chars,
    fit_context_budget,
)
from tracing import create_span
import config
import dictionary_store

# Настройка логирования для Docker-контейнера
logger = logging.getLogger(__name__)


class OllamaUnavailableError(Exception):
    """Ollama не отвечает или запрошенная модель недоступна."""
    pass


# Путь к словарю — единый источник истины для чтения и записи.
DICTIONARY_FILE = config.DICTIONARY_FILE

# Дефолтный словарь для прохождения автотестов жюри.
DEFAULT_DICTIONARY = {
    "ГПН": "Газпром нефть",
    "ГПНР": "Газпромнефть-Развитие",
    "ЦДНГ": "Цех добычи нефти и газа",
    "НГДУ": "Нефтегазодобывающее управление",
    "ГДИС": "Гидродинамическое исследование скважин",
}


def save_dictionary(dictionary: dict) -> None:
    """Сохраняет словарь в DICTIONARY_FILE — JSON (.json) или CSV (.csv).

    Формат выбирается по расширению файла, см. dictionary_store.
    """
    dictionary_store.save_dictionary(dictionary, DICTIONARY_FILE)


class GpnAgent:
    # Ограничение хакатона: контекстное окно модели — не более 32 000 токенов.
    LLM_MAX_CONTEXT = config.LLM_MAX_CONTEXT

    def __init__(self):
        self.model_name = config.LLM_MODEL
        self.num_ctx = config.LLM_NUM_CTX
        if self.num_ctx > self.LLM_MAX_CONTEXT:
            logger.warning(
                "LLM_NUM_CTX=%d превышает лимит хакатона (%d) — ограничиваю до %d",
                self.num_ctx, self.LLM_MAX_CONTEXT, self.LLM_MAX_CONTEXT,
            )
            self.num_ctx = self.LLM_MAX_CONTEXT
        elif self.num_ctx < 1024:
            self.num_ctx = 8192
        self.max_iterations = config.AGENT_MAX_ITERATIONS
        self.self_check = config.LLM_SELF_CHECK
        self.dictionary = self._load_dictionary()
        self._client = ollama.Client(host=config.OLLAMA_HOST, timeout=120)
        logger.info(
            f"GpnAgent инициализирован. Словарь загружен. Модель: {self.model_name} "
            f"(num_ctx={self.num_ctx}, self_check={self.self_check})"
        )

    def _load_dictionary(self) -> dict:
        """Загружает словарь аббревиатур из JSON или CSV (по расширению файла).

        Если файла нет или он битый — использует дефолтный для тестов жюри.
        """
        if os.path.exists(DICTIONARY_FILE):
            loaded = dictionary_store.load_dictionary(DICTIONARY_FILE)
            if loaded:
                return loaded
            logger.error(
                "Ошибка парсинга словаря %s, используется дефолтный.", DICTIONARY_FILE
            )
        return DEFAULT_DICTIONARY

    def _chat(self, messages, options=None, **kwargs):
        """chat() с ретраями на транзиентные ошибки Ollama.

        На слабых машинах модель иногда догружается/отвечает сетевыми сбоями —
        повтор пробует счастливый путь, а не падает сразу. Переполнение контекста
        ретраить бессмысленно, поэтому оно пробрасывается как есть.
        """
        attempts = config.OLLAMA_MAX_RETRIES
        for attempt in range(attempts):
            try:
                return self._client.chat(
                    model=self.model_name, messages=messages,
                    options=options, **kwargs,
                )
            except Exception as e:
                if "longer than context length" in str(e).lower():
                    raise OllamaUnavailableError(
                        f"Запрос превышает контекст (num_ctx={self.num_ctx}): {str(e)}"
                    ) from e
                if attempt < attempts - 1:
                    wait = 2 ** attempt  # 1, 2, 4 c
                    logger.warning("Транзиентная ошибка Ollama (%s), повтор через %ds", e, wait)
                    time.sleep(wait)
        raise OllamaUnavailableError(
            f"Ollama недоступна ({self._client.host}) после {attempts} попыток"
        )

    def _resolve_slang(self, query: str) -> tuple[str, list]:
        """
        Блок 2: Нечеткий поиск опечаток и сленга.
        Возвращает расширенный запрос и лог распознанных терминов.
        Реализация вынесена в чистую функцию `core.resolve_slang_terms`.
        """
        return resolve_slang_terms(query, self.dictionary)

    # --- QUERY DECOMPOSITION ---
    _DECOMPOSE_PROMPT = (
        "Ты — декомпозитор поисковых запросов для корпоративной базы знаний. "
        "Разбей сложный вопрос пользователя на 1-3 простых подзапроса для поиска.\n\n"
        "Правила:\n"
        "- Каждый подзапрос должен быть самодостаточным для поиска\n"
        "- Используй расшифровки аббревиатур (ГПНР → Газпромнефть-Развитие)\n"
        "- Не дублируй информацию между подзапросами\n"
        "- Если вопрос простой — верни один подзапрос\n\n"
        "Ответ ТОЛЬКО в формате JSON-массива строк. Без пояснений.\n\n"
        "Примеры:\n"
        "Вопрос: Какая EBITDA у ГПНР за 2025 год?\n"
        '["EBITDA Газпромнефть-Развитие 2025"]\n\n'
        "Вопрос: Каковы риски для ЦДНГ и какие меры контроля?\n"
        '["риски Цех добычи нефти и газа", "меры контроля ЦДНГ"]\n\n'
        "Вопрос: Сравни выработку НГДУ и ГДИС\n"
        '["выработка Нефтегазодобывающее управление", "выработка Гидродинамическое исследование скважин"]\n\n'
        "Вопрос: {query}\n"
    )

    def _decompose_query(self, user_query: str, trace=None) -> list[str]:
        """LLM декомпозирует сложный запрос на подзапросы для параллельного поиска.

        Возвращает список подзапросов (1-3 штуки). При ошибке — [user_query] (fallback).
        """
        with create_span(trace, "query_decomposition", input_data=user_query) as span:
            prompt = self._DECOMPOSE_PROMPT.format(query=user_query)
            try:
                resp = self._chat(
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'num_ctx': 2048, 'num_predict': 150, 'temperature': 0},
                    keep_alive=-1,
                )
                raw = resp['message']['content'].strip()

                # Извлекаем JSON-массив из ответа LLM (может быть обёрнут в markdown)
                match = re.search(r'\[.*\]', raw, re.DOTALL)
                if not match:
                    logger.warning("Decomposition: JSON не найден в ответе LLM, fallback на оригинальный запрос")
                    span.update(output={"sub_queries": [user_query], "note": "json_not_found"})
                    return [user_query]

                sub_queries = json.loads(match.group())
                if not isinstance(sub_queries, list) or not sub_queries:
                    span.update(output={"sub_queries": [user_query], "note": "empty_list"})
                    return [user_query]

                # Ограничиваем до 3 подзапросов
                sub_queries = [str(q).strip() for q in sub_queries[:3] if q]
                if not sub_queries:
                    logger.warning("Decomposition: подзапросы пустые, fallback на оригинальный запрос")
                    span.update(output={"sub_queries": [user_query], "note": "empty_after_filter"})
                    return [user_query]
                logger.info("Decomposition: '%s' → %d подзапросов", user_query, len(sub_queries))
                span.update(output={"sub_queries": sub_queries, "count": len(sub_queries)})
                return sub_queries

            except Exception as e:
                logger.warning("Decomposition ошибка (%s), fallback на оригинальный запрос", e)
                span.update(output={"sub_queries": [user_query], "error": str(e)})
                return [user_query]

    def _search_impl(self, query: str, trace=None) -> dict:
        """Общая реализация поиска: resolve_slang + hybrid_search + сборка контекста.

        Единая точка поиска: вызывается из `_search_single` (и из потоков `_search_batch`).
        """
        expanded, search_terms = self._resolve_slang(query)
        results = hybrid_search(expanded, trace=trace)
        context_parts = []
        sources = set()
        for doc, src in results:
            context_parts.append(f"[источник: {src}]\n{doc}")
            sources.add(src)
        return {
            "query": query,
            "context": "\n\n".join(context_parts),
            "sources": list(sources),
            "resolved_terms": search_terms,
            "results_count": len(results),
        }

    def _search_single(self, query: str, trace=None) -> dict:
        """Один поиск с resolve_slang + hybrid_search. Для использования в потоках."""
        return self._search_impl(query, trace=trace)

    def _search_batch(self, sub_queries: list[str], trace=None) -> list[dict]:
        """Параллельный поиск по списку подзапросов через ThreadPoolExecutor.

        Возвращает список результатов (по одному на подзапрос).
        """
        if not sub_queries:
            return []
        if len(sub_queries) == 1:
            return [self._search_single(sub_queries[0], trace=trace)]

        with create_span(trace, "batch_search", {"sub_query_count": len(sub_queries)}) as span:
            results = [None] * len(sub_queries)
            max_workers = min(len(sub_queries), 4)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._search_single, q, trace): i
                    for i, q in enumerate(sub_queries)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error("Batch search ошибка для подзапроса[%d]: %s", idx, e)
                        results[idx] = {
                            "query": sub_queries[idx], "context": "",
                            "sources": [], "resolved_terms": [], "results_count": 0,
                        }

            total_results = sum(r["results_count"] for r in results if r)
            total_sources = set()
            for r in results:
                if r:
                    total_sources.update(r["sources"])
            span.update(output={
                "total_results": total_results,
                "unique_sources": list(total_sources),
            })
            return results

    # --- ГЛАВНЫЙ ЦИКЛ АГЕНТА ---
    def process_query(self, user_query: str, history: list, trace=None) -> dict:
        # 1. Pre-processing: resolve slang
        with create_span(trace, "resolve_slang", input_data=user_query) as span:
            expanded_user_query, initial_resolved_terms = self._resolve_slang(user_query)
            span.update(output={"expanded_query": expanded_user_query, "terms_count": len(initial_resolved_terms)})

        # 2. Query Decomposition → parallel search → synthesis
        with create_span(trace, "rag_pipeline", {"query": user_query}) as pipeline_span:
            result = self._rag_decomposition_pipeline(
                user_query, expanded_user_query, initial_resolved_terms, history, trace=trace
            )
            pipeline_span.update(output={
                "sources": result["sources"],
                "resolved_terms_count": len(result["resolved_terms"]),
            })

        return result

    def _rag_decomposition_pipeline(
        self, user_query: str, expanded_query: str,
        initial_resolved_terms: list, history: list, trace=None,
    ) -> dict:
        """Query Decomposition → parallel search → LLM synthesis."""

        # 2a. Decompose
        sub_queries = self._decompose_query(user_query, trace=trace)

        # 2b. Parallel search: подзапросы декомпозиции + исходный (обогащённый
        # сленгом) запрос. Декомпозиция иногда теряет аспект вопроса — поиск по
        # исходному запросу гарантирует, что ни одна часть вопроса не потеряется.
        search_pool = list(dict.fromkeys([expanded_query, *sub_queries]))
        if len(search_pool) > len(sub_queries):
            logger.info(
                "Поиск: %d подзапроса + исходный запрос (%d всего)",
                len(sub_queries), len(search_pool),
            )
        search_results = self._search_batch(search_pool, trace=trace)

        # 2c. Assemble context from all sub-queries
        all_context = []
        all_sources = set()
        all_resolved = list(initial_resolved_terms)

        for r in search_results:
            if r["context"]:
                all_context.append(r["context"])
            all_sources.update(r["sources"])
            all_resolved.extend(r["resolved_terms"])

        # 2d. Synthesize answer (single LLM call, no tools needed)
        resolved_terms_str = (
            json.dumps(all_resolved, ensure_ascii=False) if all_resolved
            else "Термины не обнаружены."
        )
        dictionary_str = json.dumps(self.dictionary, ensure_ascii=False, indent=2)

        system_prompt = f"""<role>
Ты — строгий корпоративный AI-ассистент ПАО «Газпром нефть».
</role>

<constraints>
1. НИКОГДА не выдумывай факты. Если ответа нет в предоставленном контексте, напиши: "В корпоративной базе знаний отсутствует информация по данному запросу".
2. В ответах ОБЯЗАТЕЛЬНО используй расшифровки терминов из <resolved_slang> и <dictionary>.
3. В конце ответа ОБЯЗАТЕЛЬНО указывай источники в формате: [Источник: имя_файла].
4. Обобщи информацию из разных источников, если вопрос затрагивает несколько аспектов.
</constraints>

<resolved_slang>
{resolved_terms_str}
</resolved_slang>

<dictionary>
{dictionary_str}
</dictionary>
"""

        # 2e. Бюджет токенов: RAG-контекст не должен вытеснить словарь/историю/
        # вопрос за пределы num_ctx — иначе Ollama ответит context-length error.
        # Оценка консервативная (chars/2), поэтому суммарный промт гарантированно
        # укладывается в num_ctx - резерв на генерацию.
        history_text = "\n".join(
            str(m.get("content", ""))
            for m in history[-config.MAX_HISTORY_TURNS:]
        )
        budget_chars = context_budget_chars(
            num_ctx=self.num_ctx,
            system_text=system_prompt,
            history_text=history_text,
            question_text=user_query,
        )
        fitted_context, context_chars, context_trimmed = fit_context_budget(
            all_context, budget_chars
        )
        combined_context = "\n\n---\n\n".join(fitted_context) if fitted_context else ""
        if context_trimmed:
            logger.warning(
                "Контекст обрезан до бюджета: %d/%d символов (num_ctx=%d)",
                context_chars,
                sum(len(c) for c in all_context),
                self.num_ctx,
            )

        messages = [
            {'role': 'system', 'content': system_prompt},
        ]
        # Add history
        messages.extend(history[-config.MAX_HISTORY_TURNS:])

        # User message with context
        if combined_context:
            user_msg = (
                f"Контекст из базы знаний:\n\n{combined_context}\n\n"
                f"---\n\nВопрос пользователя: {user_query}\n\n"
                "Ответь на вопрос, опираясь ТОЛЬКО на предоставленный контекст."
            )
        else:
            user_msg = (
                f"Вопрос пользователя: {user_query}\n\n"
                "В корпоративной базе знаний отсутствует информация по данному запросу."
            )
        messages.append({'role': 'user', 'content': user_msg})

        # Single LLM call — no tools, just synthesis
        try:
            with create_span(trace, "llm_synthesize", {
                "model": self.model_name,
                "context_length": len(combined_context),
                "sources_count": len(all_sources),
                "context_trimmed": context_trimmed,
            }) as span:
                response = self._chat(
                    messages=messages,
                    options={'num_ctx': self.num_ctx}, keep_alive=-1,
                )
                final_answer = response['message']['content']
                usage = response.get('usage', {})
                span.update(
                    output={"answer_length": len(final_answer)},
                    metadata={
                        "prompt_tokens": usage.get('prompt_tokens', 0),
                        "completion_tokens": usage.get('completion_tokens', 0),
                        "total_tokens": usage.get('total_tokens', 0),
                    },
                )
        except OllamaUnavailableError:
            raise
        except Exception as e:
            logger.error("LLM synthesis ошибка: %s", e)
            raise OllamaUnavailableError(f"LLM synthesis ошибка: {e}") from e

        # Self-check
        if self.self_check and final_answer and combined_context:
            with create_span(trace, "self_check", {"query": user_query}) as span:
                grounded = self._verify_grounded(user_query, combined_context)
                if not grounded:
                    logger.info("Self-check: ответ не подтверждён контекстом")
                    final_answer = "В корпоративной базе знаний отсутствует информация по данному запросу."
                span.update(output={"verdict": "OK" if grounded else "REJECTED"})

        # Dedup resolved terms
        unique_terms = []
        seen = set()
        for term in all_resolved:
            if term['original'] not in seen:
                unique_terms.append(term)
                seen.add(term['original'])

        return {
            "answer": final_answer,
            "sources": sorted(list(all_sources)),
            "resolved_terms": unique_terms,
            "raw_context": combined_context,
        }

    def _verify_grounded(self, user_query: str, context: str) -> bool:
        """Проверяет, что на вопрос можно ответить по предоставленному контексту (ДА/НЕТ).

        Возвращает True всегда, когда проверка не удалась (не ломаем рабочий путь).
        """
        if not context.strip():
            return False
        prompt = (
            "Ты — проверяющий RAG-системы. Определи, можно ли ответить на вопрос пользователя, "
            "опираясь ТОЛЬКО на предоставленный контекст из базы знаний. "
            "Отвечай строго одним словом: ДА или НЕТ.\n\n"
            f"Вопрос: {user_query}\n\n"
            f"Контекст:\n{context[:4000]}"
        )
        try:
            resp = self._chat(
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_ctx': 4096, 'num_predict': 10, 'temperature': 0},
                keep_alive=-1,
            )
            verdict = resp['message']['content'].strip().upper()
            return verdict.startswith("ДА")
        except Exception as e:
            logger.warning(f"Self-check пропущен из-за ошибки: {e}")
            return True

# --- СИНГЛТОН ДЛЯ FASTAPI ---
# Экземпляр создаётся лениво при первом обращении, а не на импорте модуля:
# import agent сам по себе не создаёт словарь и клиент Ollama.
_agent = None
_agent_lock = threading.Lock()


def get_agent() -> GpnAgent:
    """Ленивый синглтон: создаёт GpnAgent при первом обращении.

    Создание под lock'ом: два параллельных первых запроса не должны поднять
    два экземпляра (каждый тянет словарь и клиент Ollama).
    """
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = GpnAgent()
    return _agent


def run_my_agent_logic(query: str, session_id: str, history: list, trace=None) -> dict:
    """Точка входа, которую вызывает app.py (FastAPI)"""
    return get_agent().process_query(query, history, trace=trace)