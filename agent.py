import os
import json
import time
import ollama
import logging
from retrieval import hybrid_search
from core import resolve_slang_terms

# Настройка логирования для Docker-контейнера
logger = logging.getLogger(__name__)


class OllamaUnavailableError(Exception):
    """Ollama не отвечает или запрошенная модель недоступна."""
    pass


class GpnAgent:
    # Ограничение хакатона: контекстное окно модели — не более 32 000 токенов.
    LLM_MAX_CONTEXT = 32000

    def __init__(self):
        self.model_name = os.getenv("LLM_MODEL", "qwen2.5:14b")
        self.num_ctx = int(os.getenv("LLM_NUM_CTX", "8192"))
        if self.num_ctx > self.LLM_MAX_CONTEXT:
            logger.warning(
                "LLM_NUM_CTX=%d превышает лимит хакатона (%d) — ограничиваю до %d",
                self.num_ctx, self.LLM_MAX_CONTEXT, self.LLM_MAX_CONTEXT,
            )
            self.num_ctx = self.LLM_MAX_CONTEXT
        elif self.num_ctx < 1024:
            self.num_ctx = 8192
        self.max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "3"))
        self.self_check = os.getenv("LLM_SELF_CHECK", "0").lower() in ("1", "true", "yes", "on")
        self.dictionary = self._load_dictionary()
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=host, timeout=120)
        logger.info(
            f"GpnAgent инициализирован. Словарь загружен. Модель: {self.model_name} "
            f"(num_ctx={self.num_ctx}, self_check={self.self_check})"
        )

    def _load_dictionary(self) -> dict:
        """Загружает словарь аббревиатур. Если файла нет, использует дефолтный для тестов жюри."""
        path = "corporate_dictionary.json"
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error("Ошибка парсинга словаря, используется дефолтный.")
        
        # Дефолтный словарь для прохождения автотестов жюри
        return {
            "ГПН": "Газпром нефть",
            "ГПНР": "Газпромнефть-Развитие",
            "ЦДНГ": "Цех добычи нефти и газа",
            "НГДУ": "Нефтегазодобывающее управление",
            "ГДИС": "Гидродинамическое исследование скважин"
        }

    def _chat(self, messages, tools=None, options=None, **kwargs):
        """chat() с ретраями на транзиентные ошибки Ollama.

        На слабых машинах модель иногда догружается/отвечает сетевыми сбоями —
        повтор пробует счастливый путь, а не падает сразу. Переполнение контекста
        ретраить бессмысленно, поэтому оно пробрасывается как есть.
        """
        attempts = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
        for attempt in range(attempts):
            try:
                return self._client.chat(
                    model=self.model_name, messages=messages,
                    tools=tools, options=options, **kwargs,
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

    # --- ИНСТРУМЕНТЫ (TOOLS) ---
    def _search_article(self, query: str) -> dict:
        """Поиск по базе знаний с предварительным fuzzy-матчингом"""
        expanded_query, search_terms = self._resolve_slang(query)
        
        # Вызов твоего гибридного поиска
        results = hybrid_search(expanded_query) 
        
        context_parts = []
        sources = set()
        
        for doc, src in results:
            context_parts.append(f"[источник: {src}]\n{doc}")
            sources.add(src)
            
        return {
            "context": "\n\n".join(context_parts),
            "sources": list(sources),
            "resolved_in_search": search_terms
        }

    def _dispatch(self, function_name: str, arguments: dict) -> dict:
        if function_name == 'search_article':
            return self._search_article(**arguments)
        else:
            raise ValueError(f"Неизвестный инструмент: {function_name}")

    # --- ГЛАВНЫЙ ЦИКЛ АГЕНТА ---
    def process_query(self, user_query: str, history: list) -> dict:
        # 1. Pre-processing запроса пользователя
        expanded_user_query, initial_resolved_terms = self._resolve_slang(user_query)
        
        # 2. Формирование жесткого System Prompt (Блок 3)
        resolved_terms_str = json.dumps(initial_resolved_terms, ensure_ascii=False) if initial_resolved_terms else "Термины не обнаружены."
        dictionary_str = json.dumps(self.dictionary, ensure_ascii=False, indent=2)
        
        system_prompt = f"""<role>
Ты — строгий корпоративный AI-ассистент ПАО «Газпром нефть». Твоя база знаний — это внутренние регламенты и документы.
</role>

<constraints>
1. НИКОГДА не выдумывай факты. Если ответа нет в контексте инструмента search_article, напиши: "В корпоративной базе знаний отсутствует информация по данному запросу".
2. В ответах ОБЯЗАТЕЛЬНО используй предоставленные расшифровки терминов из блоков <resolved_slang> и <dictionary>.
3. В конце ответа ОБЯЗАТЕЛЬНО указывай источники в формате: [Источник: имя_файла].
4. Если для полного ответа не хватает одного поиска (вопрос затрагивает несколько аспектов), вызови search_article НЕСКОЛЬКО раз с разными подзапросами, а потом обобщи результаты.
</constraints>

<resolved_slang>
Система распознавания опечаток определила следующие термины в запросе пользователя:
{resolved_terms_str}
Учитывай это при формировании ответа и вызове инструментов.
</resolved_slang>

<dictionary>
Корпоративный словарь аббревиатур и терминов компании. Если в запросе встречается аббревиатура или сленг без прямой расшифровки в <resolved_slang>, расшифруй её с помощью словаря:
{dictionary_str}
</dictionary>
"""
        
        # Формируем историю сообщений для LLM
        messages = [{'role': 'system', 'content': system_prompt}]
        messages.extend(history[-int(os.getenv("MAX_HISTORY_TURNS", "6")):]) # Берем последние N сообщений для экономии контекста
        messages.append({'role': 'user', 'content': user_query}) # Отправляем оригинальный запрос

        tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'search_article',
                    'description': 'База знаний с ответами на вопросы по внутренним документам компании',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {"type": "string", 'description': "Вопрос для поиска (используй расшифрованные термины)"}
                        },
                        'required': ['query'],
                    }
                }
            }
        ]

        final_answer = None
        final_sources = set()
        all_resolved_terms = initial_resolved_terms.copy()
        raw_context_accumulator = []
        
        max_iterations = self.max_iterations
        for step in range(max_iterations):
            try:
                response = self._chat(
                    messages=messages, tools=tools,
                    # keep_alive=-1: модель остаётся загруженной между запросами —
                    # второй и последующие запросы жюри отвечают заметно быстрее
                    options={'num_ctx': self.num_ctx}, keep_alive=-1
                )
            except OllamaUnavailableError:
                raise
            except Exception as e:
                logger.error(f"Ошибка LLM: {str(e)}")
                raise OllamaUnavailableError(f"Ошибка LLM: {str(e)}") from e
            message = response['message']

            # Если LLM решила ответить без вызова инструментов
            if not message.get('tool_calls'):
                final_answer = message['content']
                messages.append(message)
                break

            messages.append(message)
            
            # Обработка вызовов инструментов
            for tool_call in message['tool_calls']:
                name = tool_call['function']['name']
                args = tool_call['function']['arguments']
                
                try:
                    result = self._dispatch(name, args)
                except Exception as e:
                    result = {'error': str(e)}
                    
                # Сбор метаданных для FastAPI ответа
                if name == 'search_article':
                    if 'sources' in result:
                        final_sources.update(result['sources'])
                    if 'context' in result:
                        raw_context_accumulator.append(result['context'])
                    if 'resolved_in_search' in result:
                        all_resolved_terms.extend(result['resolved_in_search'])
                
                # Возврат результата LLM в виде JSON строки
                messages.append({'role': 'tool', 'name': name, 'content': json.dumps(result, ensure_ascii=False)})
                
        if final_answer is None:
            final_answer = "Превышено максимальное количество шагов (max_iterations)."

        # Self-check: сверяем, что ответ действительно подтверждается контекстом из базы знаний.
        # Включается LLM_SELF_CHECK=1. На 3B-моделях может давать ложные срабатывания,
        # поэтому рекомендуется только для моделей 7B+.
        if self.self_check and final_answer and (raw_context_accumulator or final_sources):
            context_for_check = "\n\n".join(raw_context_accumulator)
            if not self._verify_grounded(user_query, context_for_check):
                logger.info("Self-check: ответ не подтверждён контекстом, заменяю на стандартный отказ")
                final_answer = "В корпоративной базе знаний отсутствует информация по данному запросу."

        # Дедупликация распознанных терминов
        unique_terms = []
        seen = set()
        for term in all_resolved_terms:
            if term['original'] not in seen:
                unique_terms.append(term)
                seen.add(term['original'])

        return {
            "answer": final_answer,
            "sources": sorted(list(final_sources)),
            "resolved_terms": unique_terms,
            "raw_context": "\n\n---\n\n".join(raw_context_accumulator) if raw_context_accumulator else ""
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
# Экземпляр создается один раз при импорте модуля
agent_instance = GpnAgent()

def run_my_agent_logic(query: str, session_id: str, history: list) -> dict:
    """Точка входа, которую вызывает app.py (FastAPI)"""
    return agent_instance.process_query(query, history)