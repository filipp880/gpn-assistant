FROM ollama/ollama:latest

ARG LLM_MODEL=gemma4:e2b-it-qat

# Запекаем LLM в образ при сборке: первый контейнер с этого образа не тянет
# несколько гигабайт из интернета. Пустой том ollama_data наполняется моделью
# из образа автоматически при первом mount, повторные поднятия — почти мгновенны.
# Если LLM_MODEL изменить позже, entrypoint (в compose) докачает новую модель.
RUN set -eux; \
    nohup ollama serve >/tmp/ollama-serve.log 2>&1 & \
    for i in $(seq 1 120); do ollama list >/dev/null 2>&1 && break; sleep 1; done; \
    ollama pull "${LLM_MODEL}"