#!/usr/bin/env bash
# Демо-прогон всех сценариев из docs/demo_cases.md
# Требуется: запущенный сервис (docker compose up --build), jq опционален.
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"

pretty() {
  if command -v jq >/dev/null 2>&1; then jq .; else cat; fi
}

echo "== 1. Сленг: ГПНР =="
curl -s -X POST "$BASE/chat" -H "Content-Type: application/json" \
  -d '{"query": "Какие показатели учитываются в отчёте по ГПНР?"}' | pretty

echo "== 2. Опечатка в аббревиатуре: КПЕ -> КПЭ =="
curl -s -X POST "$BASE/chat" -H "Content-Type: application/json" \
  -d '{"query": "КПЕ по ремонту скважин в 2025 году"}' | pretty

echo "== 3. Отраслевые аббревиатуры: ГРП, ПНГ =="
curl -s -X POST "$BASE/chat" -H "Content-Type: application/json" \
  -d '{"query": "Что такое ГРП и как ПНГ влияет на экологию?"}' | pretty

echo "== 4. Гибридный поиск по базе знаний =="
curl -s -X POST "$BASE/chat" -H "Content-Type: application/json" \
  -d '{"query": "Что такое RAG и зачем он нужен?"}' | pretty

echo "== 5. Регламентированный отказ =="
curl -s -X POST "$BASE/chat" -H "Content-Type: application/json" \
  -d '{"query": "Какой курс акций Газпром нефти завтра?"}' | pretty

echo "== 6. Мульти-терн =="
SESSION=$(curl -s -X POST "$BASE/chat" -H "Content-Type: application/json" \
  -d '{"query": "Что такое KPI нефтяной компании?"}' | sed -n 's/.*"session_id":"\([^"]*\)".*/\1/p')
echo "session_id=$SESSION"
curl -s -X POST "$BASE/chat" -H "Content-Type: application/json" \
  -d "{\"query\": \"Разверни подробнее?\", \"session_id\": \"$SESSION\"}" | pretty
curl -s "$BASE/history/$SESSION" | pretty

echo "== 7. Health =="
curl -s "$BASE/health" | pretty

echo "Готово."