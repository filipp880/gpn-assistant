#!/usr/bin/env bash
# Сбор массива реальных ответов /chat в docs/screenshots/ для отчёта.
# Требуется: запущенный сервис (docker compose up --build).
set -euo pipefail
Base="http://localhost:8000"
OutDir="$(cd "$(dirname "$0")/.." && pwd)/docs/screenshots"
mkdir -p "$OutDir"

queries=(
  "slang|Какие показатели учитываются в отчёте по ГПНР?"
  "typo|КПЕ по ремонту скважин в 2025 году"
  "industry|Что такое ГРП и как ПНГ влияет на экологию?"
  "kb|Что такое RAG и зачем он нужен?"
  "refusal|Какой курс акций Газпром нефти завтра?"
)

: > "$OutDir/chat_log.ndjson"
for item in "${queries[@]}"; do
  note="${item%%|*}"
  q="${item#*|}"
  body="{\"query\":\"$q\"}"
  start=$(date +%s%3N)
  resp=$(curl -sf -X POST "$Base/chat" -H 'Content-Type: application/json' -d "$body") || {
    echo "{\"query\":\"$q\",\"note\":\"$note\",\"error\":\"curl failed\"}" >> "$OutDir/chat_log.ndjson"
    continue
  }
  end=$(date +%s%3N)
  printf '%s' "{\"query\":\"$q\",\"note\":\"$note\",\"latency_ms\":$((end-start)),\"response\":$resp}" >> "$OutDir/chat_log.ndjson"
  echo >> "$OutDir/chat_log.ndjson"
  echo "[$note] $q -> $((end-start)) ms"
done
echo "Логи сохранены в $OutDir/chat_log.ndjson"