# Демо-прогон всех сценариев из docs/demo_cases.md для Windows PowerShell
# Требуется: запущенный сервис (docker compose up --build).
$Base = "http://localhost:8000"

function Post-Chat([string]$Query, [string]$SessionId) {
    $body = @{ query = $Query }
    if ($SessionId) { $body.session_id = $SessionId }
    Invoke-RestMethod -Uri "$Base/chat" -Method Post -ContentType "application/json" `
        -Body ($body | ConvertTo-Json -Compress) | ConvertTo-Json -Depth 5
}

Write-Host "== 1. Сленг: ГПНР =="
Post-Chat "Какие показатели учитываются в отчёте по ГПНР?"

Write-Host "== 2. Опечатка в аббревиатуре: КПЕ -> КПЭ =="
Post-Chat "КПЕ по ремонту скважин в 2025 году"

Write-Host "== 3. Отраслевые аббревиатуры: ГРП, ПНГ =="
Post-Chat "Что такое ГРП и как ПНГ влияет на экологию?"

Write-Host "== 4. Гибридный поиск по базе знаний =="
Post-Chat "Что такое RAG и зачем он нужен?"

Write-Host "== 5. Регламентированный отказ =="
Post-Chat "Какой курс акций Газпром нефти завтра?"

Write-Host "== 6. Мульти-терн =="
$r = Invoke-RestMethod -Uri "$Base/chat" -Method Post -ContentType "application/json" `
    -Body '{"query":"Что такое KPI нефтяной компании?"}'
Write-Host "session_id=$($r.session_id)"
Post-Chat "Разверни подробнее?" $r.session_id
Invoke-RestMethod -Uri "$Base/history/$($r.session_id)" | ConvertTo-Json -Depth 5

Write-Host "== 7. Health =="
Invoke-RestMethod -Uri "$Base/health" | ConvertTo-Json

Write-Host "Готово."