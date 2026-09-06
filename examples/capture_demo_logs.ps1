# Сбор массива реальных ответов /chat в docs/screenshots/ для отчёта.
# Требуется: запущенный сервис (docker compose up --build).
# Примеры оформления запросов из docs/demo_cases.md — см. run_demo.ps1.
$Base = "http://localhost:8000"
$OutDir = Join-Path $PSScriptRoot "..\docs\screenshots"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$queries = @(
    @{ q = "Какие показатели учитываются в отчёте по ГПНР?"; note = "slang"},
    @{ q = "КПЕ по ремонту скважин в 2025 году";           note = "typo"},
    @{ q = "Что такое ГРП и как ПНГ влияет на экологию?"; note = "industry"},
    @{ q = "Что такое RAG и зачем он нужен?";             note = "kb"},
    @{ q = "Какой курс акций Газпром нефти завтра?";      note = "refusal"}
)

$data = @()
foreach ($item in $queries) {
    $body = @{ query = $item.q } | ConvertTo-Json -Compress
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = Invoke-RestMethod -Uri "$Base/chat" -Method Post `
            -ContentType "application/json" -Body $body
        $sw.Stop()
        $data += [ordered]@{
            query      = $item.q
            note       = $item.note
            latency_ms = $sw.ElapsedMilliseconds
            answer     = $resp.answer
            sources    = $resp.sources
            resolved_terms = $resp.resolved_terms
        }
    } catch {
        $sw.Stop()
        $data += [ordered]@{
            query      = $item.q
            note       = $item.note
            latency_ms = $sw.ElapsedMilliseconds
            error      = $_.Exception.Message
        }
    }
    Write-Host "[$($item.note)] $($item.q) -> $($sw.ElapsedMilliseconds) ms"
}

$data | ConvertTo-Json -Depth 8 | Set-Content -Path (Join-Path $OutDir "chat_log.json") -Encoding UTF8
Write-Host "Логи сохранены в $OutDir\chat_log.json"