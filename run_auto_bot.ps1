Set-Location "E:\FutureAdmiral-Quant"
$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path "reports\bot_run.log" -Value "`n=== AUTO-BOT FULL SCAN: $ts ==="

& $py fa_quant_engine.py *>> "reports\bot_run.log"
& $py live_paper_bot.py *>> "reports\bot_run.log"

$files = @("reports\live_stats.json", "reports\audit_trade_log.csv", "reports\live_dual_wallet.json", "reports\live_paper_wallet.json", "reports\live_paper_trades.csv", "reports\student_proof_summary.md") | Where-Object { Test-Path $_ }
if ($files.Count -gt 0) {
    git add $files
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m "Auto-bot 4h scan & dual-tier update ($ts)"
        git push
    }
}
