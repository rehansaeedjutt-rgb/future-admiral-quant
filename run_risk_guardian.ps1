Set-Location "E:\FutureAdmiral-Quant"
$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
& $py live_paper_bot.py --guardian *>> "reports\guardian.log"

$files = @("reports\live_dual_wallet.json", "reports\live_paper_wallet.json", "reports\live_paper_trades.csv", "reports\student_proof_summary.md") | Where-Object { Test-Path $_ }
if ($files.Count -gt 0) {
    git add $files
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm"
        git commit -m "Risk Guardian live trade update ($ts)"
        git push
    }
}
