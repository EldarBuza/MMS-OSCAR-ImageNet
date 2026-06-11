# Quick status check for the running OSCAR eval.
# Usage from PowerShell:  .\scripts\check_eval_status.ps1
# Or from any shell:  powershell -File scripts\check_eval_status.ps1

$repo = 'C:\Users\User\Desktop\Fax\MMS-OSCAR-ImageNet'
$logFile = "$repo\logs\imagenet_val_5k.log"
$errFile = "$repo\logs\imagenet_val_5k.err"
$resultsDir = "$repo\results\imagenet_val_5k"
$csvFile = "$resultsDir\results.csv"

# Find eval processes; pick the worker (highest CPU consumption — skips the Start-Process launcher).
$procs = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object { $_.CommandLine -like '*main_test.py*imagenet_val*' }
$proc = $procs | Sort-Object @{E={(Get-Process -Id $_.ProcessId -EA SilentlyContinue).CPU}; Descending=$true} | Select-Object -First 1

if ($proc) {
    $live = Get-Process -Id $proc.ProcessId -EA SilentlyContinue
    if ($live -and $live.StartTime) {
        $started = $live.StartTime
        $elapsedMin = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
        $cpuS = [math]::Round($live.CPU, 1)
        $wsMb = [math]::Round($live.WS / 1MB, 0)
        Write-Host "[ALIVE]  pid=$($proc.ProcessId)  elapsed=${elapsedMin}min  cpu=${cpuS}s  ram=${wsMb}MB  started=$($started.ToString('HH:mm:ss'))" -ForegroundColor Green
    } else {
        Write-Host "[ALIVE]  pid=$($proc.ProcessId)  (couldn't read StartTime)" -ForegroundColor Green
    }
} elseif (Test-Path $csvFile) {
    Write-Host "[FINISHED]  results.csv present — see below" -ForegroundColor Cyan
} else {
    Write-Host "[NOT RUNNING]  process gone and no final CSV — check logs\imagenet_val_5k.err" -ForegroundColor Red
}

Write-Host ""
Write-Host "--- bpp directories created (each one = one of 8 rate points completed) ---"
$dirs = Get-ChildItem $resultsDir -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime
$dirs | ForEach-Object {
    $count = (Get-ChildItem $_.FullName -File -ErrorAction SilentlyContinue | Measure-Object).Count
    $age = [math]::Round(((Get-Date) - $_.LastWriteTime).TotalMinutes, 1)
    Write-Host "  $($_.Name)  files=$count  finished_at=$($_.LastWriteTime.ToString('HH:mm:ss'))  ${age}min ago"
}
Write-Host "Total bpp folders: $($dirs.Count) / 8"

if (Test-Path $csvFile) {
    Write-Host ""
    Write-Host "--- results.csv (final metrics) ---"
    Get-Content $csvFile
}

# Show last few log lines if any
if ((Test-Path $logFile) -and ((Get-Item $logFile).Length -gt 0)) {
    Write-Host ""
    Write-Host "--- last log lines ---"
    Get-Content $logFile -Tail 5
}
