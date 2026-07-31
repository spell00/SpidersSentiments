param(
  [int]$Tail = 50,
  [string]$WorkingDir = (Split-Path -Parent $PSScriptRoot)
)

Set-Location -Path $WorkingDir

$LogDir = Join-Path $WorkingDir "logs"
$PidFile = Join-Path $LogDir "background_pids.json"

if (-not (Test-Path $LogDir)) {
  Write-Host "No logs directory found at $LogDir" -ForegroundColor Yellow
  exit 1
}

if (-not (Test-Path $PidFile)) {
  Write-Host "No PID file found at $PidFile. Background jobs may not be running." -ForegroundColor Yellow
} else {
  try {
    $mapRaw = Get-Content $PidFile -Raw | ConvertFrom-Json
  } catch {
    Write-Host "PID file appears corrupt: $PidFile" -ForegroundColor Red
    $mapRaw = $null
  }

  $map = @{}
  if ($mapRaw -ne $null) {
    if ($mapRaw -is [hashtable]) { $map = $mapRaw } else { foreach ($p in $mapRaw.PSObject.Properties) { $map[$p.Name] = $p.Value } }
  }

  if ($map.Count -eq 0) {
    Write-Host "PID map is empty; no tracked processes." -ForegroundColor Yellow
  } else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Process status:" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    foreach ($name in $map.Keys) {
      $procId = [int]$map[$name]
      try {
        $proc = Get-Process -Id $procId -ErrorAction Stop
        Write-Host ("  [OK] {0,-26} : RUNNING (PID {1})" -f $name, $proc.Id) -ForegroundColor Green
      } catch {
        Write-Host ("  [X]  {0,-26} : STOPPED (was PID {1})" -f $name, $procId) -ForegroundColor Red
      }
    }
    Write-Host ""
  }
}

function Show-LogTail {
  param([string]$Label, [string]$Path, [int]$Lines)
  Write-Host "`n=== $Label ($Path) ===" -ForegroundColor Cyan
  if (Test-Path $Path) {
    Get-Content -Tail $Lines -Path $Path | ForEach-Object { $_ }
  } else {
    Write-Host "(no log file yet)" -ForegroundColor Yellow
  }
}

# Show tails of each key log
Show-LogTail -Label "Orchestrator" -Path (Join-Path $LogDir "guardian_orchestrator.out.log") -Lines $Tail
Show-LogTail -Label "Datasets Update Loop" -Path (Join-Path $LogDir "datasets_update_loop.out.log") -Lines $Tail
Show-LogTail -Label "Refresh Replies Loop" -Path (Join-Path $LogDir "refresh_replies_loop.out.log") -Lines $Tail
Show-LogTail -Label "ML Training Loop" -Path (Join-Path $LogDir "ml_training_loop.out.log") -Lines $Tail

# Quick DB file freshness
function Show-PathFreshness {
  param([string]$Label, [string]$Path)
  if (Test-Path $Path) {
    $it = Get-Item $Path
    Write-Host ("{0}: {1} (LastWriteTime: {2})" -f $Label, $Path, $it.LastWriteTime) -ForegroundColor Gray
  }
}

Show-PathFreshness -Label "SQL DB (scraped_articles)" -Path (Join-Path $WorkingDir "data\spider_guardian.sqlite")
Show-PathFreshness -Label "Trending DB" -Path (Join-Path $WorkingDir "data\spider_trending.sqlite")
