param(
  [string]$WorkingDir = (Split-Path -Parent $PSScriptRoot)
)

$LogDir = Join-Path $WorkingDir "logs"
$PidFile = Join-Path $LogDir "background_pids.json"

if (-not (Test-Path $PidFile)) {
  Write-Host "No PID file found at $PidFile. Nothing to stop."
  exit 0
}

try {
  $map = Get-Content $PidFile -Raw | ConvertFrom-Json
} catch {
  Write-Host "PID file is corrupt; deleting."
  Remove-Item -Force $PidFile -ErrorAction SilentlyContinue
  exit 0
}

$stopped = 0
foreach ($key in $map.PSObject.Properties.Name) {
  $procId = [int]$map[$key]
  try {
    $proc = Get-Process -Id $procId -ErrorAction Stop
    Write-Host "Stopping $key (PID $procId) ..."
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    $stopped++
  } catch {
    # Already stopped
  }
}

Remove-Item -Force $PidFile -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "Stopped $stopped background processes" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""
