param(
  [string]$PythonExe = (Join-Path (Split-Path -Parent $PSScriptRoot) ".conda\\python.exe"),
  [string]$WorkingDir = (Split-Path -Parent $PSScriptRoot),
  [string]$SqlDb = "data\\spider_guardian.sqlite",
  [string]$TrendingDb = "data\\spider_trending.sqlite",
  [string]$SeleniumDriver = "firefox",
  [switch]$ShowBrowser,
  [string]$TwitterAuthCookie = $null,
  [string]$LangsmithApiKey = $null,
  [int]$UpdateIntervalSeconds = 3600,
  [int]$RefreshIntervalSeconds = 7200,
  [int]$TrainIntervalSeconds = 86400,
  [int]$MonitorRefreshSeconds = 30
)

# Ensure working dir
Set-Location -Path $WorkingDir

# Prepare logs
$LogDir = Join-Path $WorkingDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$PidFile = Join-Path $LogDir "background_pids.json"

# Optionally set environment variables for child processes
if ($TwitterAuthCookie) {
  $env:TWITTER_AUTH_COOKIE = $TwitterAuthCookie
  Write-Host "Set TWITTER_AUTH_COOKIE for this session." -ForegroundColor Green
}
if ($LangsmithApiKey) {
  $env:LANGSMITH_API_KEY = $LangsmithApiKey
  Write-Host "Set LANGSMITH_API_KEY for this session." -ForegroundColor Green
}

function Save-PidEntry {
  param([string]$Name, [int]$ProcId)
  $map = @{}
  if (Test-Path $PidFile) {
    try {
      $existing = Get-Content $PidFile -Raw | ConvertFrom-Json
      if ($existing -is [hashtable]) {
        $map = $existing
      } elseif ($null -ne $existing) {
        foreach ($prop in $existing.PSObject.Properties) {
          $map[$prop.Name] = $prop.Value
        }
      }
    } catch { $map = @{} }
  }
  $map[$Name] = $ProcId
  ($map | ConvertTo-Json -Depth 5) | Out-File -FilePath $PidFile -Encoding UTF8
}

function Start-LoggedProcess {
  param(
    [string]$Name,
    [string]$Exe,
    [string[]]$ArgumentList,
    [string]$StdOutPath,
    [string]$StdErrPath
  )
  Write-Host "Starting $Name ..." -ForegroundColor Cyan
  $alist = @()
  if ($null -ne $ArgumentList) { $alist = @($ArgumentList | Where-Object { $_ -ne $null -and $_ -ne "" }) }
  $p = Start-Process -FilePath $Exe -ArgumentList $alist -WorkingDirectory $WorkingDir -WindowStyle Hidden -RedirectStandardOutput $StdOutPath -RedirectStandardError $StdErrPath -PassThru
  Save-PidEntry -Name $Name -ProcId $p.Id
  return $p
}

function Start-LoopingJob {
  param(
    [string]$Name,
    [string]$CommandScript,
    [string]$StdOutPath,
    [string]$StdErrPath
  )
  $encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($CommandScript))
  $proc = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encoded) -WorkingDirectory $WorkingDir -WindowStyle Hidden -RedirectStandardOutput $StdOutPath -RedirectStandardError $StdErrPath -PassThru
  Save-PidEntry -Name $Name -ProcId $proc.Id
  return $proc
}

# 1) Long-running Guardian Orchestrator (replies, trending, follow-ups, scheduled posts)
$orchArgs = @("-m", "spider_guardian.scripts.guardian_orchestrator", "--log-level", "INFO", "--selenium-driver", $SeleniumDriver)
if ($ShowBrowser) { $orchArgs += "--show-browser" }
$orchOut = Join-Path $LogDir "guardian_orchestrator.out.log"
$orchErr = Join-Path $LogDir "guardian_orchestrator.err.log"

# Build environment variable setting commands for Python
$envSetup = ""
if ($TwitterAuthCookie) {
  $envSetup += "`$env:TWITTER_AUTH_COOKIE='$TwitterAuthCookie'; "
}
if ($LangsmithApiKey) {
  $envSetup += "`$env:LANGSMITH_API_KEY='$LangsmithApiKey'; "
}

if ($envSetup) {
  # Use PowerShell wrapper to set environment and run Python
  $argString = ($orchArgs | ForEach-Object { if ($_ -match '\s') { "'$_'" } else { $_ } }) -join ' '
  $orchScript = @"
$envSetup
& '$PythonExe' $argString
"@
  $encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($orchScript))
  $proc = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encoded) -WorkingDirectory $WorkingDir -WindowStyle Hidden -RedirectStandardOutput $orchOut -RedirectStandardError $orchErr -PassThru
  Save-PidEntry -Name "guardian_orchestrator" -ProcId $proc.Id
} else {
  Start-LoggedProcess -Name "guardian_orchestrator" -Exe $PythonExe -ArgumentList $orchArgs -StdOutPath $orchOut -StdErrPath $orchErr | Out-Null
}

# 2) Periodic dataset updates (trending + all SQL datasets) with safe upsert
$updOut = Join-Path $LogDir "datasets_update_loop.out.log"
$updErr = Join-Path $LogDir "datasets_update_loop.err.log"
$updateScript = @"
`$ProgressPreference = 'SilentlyContinue'
$(if ($TwitterAuthCookie) { "`$env:TWITTER_AUTH_COOKIE='$TwitterAuthCookie'" })
$(if ($LangsmithApiKey) { "`$env:LANGSMITH_API_KEY='$LangsmithApiKey'" })
Write-Host ""
Write-Host "==================== SESSION START: `$(Get-Date -Format o) ===================="
Write-Host ""
while (`$true) {
  try {
    Write-Host ("$(Get-Date -Format o) [update] cycle start")
    & '$PythonExe' 'update_datasets.py' --db '$TrendingDb' --dataset 'spider-trending-dataset' --match-key 'post_id' --upload --sql-db '$SqlDb' --upload-all-sql --max-examples 200 2>&1 | ForEach-Object { "$(Get-Date -Format o) [update] `$_." }
    Write-Host ("$(Get-Date -Format o) [update] cycle complete")
  } catch {
    Write-Host "$(Get-Date -Format o) [update] Error: `$(`$_.Exception.Message)"
  }
  Start-Sleep -Seconds $UpdateIntervalSeconds
}
"@
Start-LoopingJob -Name "datasets_update_loop" -CommandScript $updateScript -StdOutPath $updOut -StdErrPath $updErr | Out-Null

# 3) Periodic refresh of reply metrics
$refOut = Join-Path $LogDir "refresh_replies_loop.out.log"
$refErr = Join-Path $LogDir "refresh_replies_loop.err.log"
$refreshScript = @"
`$ProgressPreference = 'SilentlyContinue'
$(if ($TwitterAuthCookie) { "`$env:TWITTER_AUTH_COOKIE='$TwitterAuthCookie'" })
$(if ($LangsmithApiKey) { "`$env:LANGSMITH_API_KEY='$LangsmithApiKey'" })
Write-Host ""
Write-Host "==================== SESSION START: `$(Get-Date -Format o) ===================="
Write-Host ""
while (`$true) {
  try {
    Write-Host ("$(Get-Date -Format o) [refresh] cycle start")
    & '$PythonExe' -m 'spider_guardian.scripts.refresh_my_replies' 2>&1 | ForEach-Object { "$(Get-Date -Format o) [refresh] `$_." }
    Write-Host ("$(Get-Date -Format o) [refresh] cycle complete")
  } catch {
    Write-Host "$(Get-Date -Format o) [refresh] Error: `$(`$_.Exception.Message)"
  }
  Start-Sleep -Seconds $RefreshIntervalSeconds
}
"@
Start-LoopingJob -Name "refresh_replies_loop" -CommandScript $refreshScript -StdOutPath $refOut -StdErrPath $refErr | Out-Null

# 4) Daily ML training (optional loop)
$mlOut = Join-Path $LogDir "ml_training_loop.out.log"
$mlErr = Join-Path $LogDir "ml_training_loop.err.log"
$trainScript = @"
`$ProgressPreference = 'SilentlyContinue'
$(if ($TwitterAuthCookie) { "`$env:TWITTER_AUTH_COOKIE='$TwitterAuthCookie'" })
$(if ($LangsmithApiKey) { "`$env:LANGSMITH_API_KEY='$LangsmithApiKey'" })
Write-Host ""
Write-Host "==================== SESSION START: `$(Get-Date -Format o) ===================="
Write-Host ""
while (`$true) {
  try {
    Write-Host ("$(Get-Date -Format o) [ml] cycle start")
    & '$PythonExe' -m 'spider_guardian.ml_bot' 'train' --min-replies 50 --min-trending 20 2>&1 | ForEach-Object { "$(Get-Date -Format o) [ml] `$_." }
    Write-Host ("$(Get-Date -Format o) [ml] cycle complete")
  } catch {
    Write-Host "$(Get-Date -Format o) [ml] Error: `$(`$_.Exception.Message)"
  }
  Start-Sleep -Seconds $TrainIntervalSeconds
}
"@
Start-LoopingJob -Name "ml_training_loop" -CommandScript $trainScript -StdOutPath $mlOut -StdErrPath $mlErr | Out-Null

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "All background processes started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Running processes:" -ForegroundColor Cyan
Write-Host "  - Guardian Orchestrator (replies, trending, posts)" -ForegroundColor White
Write-Host "  - Dataset Update Loop (hourly)" -ForegroundColor White
Write-Host "  - Reply Metrics Refresh Loop (every 2 hours)" -ForegroundColor White
Write-Host "  - ML Training Loop (daily)" -ForegroundColor White
Write-Host ""
Write-Host "Logs:     $LogDir" -ForegroundColor Yellow
Write-Host "PIDs:     $PidFile" -ForegroundColor Yellow
Write-Host ""
Write-Host "Monitoring logs every $MonitorRefreshSeconds seconds. Press Ctrl+C to stop monitoring (processes keep running)." -ForegroundColor Magenta
Write-Host ""

# Monitor loop: watch logs and report updates
$logFiles = @{
  "Orchestrator" = $orchOut
  "Datasets" = $updOut
  "Refresh" = $refOut
  "ML Training" = $mlOut
}

$lastLines = @{}
foreach ($name in $logFiles.Keys) {
  $lastLines[$name] = 0
}

try {
  while ($true) {
    Start-Sleep -Seconds $MonitorRefreshSeconds
    
    $timestamp = Get-Date -Format "HH:mm:ss"
    $anyUpdates = $false
    
    foreach ($name in $logFiles.Keys) {
      $path = $logFiles[$name]
      if (-not (Test-Path $path)) { continue }
      
      $lines = Get-Content $path -ErrorAction SilentlyContinue
      if ($null -eq $lines) { $lines = @() }
      $currentCount = $lines.Count
      
      if ($currentCount -gt $lastLines[$name]) {
        $newLines = $lines[($lastLines[$name])..($currentCount-1)]
        $lastLines[$name] = $currentCount
        
        if ($newLines.Count -gt 0) {
          if (-not $anyUpdates) {
            Write-Host "[$timestamp] " -ForegroundColor DarkGray -NoNewline
            Write-Host "New activity detected:" -ForegroundColor Yellow
            $anyUpdates = $true
          }
          
          Write-Host "  [$name]" -ForegroundColor Cyan
          foreach ($line in $newLines[-5..-1]) {
            if ($null -ne $line) {
              Write-Host "    $line" -ForegroundColor Gray
            }
          }
        }
      }
    }
    
    if (-not $anyUpdates) {
      Write-Host "[$timestamp] " -ForegroundColor DarkGray -NoNewline
      Write-Host "All processes running quietly..." -ForegroundColor DarkGray
    }
  }
} catch {
  Write-Host ""
  Write-Host "Monitoring stopped. Background processes are still running." -ForegroundColor Yellow
  Write-Host "To stop all processes: .\scripts\stop_all_background.ps1" -ForegroundColor Yellow
}
