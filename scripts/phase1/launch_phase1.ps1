param([switch]$WaitForTraining)
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$pythonExe = Join-Path $projectRoot 'venv\Scripts\python.exe'
$runner = Join-Path $PSScriptRoot 'run_phase1.py'
$logDirectory = Join-Path $projectRoot 'reports\logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$processFile = Join-Path $projectRoot 'reports\phase1_process.json'
if (Test-Path -LiteralPath $processFile) {
    $recorded = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
    $existing = Get-Process -Id $recorded.pid -ErrorAction SilentlyContinue
    if ($existing -and $existing.ProcessName -like 'python*') {
        throw "Training pipeline already running with PID $($recorded.pid)"
    }
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logDirectory "phase1_$stamp.stdout.log"
$stderr = Join-Path $logDirectory "phase1_$stamp.stderr.log"
$runnerArguments = @('-u',('"' + $runner + '"'))
if ($WaitForTraining) { $runnerArguments += '--wait-for-training' }
$process = Start-Process -FilePath $pythonExe -ArgumentList $runnerArguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
[PSCustomObject]@{ProcessId=$process.Id;Stdout=$stdout;Stderr=$stderr;Status=(Join-Path $projectRoot 'reports\phase1_training_status.json')}
