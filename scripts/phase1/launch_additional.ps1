param([Parameter(Mandatory=$true)][string[]]$Scripts, [int]$WaitPid = 0)
# Run additional-model scripts sequentially in a hidden background process with logs.
# Example: .\scripts\phase1\launch_additional.ps1 -Scripts train_yolov5s.py,train_faster_rcnn.py
# -WaitPid N: start only after process N exits (keeps one training job on the GPU).
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$pythonExe = Join-Path $projectRoot 'venv\Scripts\python.exe'
$logDirectory = Join-Path $projectRoot 'reports\phase1_additional_models\logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
# An entry may carry arguments: 'train_faster_rcnn.py --full-schedule'.
$index = 0
$chain = ($Scripts | ForEach-Object {
    $index++
    $script, $scriptArgs = $_ -split ' ', 2
    $name = [System.IO.Path]::GetFileNameWithoutExtension($script)
    $log = Join-Path $logDirectory "${name}_${stamp}_$index.log"
    "& '$pythonExe' -u '$(Join-Path $PSScriptRoot $script)' $scriptArgs *>&1 | Out-File -FilePath '$log' -Encoding utf8; if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }"
}) -join '; '
if ($WaitPid -gt 0) { $chain = "Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue; " + $chain }
$process = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-Command',$chain) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
[PSCustomObject]@{ProcessId=$process.Id;Logs=$logDirectory;Stamp=$stamp}
