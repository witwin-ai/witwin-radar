param(
    [string]$MatlabExe = 'D:\Softwares\MATLAB\bin\matlab.exe',
    [int]$TimeoutSeconds = 120
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskOutput = Join-Path $taskRoot 'output\doppler-repair\matlab'
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
$taskPrefs = Join-Path $taskOutput 'preferences'
New-Item -ItemType Directory -Force -Path $taskPrefs | Out-Null
$env:MATLAB_PREFDIR = $taskPrefs
$taskLog = Join-Path $taskOutput 'matlab-run.log'
$taskArguments = @('-wait', '-batch', '"addpath(''tools''); compare_matlab_radar(''output/doppler-repair/matlab'')"',
    '-logfile', ('"' + $taskLog + '"'))
$taskClock = [Diagnostics.Stopwatch]::StartNew()
$taskProcess = Start-Process -FilePath $MatlabExe -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru
$taskCompleted = $taskProcess.WaitForExit($TimeoutSeconds * 1000)
if (!$taskCompleted) {
    # Terminate only this launch's descendants, then its wrapper.
    $taskChildren = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $taskProcess.Id }
    foreach ($taskChild in $taskChildren) {
        Stop-Process -Id $taskChild.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $taskProcess.Id -Force -ErrorAction SilentlyContinue
}
$taskStatus = @{
    executable = $MatlabExe
    completed = $taskCompleted
    elapsed_seconds = $taskClock.Elapsed.TotalSeconds
    timeout_seconds = $TimeoutSeconds
    exit_code = $(if ($taskCompleted) { $taskProcess.ExitCode } else { $null })
    comparison_executed = (Test-Path (Join-Path $taskOutput 'multipath-matlab.mat'))
}
$taskStatus | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskOutput 'launch-status.json')
$taskStatus | ConvertTo-Json
if (!$taskCompleted) { exit 124 }
exit $taskProcess.ExitCode
