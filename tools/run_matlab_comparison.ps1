param(
    [string]$MatlabExe = 'D:\Softwares\MATLAB\bin\matlab.exe',
    [ValidateRange(1, 3600)][int]$TimeoutSeconds = 120,
    [string]$OutputDirectory = 'output/doppler-repair/matlab',
    [ValidatePattern('^[A-Za-z][A-Za-z0-9_]*$')][string]$ComparisonFunction = 'compare_matlab_radar',
    [switch]$IsolatedPreferences
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskOutput = [IO.Path]::GetFullPath((Join-Path $taskRoot $OutputDirectory))
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
if ($IsolatedPreferences) {
    $taskPrefs = Join-Path $taskOutput 'preferences'
    New-Item -ItemType Directory -Force -Path $taskPrefs | Out-Null
    $env:MATLAB_PREFDIR = $taskPrefs
}
$taskLog = Join-Path $taskOutput 'matlab-run.log'
$taskMatlabDirectory = $taskOutput.Replace('\', '/').Replace("'", "''")
$taskArguments = @('-wait', '-batch', ('"addpath(''tools''); ' + $ComparisonFunction + '(''' + $taskMatlabDirectory + ''')"'),
    '-logfile', ('"' + $taskLog + '"'))
$taskClock = [Diagnostics.Stopwatch]::StartNew()
$taskStartedUtc = [DateTime]::UtcNow
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
$taskExpected = @(Get-ChildItem -LiteralPath $taskOutput -Filter '*-input.mat' | ForEach-Object {
    Join-Path $taskOutput ($_.Name.Replace('-input.mat', '-matlab.mat'))
})
$taskFresh = @($taskExpected | Where-Object {
    (Test-Path -LiteralPath $_) -and (Get-Item -LiteralPath $_).LastWriteTimeUtc -ge $taskStartedUtc
})
$taskStatus = @{
    executable = $MatlabExe
    completed = $taskCompleted
    elapsed_seconds = $taskClock.Elapsed.TotalSeconds
    timeout_seconds = $TimeoutSeconds
    exit_code = $(if ($taskCompleted) { $taskProcess.ExitCode } else { $null })
    comparison_executed = ($taskCompleted -and $taskProcess.ExitCode -eq 0 -and
        $taskExpected.Count -gt 0 -and $taskFresh.Count -eq $taskExpected.Count)
    expected_cases = $taskExpected.Count
    fresh_results = $taskFresh.Count
}
$taskStatus | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskOutput 'launch-status.json')
$taskStatus | ConvertTo-Json
if (!$taskCompleted) { exit 124 }
if ($taskProcess.ExitCode -eq 0 -and !$taskStatus.comparison_executed) { exit 2 }
exit $taskProcess.ExitCode
