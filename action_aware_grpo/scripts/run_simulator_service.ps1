param(
    [Parameter(Mandatory=$true)][string]$EvalRoot,
    [Parameter(Mandatory=$true)][string]$TaskJsonRoot,
    [string]$PythonExe = 'python',
    [string]$BindAddress = '127.0.0.1',
    [int]$Port = 8765,
    [int]$UEPort = 9396,
    [string]$Resolution = '256,256'
)
$ErrorActionPreference = 'Stop'
$runtimeRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\runtime'))
$evalPath = (Resolve-Path -LiteralPath $EvalRoot).Path
$taskPath = (Resolve-Path -LiteralPath $TaskJsonRoot).Path
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. Reuse or stop the existing service explicitly."
}
Push-Location $evalPath
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $evalPath + [IO.Path]::PathSeparator + $previousPythonPath
try {
    & $PythonExe (Join-Path $runtimeRoot 'client.py') --mode service --host $BindAddress --port $Port --ue_port $UEPort --resolution $Resolution --task_json_root $taskPath --action_frame body --body_apply_order trans_first
    if ($LASTEXITCODE -ne 0) { throw "Simulator service exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
