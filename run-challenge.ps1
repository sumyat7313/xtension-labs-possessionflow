param(
    [ValidateSet('Evaluate', 'Notebook', 'Baseline', 'Predict', 'Check', 'Test', 'Demo', 'Score', 'Package')]
    [string]$Action = 'Check',
    [string]$DataDir = 'data',
    [string]$RunDir = 'runs/xtension-labs',
    [string]$TeamName = 'Xtension Labs',
    [string]$PythonPath = '',
    [string]$ModelPath = ''
)

$ErrorActionPreference = 'Stop'
$environmentPathFile = Join-Path $PSScriptRoot '.python-environment'
if ($PythonPath) {
    $python = $PythonPath
} elseif (Test-Path -LiteralPath $environmentPathFile) {
    $environmentRoot = (Get-Content -LiteralPath $environmentPathFile -Raw).Trim()
    $python = Join-Path $environmentRoot 'Scripts\python.exe'
} elseif (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')) {
    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw 'Python is missing. Follow SOLUTION.md or supply -PythonPath with a Python 3.12 executable.'
    }
    $python = $pythonCommand.Source
}
if (-not (Test-Path -LiteralPath $python)) {
    throw 'The Python executable is missing. Follow SOLUTION.md or supply -PythonPath.'
}

# Keep writable runtime files local to this checkout, including fresh source ZIPs.
$runtimeRoot = Join-Path $PSScriptRoot 'runs\local-runtime'
$env:TEMP = Join-Path $runtimeRoot 'temp'
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = Join-Path $runtimeRoot 'matplotlib'
$env:JUPYTER_CONFIG_DIR = Join-Path $runtimeRoot 'jupyter-config'
$env:JUPYTER_DATA_DIR = Join-Path $runtimeRoot 'jupyter-data'
$env:JUPYTER_RUNTIME_DIR = Join-Path $runtimeRoot 'jupyter-runtime'
$env:IPYTHONDIR = Join-Path $runtimeRoot 'ipython'
foreach ($folder in @($env:TEMP, $env:MPLCONFIGDIR, $env:JUPYTER_CONFIG_DIR,
                     $env:JUPYTER_DATA_DIR, $env:JUPYTER_RUNTIME_DIR, $env:IPYTHONDIR)) {
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
}

Push-Location $PSScriptRoot
try {
    switch ($Action) {
        'Evaluate' { & $python 'evaluation\run_evaluation.py' }
        'Notebook' { & $python -m notebook --ServerApp.ip=127.0.0.1 }
        'Baseline' { & $python 'starter\baseline.py' }
        'Predict' {
            $predictArguments = @('-m', 'possession', 'predict', '--data-dir', $DataDir,
                                  '--config', 'configs/default.json', '--out', $RunDir)
            if (-not $ModelPath) {
                $packagedModel = Join-Path $PSScriptRoot 'models\submission.json'
                if (Test-Path -LiteralPath $packagedModel) { $ModelPath = $packagedModel }
            }
            if ($ModelPath) { $predictArguments += @('--model', $ModelPath) }
            & $python @predictArguments
        }
        'Check' { & $python -m possession doctor --data-dir $DataDir }
        'Test' { & $python -m unittest discover -s tests -v }
        'Demo' { & $python -m possession demo }
        'Score' { & $python -m possession evaluate --run-dir $RunDir }
        'Package' { & $python -m possession bundle --run-dir $RunDir --team-name $TeamName --out (Join-Path 'dist' $TeamName) }
    }
    $resultCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $resultCode
