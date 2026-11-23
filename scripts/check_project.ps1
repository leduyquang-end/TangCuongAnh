$ErrorActionPreference = 'Stop'
$applicationRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $applicationRoot '.venv\Scripts\python.exe'
Push-Location -LiteralPath $applicationRoot
try {
    & $projectPython -B -m kind_enhance doctor
    if ($LASTEXITCODE -ne 0) { throw 'Runtime check failed.' }
    & $projectPython -B -m unittest discover -s tests
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
}
finally { Pop-Location }
