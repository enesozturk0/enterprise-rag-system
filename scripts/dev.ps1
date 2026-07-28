[CmdletBinding()]
param(
    [ValidateSet(
        "help",
        "lint",
        "format",
        "type-check",
        "test",
        "test-all",
        "test-cov",
        "quality",
        "docker-config",
        "docker-build",
        "docker-up",
        "docker-down",
        "docker-ps",
        "docker-logs"
    )]
    [string] $Task = "help"
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments)][string[]] $Arguments)

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Invoke-Compose {
    param([Parameter(ValueFromRemainingArguments)][string[]] $Arguments)

    & docker compose @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

switch ($Task) {
    "help" {
        @"
Available tasks:
  lint           Run Ruff checks
  format         Format Python files
  type-check     Run MyPy
  test           Run unit tests
  test-all       Run unit and integration tests
  test-cov       Run unit tests with coverage gate
  quality        Run all local quality checks
  docker-config  Validate Compose configuration
  docker-build   Build API and UI images
  docker-up      Start the complete stack
  docker-down    Stop containers without deleting volumes
  docker-ps      Show container status
  docker-logs    Follow service logs
"@
    }
    "lint" {
        Invoke-Python -m ruff check src tests scripts
        Invoke-Python -m ruff format --check src tests scripts
    }
    "format" {
        Invoke-Python -m ruff check --fix src tests scripts
        Invoke-Python -m ruff format src tests scripts
    }
    "type-check" {
        Invoke-Python -m mypy src --ignore-missing-imports
    }
    "test" {
        Invoke-Python -m pytest tests --ignore=tests/integration
    }
    "test-all" {
        Invoke-Python -m pytest tests
    }
    "test-cov" {
        Invoke-Python -m pytest tests `
            --ignore=tests/integration `
            --cov=src `
            --cov-report=term-missing `
            --cov-fail-under=95
    }
    "quality" {
        & $PSCommandPath lint
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        & $PSCommandPath type-check
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        & $PSCommandPath test-cov
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "docker-config" {
        Invoke-Compose config --quiet
    }
    "docker-build" {
        Invoke-Compose build api ui
    }
    "docker-up" {
        Invoke-Compose up --detach --build
    }
    "docker-down" {
        Invoke-Compose down
    }
    "docker-ps" {
        Invoke-Compose ps
    }
    "docker-logs" {
        Invoke-Compose logs --follow
    }
}
