param(
    [string]$ScenarioFile = "examples/scenario_verification_demo.json",
    [string]$LogFile = "",
    [switch]$DebugLog = $true
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

if ($DebugLog) {
    $env:MILP_DEBUG = "1"
    if ([string]::IsNullOrWhiteSpace($LogFile)) {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $LogFile = "outputs/online_gui_debug_$stamp.log"
    }
}

$args = @("-u", "-m", "milp_sim.main", "--gui-online")
if (-not [string]::IsNullOrWhiteSpace($ScenarioFile)) {
    $args += @("--scenario-file", $ScenarioFile)
}

$prevErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$exitCode = 0
try {
    if (-not [string]::IsNullOrWhiteSpace($LogFile)) {
        $logDir = Split-Path -Parent $LogFile
        if (-not [string]::IsNullOrWhiteSpace($logDir)) {
            New-Item -ItemType Directory -Force -Path $logDir | Out-Null
        }
        Write-Host "Logging to $LogFile"
        & python @args 2>&1 | Tee-Object -FilePath $LogFile
    } else {
        & python @args
    }
    $exitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $prevErrorActionPreference
}

if ($exitCode -ne 0) {
    throw "python exited with code $exitCode"
}
