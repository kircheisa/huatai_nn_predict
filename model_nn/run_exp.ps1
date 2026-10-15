# Sequential experiments: training window + recency decay (2 seeds each)
# ASCII-only; paths via $PSScriptRoot to avoid PS5 ANSI pitfalls
$ErrorActionPreference = "Continue"
$env:PYTHONDONTWRITEBYTECODE = '1'
$py = "D:\soft\Anaconda3\python.exe"
$script = (Get-ChildItem -Path $PSScriptRoot -Filter "01_*.py" | Select-Object -First 1).FullName
$outDir = Join-Path $PSScriptRoot "out"
$log = Join-Path $outDir "run_exp.log"
"script=$script" | Out-File -Append $log -Encoding utf8

$base = @("--hidden-size", "64", "--lr", "5e-4", "--n-seeds", "2",
          "--epochs", "60", "--patience", "8",
          "--threshold-mode", "val_macro_f1", "--label-smoothing", "0.05")

$configs = @(
    @{ tag = "WIN750";    extra = @("--max-train-days", "750") },
    @{ tag = "WIN504";    extra = @("--max-train-days", "504") },
    @{ tag = "DECAY252";  extra = @("--recency-halflife", "252") },
    @{ tag = "W750D504";  extra = @("--max-train-days", "750", "--recency-halflife", "504") }
)

foreach ($c in $configs) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    "`n===== EXP $($c.tag) START $(Get-Date -Format 'HH:mm:ss') =====" | Out-File -Append $log -Encoding utf8
    & $py -u $script @base --tag $c.tag @($c.extra) *>> $log
    "===== EXP $($c.tag) END elapsed=$([int]$sw.Elapsed.TotalSeconds)s =====" | Out-File -Append $log -Encoding utf8
}
"EXP_ALL_DONE" | Out-File -Append $log -Encoding utf8
