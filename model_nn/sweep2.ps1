# Resume sweep: run remaining configs C/D/E only (A/B already done)
# All paths via $PSScriptRoot; ASCII-only to avoid PS5 ANSI pitfalls
$ErrorActionPreference = "Continue"
$env:PYTHONDONTWRITEBYTECODE = '1'
$py = "D:\soft\Anaconda3\python.exe"
$script = (Get-ChildItem -Path $PSScriptRoot -Filter "01_*.py" | Select-Object -First 1).FullName
$outDir = Join-Path $PSScriptRoot "out"
$log = Join-Path $outDir "sweep_run.log"
"script=$script" | Out-File -Append $log -Encoding utf8

$configs = @(
    @{ tag = "C_2blocks"; extra = @("--n-blocks", "2") },
    @{ tag = "D_topk60";  extra = @("--top-k", "60") },
    @{ tag = "E_seq40";   extra = @("--seq-len", "40") }
)

foreach ($c in $configs) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    "`n===== CONFIG $($c.tag) START $(Get-Date -Format 'HH:mm:ss') =====" | Out-File -Append $log -Encoding utf8
    & $py -u $script --n-seeds 2 --epochs 60 --patience 8 --tag $c.tag $c.extra *>> $log
    "===== CONFIG $($c.tag) END elapsed=$([int]$sw.Elapsed.TotalSeconds)s =====" | Out-File -Append $log -Encoding utf8
}
"SWEEP2_ALL_DONE" | Out-File -Append $log -Encoding utf8
