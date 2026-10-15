# Hyperparameter sweep: run configs sequentially (2 seeds, 60 epochs, patience 8)
# All paths discovered via $PSScriptRoot to avoid non-ASCII literals (PS5 ANSI pitfall)
$ErrorActionPreference = "Continue"
$env:PYTHONDONTWRITEBYTECODE = '1'
$py = "D:\soft\Anaconda3\python.exe"
$script = (Get-ChildItem -Path $PSScriptRoot -Filter "01_*.py" | Select-Object -First 1).FullName
$outDir = Join-Path $PSScriptRoot "out"
$log = Join-Path $outDir "sweep_run.log"
"script=$script" | Out-File -Append $log -Encoding utf8

$configs = @(
    @{ tag = "A_base";    extra = @() },
    @{ tag = "B_h64_lr5"; extra = @("--hidden-size", "64", "--lr", "5e-4") },
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
"SWEEP_ALL_DONE" | Out-File -Append $log -Encoding utf8
