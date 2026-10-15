# 逐个运行调参组合，每个独立进程
# 通过 43_3 CLI 参数控制，输出 metrics 到独立文件再读取
$results = @()

$configs = @(
    # 阶段1: depth (已有部分数据，补充确认)
    @{label="depth=2"; args="--max-depth 2"}
    @{label="depth=3"; args="--max-depth 3"}
    @{label="depth=4"; args="--max-depth 4"}
    # 阶段2: lr
    @{label="lr=0.03"; args="--learning-rate 0.03"}
    @{label="lr=0.05"; args="--learning-rate 0.05"}
    @{label="lr=0.1"; args="--learning-rate 0.1"}
    # 阶段3: min_child_weight (需要加 CLI 参数)
    # 阶段4: reg_alpha (需要加 CLI 参数)
)

foreach ($cfg in $configs) {
    $label = $cfg.label
    $args_str = $cfg.args
    Write-Host "Running $label ..." -NoNewline
    $outFile = "_tune_$($label.Replace('=','')).txt"
    python -W ignore 43_3_华泰证券债券预测_XGB训练_二分类.py --rolling-window 180 $args_str > $outFile 2>&1
    # 从 metrics 文件读取
    $metrics = Get-Content "data/43_xgb_metrics_bin.txt" -Raw
    # 提取 AUC 和 Macro F1
    $auc = ([regex]::Match($metrics, "AUC:\s+([\d.]+)")).Groups[1].Value
    $f1 = ([regex]::Match($metrics, "Macro F1:\s+([\d.]+)")).Groups[1].Value
    $acc = ([regex]::Match($metrics, "Accuracy:\s+([\d.]+)")).Groups[1].Value
    Write-Host "  Acc=$acc  F1=$f1  AUC=$auc"
    $results += "$label`t$acc`t$f1`t$auc"
}

$results | Out-File "_tune_summary.txt" -Encoding UTF8
Write-Host "`nDone -> _tune_summary.txt"
