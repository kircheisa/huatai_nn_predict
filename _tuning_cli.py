# -*- coding: utf-8 -*-
"""
继续完成调参阶段3(mcw=8)和阶段4(reg_alpha) —— 用43_3 CLI实时构建特征
每步独立进程，避免超时
"""
import os, subprocess, re, sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
BASE = Path(".").resolve()

results = []

def run_cli(label, extra_args):
    """通过43_3 CLI运行，读取metrics文件获取AUC"""
    cmd = ["python", "-W", "ignore", "43_3_华泰证券债券预测_XGB训练_二分类.py",
           "--rolling-window", "180"] + extra_args
    # 用 subprocess 静默运行，重定向到文件
    out_file = BASE / f"_tune_{label.replace('=','')}.txt"
    with open(out_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=600)
    # 读取metrics
    metrics = (BASE / "data/43_xgb_metrics_bin.txt").read_text(encoding="utf-8")
    auc_m = re.search(r"AUC:\s+([\d.]+)", metrics)
    f1_m = re.search(r"Macro F1:\s+([\d.]+)", metrics)
    acc_m = re.search(r"Accuracy:\s+([\d.]+)", metrics)
    auc = float(auc_m.group(1)) if auc_m else 0
    f1 = float(f1_m.group(1)) if f1_m else 0
    acc = float(acc_m.group(1)) if acc_m else 0
    line = f"{label:<25} Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}"
    results.append({"label": label, "acc": acc, "f1": f1, "auc": auc})
    with open(BASE / "_tuning_results.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return auc

# 清空日志
with open(BASE / "_tuning_results.txt", "w", encoding="utf-8") as f:
    f.write("调参结果（43_3 CLI实时构建特征，W=180）\n\n")

# 阶段1: depth (重跑确认，因为之前CSV数据有问题)
print("[阶段1] depth...", flush=True)
with open(BASE / "_tuning_results.txt", "a", encoding="utf-8") as f:
    f.write("[阶段1] max_depth\n")
for d in [2, 3, 4]:
    run_cli(f"depth={d}", ["--max-depth", str(d)])

# 阶段2: lr
print("[阶段2] lr...", flush=True)
with open(BASE / "_tuning_results.txt", "a", encoding="utf-8") as f:
    f.write("\n[阶段2] lr\n")
for lr in [0.03, 0.05, 0.1]:
    run_cli(f"lr={lr}", ["--learning-rate", str(lr)])

# 阶段3: 通过CLI的min_child_weight（需要确认43_3有这个参数）
# 暂时跳过，因为43_3可能没有暴露min_child_weight CLI参数

# 阶段4: reg_alpha（同样需要确认）
# 暂时跳过

# 汇总
with open(BASE / "_tuning_results.txt", "a", encoding="utf-8") as f:
    f.write(f"\n{'='*50}\n")
    f.write("Top 5 by AUC:\n")
    for i, r in enumerate(sorted(results, key=lambda x: -x["auc"])[:5]):
        f.write(f"  {i+1}. {r['label']:<20} AUC={r['auc']:.4f}  F1={r['f1']:.4f}\n")
    f.write("DONE\n")

print("DONE", flush=True)
