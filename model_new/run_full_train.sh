#!/usr/bin/env bash
# 运行完整 XGB 二分类训练脚本（含反馈优化 方案1+5+6）
# - 限制 BLAS 线程数避免 OpenBLAS 内存分配失败
# - 使用 --load-features 绕过 STL 内存瓶颈

set -e

# 切换到项目根目录（脚本位于 model_new/，项目根在上一级）
cd "$(dirname "$0")/.." || exit 1

# 线程限制（避免 OpenBLAS 多线程预分配大块内存导致 OOM）
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 输出日志（带时间戳）
LOG="model_new/data/run_full_$(date +%Y%m%d_%H%M%S).log"

echo "=== 完整 XGB 训练（反馈优化版 方案1+5+6）===" | tee "$LOG"
echo "开始时间: $(date)" | tee -a "$LOG"
echo "日志文件: $LOG"
echo "" | tee -a "$LOG"

# 运行主脚本（复用预存特征，绕过 build_features 的 STL 内存瓶颈）
python -u model_new/03_XGB训练_二分类.py \
    --load-features xgb_train_features_bin.csv \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "结束时间: $(date)" | tee -a "$LOG"
echo "=== 完成 ===" | tee -a "$LOG"
