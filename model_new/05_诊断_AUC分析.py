# -*- coding: utf-8 -*-
"""
诊断脚本：AUC<0.5 根因分析 + 2024-2025 预测准确率深度归因

目标：
  1) 标签分布随时间变化（是否单边市导致标签极度不平衡）
  2) 模型是否系统性反向（预测概率方向 vs 真实方向）
  3) 2024-2025 预测准确率拆解（按季度、按标签、按概率区间）
  4) best_iteration=0 现象分析（early stopping 立即触发的原因）
  5) 按时段（2022前/后、2024-2025）拆分 AUC/Acc

输入：model_new/data/ 下已有文件
  - xgb_test_predictions_bin.csv  (逐日预测)
  - xgb_fold_metrics_bin.csv      (逐fold指标)
  - rate.csv                      (原始利率，用于计算真实5日变动方向)
输出：model_new/data/diagnostic_report.txt
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "model_new" / "data"

# 支持 --suffix 参数（双周模型用 _10d）
SUF = "_bin"
if "--suffix" in sys.argv:
    idx = sys.argv.index("--suffix")
    SUF = sys.argv[idx + 1]

# === 加载 ===
pred = pd.read_csv(DATA / f"xgb_test_predictions{SUF}.csv")
pred["date"] = pd.to_datetime(pred["date"])
pred["year"] = pred["date"].dt.year
pred["quarter"] = pred["date"].dt.to_period("Q").astype(str)

fold_df = pd.read_csv(DATA / f"xgb_fold_metrics{SUF}.csv")

# 利率真实走势（用于判断单边市）
rate = pd.read_csv(DATA / "rate.csv")
rate["time"] = pd.to_datetime(rate["time"])
rate["year"] = rate["time"].dt.year

lines = []
def p(s=""):
    print(s, flush=True)
    lines.append(str(s))

p("=" * 70)
p("AUC<0.5 根因诊断 + 2024-2025 归因报告")
p("=" * 70)

# ----------------------------------------------------------------
# 1. 标签分布随时间变化（测试集）
# ----------------------------------------------------------------
p("\n[1] 标签分布随时间（测试集，下行=1/上行=0）")
label_by_year = pred.groupby("year")["y_true"].agg(
    n="count", down_rate="mean"  # down_rate = 下行占比
)
label_by_year["up_rate"] = 1 - label_by_year["down_rate"]
p(label_by_year.round(3).to_string())

# ----------------------------------------------------------------
# 2. 真实利率年度走势（判断单边市）
# ----------------------------------------------------------------
p("\n[2] 真实10Y利率年度统计（判断单边/震荡市）")
rate_yearly = rate.groupby("year")["rate"].agg(
    n="count", start_rate=lambda s: s.iloc[0], end_rate=lambda s: s.iloc[-1],
    mean_rate="mean", min_rate="min", max_rate="max",
)
rate_yearly["annual_change_bp"] = (rate_yearly["end_rate"] - rate_yearly["start_rate"]) * 100  # bp
rate_yearly["range_bp"] = (rate_yearly["max_rate"] - rate_yearly["min_rate"]) * 100
p(rate_yearly.round(2).to_string())

# ----------------------------------------------------------------
# 3. 模型预测分布：是否系统性偏向"下行"
# ----------------------------------------------------------------
p("\n[3] 模型预测分布（是否系统性偏向某一类）")
pred_dist = pred.groupby("year")["y_pred"].agg(
    pred_down_rate=lambda s: s.mean()  # 预测下行的比例
)
pred_dist["pred_up_rate"] = 1 - pred_dist["pred_down_rate"]
pred_dist["true_down_rate"] = label_by_year["down_rate"]
pred_dist["预测偏差"] = pred_dist["pred_down_rate"] - pred_dist["true_down_rate"]
p(pred_dist.round(3).to_string())

# ----------------------------------------------------------------
# 4. 分时段 AUC / Acc / MacroF1（双轨评估）
# ----------------------------------------------------------------
p("\n[4] 分时段核心指标（重点：2024-2025）")


def metrics_block(df, name):
    if len(df) < 5:
        p(f"  {name}: 样本不足 ({len(df)})")
        return
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    prob1 = df["prob_1_下行"].values
    try:
        auc = roc_auc_score(y_true, prob1)
    except ValueError:
        auc = float("nan")
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    maj = pd.Series(y_true).mode().iloc[0]
    maj_acc = (y_true == maj).mean()
    p(f"  {name:20s} | n={len(df):4d} | Acc={acc:.4f}(基线{maj_acc:.4f}) | "
      f"MacroF1={mf1:.4f} | AUC={auc:.4f} | 下行占比={y_true.mean():.3f} | 预测下行占比={y_pred.mean():.3f}")


metrics_block(pred, "全部测试期")
metrics_block(pred[pred["date"] < pd.Timestamp("2022-01-01")], "2022前")
metrics_block(pred[pred["date"] >= pd.Timestamp("2022-01-01")], "2022后")
metrics_block(pred[pred["date"] >= pd.Timestamp("2024-01-01")], "2024-2026")
metrics_block(pred[(pred["date"] >= pd.Timestamp("2024-01-01")) & (pred["date"] < pd.Timestamp("2025-01-01"))], "2024年")
metrics_block(pred[(pred["date"] >= pd.Timestamp("2025-01-01"))], "2025-2026")

# ----------------------------------------------------------------
# 5. 逐季度准确率（聚焦2024-2025）
# ----------------------------------------------------------------
p("\n[5] 逐季度准确率（聚焦2024-2025）")
q_stats = pred.groupby("quarter").apply(
    lambda g: pd.Series({
        "n": len(g),
        "acc": accuracy_score(g["y_true"], g["y_pred"]) if len(g) > 0 else np.nan,
        "下行占比": g["y_true"].mean(),
        "预测下行占比": g["y_pred"].mean(),
        "平均概率_下行": g["prob_1_下行"].mean(),
    }), include_groups=False
).round(3)
p(q_stats.to_string())

# ----------------------------------------------------------------
# 6. 模型是否系统性反向：概率方向 vs 真实方向
# ----------------------------------------------------------------
p("\n[6] 系统性反向检验（AUC<0.5 意味着概率方向反了？）")
# 如果把概率取反（1-p），AUC 是否变 > 0.5？若大幅提升，说明标签/特征符号搞反
y_true = pred["y_true"].values
prob1 = pred["prob_1_下行"].values
try:
    auc_orig = roc_auc_score(y_true, prob1)
    auc_inv = roc_auc_score(y_true, 1 - prob1)
    p(f"  原始 prob_下行 AUC: {auc_orig:.4f}")
    p(f"  反向 1-prob    AUC: {auc_inv:.4f}")
    if auc_inv > 0.55 and auc_orig < 0.50:
        p(f"  >>> 警告：反向后 AUC 显著提升，疑似标签/特征符号定义冲突")
    elif abs(auc_orig - 0.5) < 0.03:
        p(f"  >>> 概率几乎无判别力（AUC≈0.5），标签接近随机，特征无信号")
    else:
        p(f"  >>> 方向正常但信号弱（AUC 略低于 0.5 在可接受波动范围内）")
except ValueError as e:
    p(f"  AUC 计算失败: {e}")

# 按时段再验一次反向
for mask_name, mask in [
    ("2022前", pred["date"] < pd.Timestamp("2022-01-01")),
    ("2022后", pred["date"] >= pd.Timestamp("2022-01-01")),
    ("2024-2026", pred["date"] >= pd.Timestamp("2024-01-01")),
]:
    sub = pred[mask]
    if len(sub) < 10:
        continue
    try:
        a1 = roc_auc_score(sub["y_true"], sub["prob_1_下行"])
        a2 = roc_auc_score(sub["y_true"], 1 - sub["prob_1_下行"])
        p(f"  {mask_name:10s}: 原始AUC={a1:.4f} | 反向AUC={a2:.4f}")
    except ValueError:
        pass

# ----------------------------------------------------------------
# 7. best_iteration=0 现象（early stopping 立即触发）
# ----------------------------------------------------------------
p("\n[7] best_iteration 分布（early stopping 诊断）")
zero_iter = fold_df[fold_df["best_iteration"] == 0]
p(f"  best_iteration=0 的 fold 数: {len(zero_iter)} / {len(fold_df)}")
p(f"  这些 fold 平均 Acc: {zero_iter['accuracy'].mean():.4f} | 平均 AUC: {zero_iter['auc'].mean():.4f}")
p(f"  正常 fold 平均 Acc: {fold_df[fold_df['best_iteration']>0]['accuracy'].mean():.4f} | "
  f"平均 AUC: {fold_df[fold_df['best_iteration']>0]['auc'].mean():.4f}")
p("  >>> best_iteration=0 意味着第一棵树就让验证集 logloss 上升，")
p("      通常是因为训练集与验证集分布不一致（regime 切换点）或特征为纯噪音")

# 哪些时期容易出现 best_iter=0
p("\n  best_iteration=0 出现的时期：")
for _, r in zero_iter.iterrows():
    p(f"    fold {int(r['fold'])}: {r['date_start']} ~ {r['date_end']} | Acc={r['accuracy']:.3f} AUC={r['auc']:.3f}")

# ----------------------------------------------------------------
# 8. 2024-2025 深度归因
# ----------------------------------------------------------------
p("\n[8] 2024-2025 预测准确率低的原因拆解")
sub2425 = pred[pred["date"] >= pd.Timestamp("2024-01-01")].copy()
if len(sub2425) > 0:
    # 8.1 标签不平衡度
    down_rate = sub2425["y_true"].mean()
    p(f"  2024-2026 标签：下行占比 {down_rate:.3f}，上行占比 {1-down_rate:.3f}")
    if down_rate > 0.7 or down_rate < 0.3:
        p(f"  >>> 标签极度不平衡（>0.7 或 <0.3），单边市特征明显")

    # 8.2 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(sub2425["y_true"], sub2425["y_pred"], labels=[0, 1])
    p(f"\n  2024-2026 混淆矩阵 (真实\\预测):")
    p(f"           pred_上行  pred_下行")
    p(f"  true_上行   {cm[0,0]:6d}     {cm[0,1]:6d}")
    p(f"  true_下行   {cm[1,0]:6d}     {cm[1,1]:6d}")

    # 8.3 概率置信度分布
    p(f"\n  2024-2026 预测概率分布：")
    p(f"    prob_下行 均值: {sub2425['prob_1_下行'].mean():.4f} | 标准差: {sub2425['prob_1_下行'].std():.4f}")
    p(f"    概率>0.6 的样本占比: {(sub2425['prob_1_下行']>0.6).mean():.3f}")
    p(f"    概率在[0.4,0.6]摇摆带占比: {((sub2425['prob_1_下行']>=0.4)&(sub2425['prob_1_下行']<=0.6)).mean():.3f}")
    if ((sub2425["prob_1_下行"] >= 0.4) & (sub2425["prob_1_下行"] <= 0.6)).mean() > 0.5:
        p(f"    >>> 过半样本概率落在摇摆带[0.4,0.6]，模型信心不足")

    # 8.4 概率分桶准确率
    p(f"\n  2024-2026 按预测概率分桶的下行走真实命中率：")
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    labels_bin = ["<0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", ">0.7"]
    sub2425["bucket"] = pd.cut(sub2425["prob_1_下行"], bins=bins, labels=labels_bin, include_lowest=True)
    bucket_stats = sub2425.groupby("bucket", observed=True).agg(
        n=("y_true", "count"),
        真实下行走占比=("y_true", "mean"),
    )
    p(bucket_stats.to_string())
    p("  >>> 理想情况下：桶越靠右(预测下行概率越高)，真实下行走占比应单调上升")
    p("      若非单调，说明概率排序能力差（AUC低）")

# ----------------------------------------------------------------
# 9. 标签可预测性：5日方向是否接近随机
# ----------------------------------------------------------------
p("\n[9] 标签自相关性检验（5日方向是否本身就有惯性/可预测）")
# 用全量标签（不限测试集）：当前下行占比如果接近0.5且无明显自相关，则标签近乎随机
# 这里用预测集近似
y = pred["y_true"].values
# 简单 lag-1 自相关
if len(y) > 10:
    autocorr = np.corrcoef(y[:-1], y[1:])[0, 1]
    p(f"  标签 lag-1 自相关系数: {autocorr:.4f}")
    if abs(autocorr) < 0.1:
        p(f"  >>> 自相关极低，方向几乎无连续性，预测难度极高（接近掷硬币）")

p("\n" + "=" * 70)
p("诊断完成")

# 保存
report_name = f"diagnostic_report{SUF}.txt"
(DATA / report_name).write_text("\n".join(lines), encoding="utf-8")
print(f"\n报告已保存: {DATA / report_name}")
