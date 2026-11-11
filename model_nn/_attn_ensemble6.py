# -*- coding: utf-8 -*-
"""
注意力缩放配对实验的融合记录：BASE_ATTN 与 SCALED_ATTN 概率平均（6模型集成）

背景：SelfAttention 加 1/sqrt(d) 缩放前后的两次默认配置运行（各3-seed集成），
单版指标持平（AUC 0.5745 vs 0.5741），但两版预测方向一致率仅76%（相关性0.60），
概率平均后降方差有净增益（AUC 0.5815，接近XGB基线0.5822）。

输入（model_nn/out/）：
  BASE_ATTN_nn_test_predictions.csv   -> 未缩放版逐日预测
  SCALED_ATTN_nn_test_predictions.csv -> 缩放版逐日预测
  BASE_ATTN_nn_fold_metrics.csv       -> fold边界（两版同28 folds）

输出（model_nn/out/）：
  ENSEMBLE6_nn_test_predictions.csv -> 融合逐日预测
  ENSEMBLE6_nn_fold_metrics.csv     -> 融合逐fold指标
  ENSEMBLE6_nn_metrics.txt          -> 融合全局/分年指标
  sweep_log.csv 追加一行 ENSEMBLE6

用法：
  python model_nn/_attn_ensemble6.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent / "out"

BASE_TAG = "BASE_ATTN"
SCAL_TAG = "SCALED_ATTN"
TAG = "ENSEMBLE6"


def load_predictions() -> pd.DataFrame:
    b = pd.read_csv(OUT / f"{BASE_TAG}_nn_test_predictions.csv")
    s = pd.read_csv(OUT / f"{SCAL_TAG}_nn_test_predictions.csv")
    m = b.merge(s, on=["date", "row"], suffixes=("_b", "_s"))
    assert len(m) == len(b) == len(s), "两版预测样本未逐行对齐"
    assert (m["y_true_b"] == m["y_true_s"]).all(), "两版标签不一致"
    m = m.rename(columns={"y_true_b": "y_true"})
    m["proba_down"] = (m["proba_down_b"] + m["proba_down_s"]) / 2.0
    m["threshold"] = 0.5
    m["pred"] = (m["proba_down"] >= 0.5).astype(int)
    return m


def assign_folds(pred: pd.DataFrame) -> pd.DataFrame:
    """按fold边界表把逐日预测映射回fold（两版共用同一组边界）"""
    fm = pd.read_csv(OUT / f"{BASE_TAG}_nn_fold_metrics.csv")
    fm["date_start"] = pd.to_datetime(fm["date_start"])
    fm["date_end"] = pd.to_datetime(fm["date_end"])
    d = pd.to_datetime(pred["date"])
    fold_ids = pd.Series(np.full(len(pred), -1), index=pred.index)
    for _, r in fm.iterrows():
        fold_ids[(d >= r["date_start"]) & (d <= r["date_end"])] = r["fold"]
    assert (fold_ids >= 0).all(), "存在未落入任何fold的预测日期"
    pred["fold"] = fold_ids
    return pred, fm


def main():
    pred = load_predictions()
    pred, fm = assign_folds(pred)

    # 逐fold指标
    rows = []
    for _, r in fm.iterrows():
        g = pred.loc[pred["fold"] == r["fold"]]
        try:
            auc = roc_auc_score(g["y_true"], g["proba_down"])
        except ValueError:
            auc = np.nan
        rows.append({"fold": r["fold"], "date_start": r["date_start"].date().isoformat(),
                     "date_end": r["date_end"].date().isoformat(), "n_test": len(g),
                     "accuracy": accuracy_score(g["y_true"], g["pred"]),
                     "macro_f1": f1_score(g["y_true"], g["pred"], average="macro",
                                          zero_division=0),
                     "auc": auc, "threshold": 0.5})
    fold_metrics = pd.DataFrame(rows)

    # 全局 + 分年指标
    yt, yp, pb = pred["y_true"], pred["pred"], pred["proba_down"]
    g_acc = accuracy_score(yt, yp)
    g_f1 = f1_score(yt, yp, average="macro", zero_division=0)
    g_auc = roc_auc_score(yt, pb)
    pred["year"] = pd.to_datetime(pred["date"]).dt.year
    year_stats = pred.groupby("year").apply(
        lambda g: pd.Series({
            "n": len(g),
            "acc": accuracy_score(g["y_true"], g["pred"]),
            "auc": roc_auc_score(g["y_true"], g["proba_down"])
            if g["y_true"].nunique() > 1 else np.nan,
        }), include_groups=False)

    lines = [
        "=" * 56,
        f"{TAG}: {BASE_TAG} + {SCAL_TAG} 概率平均（1块, hidden=32, top-k=30）",
        "两版各3-seed集成 -> 等效6模型；th=0.5 fixed；其余配置同默认",
        "-" * 56,
        f"全局: Acc={g_acc:.4f}  MacroF1={g_f1:.4f}  AUC={g_auc:.4f}",
        f"单版: {BASE_TAG} Acc=0.5385/AUC=0.5745 | {SCAL_TAG} Acc=0.5442/AUC=0.5741",
        "XGB基线: Acc=0.5635  MacroF1=0.5627  AUC=0.5822",
        "-" * 56,
        "分年:",
        year_stats.round(4).to_string(),
        "-" * 56,
        f"folds={len(fold_metrics)}  预测样本={len(pred)}",
        "=" * 56,
    ]
    report = "\n".join(lines)
    print(report)

    pred.to_csv(OUT / f"{TAG}_nn_test_predictions.csv", index=False,
                encoding="utf-8-sig")
    fold_metrics.to_csv(OUT / f"{TAG}_nn_fold_metrics.csv", index=False,
                        encoding="utf-8-sig")
    (OUT / f"{TAG}_nn_metrics.txt").write_text(report, encoding="utf-8")

    # sweep汇总日志（与01脚本同格式；secs=两版训练耗时之和280+311）
    with open(OUT / "sweep_log.csv", "a", encoding="utf-8-sig") as f:
        f.write(f"{TAG},{g_acc:.4f},{g_f1:.4f},{g_auc:.4f},"
                f"1,32,60,30,6,0.05,591\n")

    print(f"\n输出: {OUT / f'{TAG}_nn_metrics.txt'}")


if __name__ == "__main__":
    main()
