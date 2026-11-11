# -*- coding: utf-8 -*-
"""
跨数据三源融合分析：旧数据NN(ENSEMBLE6) + 新数据NN_B + 新数据XGB

三源均为 5 天 horizon、滚动 fold 样本外概率：
  OLD  model_nn/out/ENSEMBLE6_nn_test_predictions.csv (2018-09~2026-01, 旧280特征)
  NNB  model_nn/out/NNV2_B_nnv2_test_predictions.csv  (2018-01~2026-10, 新281特征)
  XGB  data_new/out/xgbv2_test_predictions.csv        (2018-01~2026-10, 新281特征)

评估：
  1) 单源 / 两两 / 三源等权 概率平均
  2) 三源按季滚动加权（无泄漏）：每季度用"上一季度"各源AUC算权重
     w_i ∝ max(auc_i - 0.5, 0.02)，归一化；首季等权
  三源评估窗=三者日期交集（旧源只到2026-01）；另报2025全年重点指标。

输出：data_new/out/跨数据融合分析.txt + 融合逐日概率csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE = Path(__file__).resolve().parent.parent
NN_OUT = BASE / "model_nn" / "out"
OUT = BASE / "data_new" / "out"

SOURCES = {
    "OLD": NN_OUT / "ENSEMBLE6_nn_test_predictions.csv",
    "NNB": NN_OUT / "NNV2_B_nnv2_test_predictions.csv",
    "XGB": BASE / "data_new" / "out" / "xgbv2_test_predictions.csv",
}


def ev(y, p, th=0.5):
    pred = (p >= th).astype(int)
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = np.nan
    return accuracy_score(y, pred), f1_score(y, pred, average="macro",
                                             zero_division=0), auc


def main():
    dfs = {}
    for k, path in SOURCES.items():
        d = pd.read_csv(path)[["date", "y_true", "proba_down"]]
        d = d.rename(columns={"y_true": f"y_{k}", "proba_down": k})
        dfs[k] = d
    m = dfs["OLD"]
    for k in ["NNB", "XGB"]:
        m = m.merge(dfs[k], on="date")
    # 标签一致性：旧rate序列与中债序列有微小数值差，近零变动日符号翻转，
    # 三源对比仅在标签一致的日子上做（丢弃日多为死区内样本，披露于报告）
    agree = (m["y_NNB"] == m["y_OLD"]) & (m["y_XGB"] == m["y_OLD"])
    n_drop = int((~agree).sum())
    m = m.loc[agree]
    m["y"] = m["y_OLD"].astype(int)
    m = m[["date", "y", "OLD", "NNB", "XGB"]].reset_index(drop=True)
    print(f"三源交集: {len(m)}天  {m['date'].min()} ~ {m['date'].max()} "
          f"(标签不一致剔除{n_drop}天, {n_drop/(len(m)+n_drop):.1%})")

    # 组合
    m["OLD+NNB"] = (m["OLD"] + m["NNB"]) / 2
    m["OLD+XGB"] = (m["OLD"] + m["XGB"]) / 2
    m["NNB+XGB"] = (m["NNB"] + m["XGB"]) / 2
    m["三源等权"] = (m["OLD"] + m["NNB"] + m["XGB"]) / 3

    # 按季滚动加权（无泄漏）：上季AUC -> 本季权重
    q = pd.to_datetime(m["date"]).dt.to_period("Q")
    m["q"] = q.astype(str)
    weights = np.ones((len(m), 3))
    qs = m["q"].unique()
    for i, cur in enumerate(qs):
        mask = (m["q"] == cur).values
        if i == 0:
            continue
        prev = qs[i - 1]
        pm = m["q"] == prev
        aucs = []
        for c in ["OLD", "NNB", "XGB"]:
            yv, pv = m.loc[pm, "y"], m.loc[pm, c]
            try:
                aucs.append(roc_auc_score(yv, pv))
            except ValueError:
                aucs.append(0.5)
        w = np.array([max(a - 0.5, 0.02) for a in aucs])
        weights[mask] = w / w.sum()
    m["三源滚动加权"] = (weights[:, 0] * m["OLD"] + weights[:, 1] * m["NNB"]
                        + weights[:, 2] * m["XGB"])

    combos = ["OLD", "NNB", "XGB", "OLD+NNB", "OLD+XGB", "NNB+XGB",
              "三源等权", "三源滚动加权"]

    lines = ["=" * 66, f"跨数据融合（交集 {m['date'].min()}~{m['date'].max()}, n={len(m)}）"]
    year = pd.to_datetime(m["date"]).dt.year
    for scope, mask in [("全局", np.ones(len(m), bool)),
                        ("2024", (year == 2024).values),
                        ("2025", (year == 2025).values)]:
        lines.append("-" * 66)
        lines.append(f"[{scope}]  n={int(mask.sum())}")
        for c in combos:
            acc, f1, auc = ev(m.loc[mask, "y"], m.loc[mask, c])
            lines.append(f"  {c:<10} Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    lines.append("-" * 66)
    lines.append("注: OLD源只覆盖到2026-01，故无2026全年三源对比；")
    lines.append("    新数据内部2026全年对比见 NNV2_B/xgbv2（NN_B 0.713/0.573 最强）。")
    lines.append("=" * 66)
    report = "\n".join(lines)
    print(report)
    OUT.mkdir(exist_ok=True)
    (OUT / "跨数据融合分析.txt").write_text(report, encoding="utf-8")
    m.to_csv(OUT / "跨数据融合逐日.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
