# -*- coding: utf-8 -*-
"""
阈值后校准分析（2025年专项）
====================================
输入: out/H10_S40_FINAL5_nn_test_predictions.csv（10天版 5-seed 的逐fold样本外预测）
所有预测均为OOS概率，本脚本只做"事后阈值选择"，不重训练。

三种无泄漏校准方案（2025样本绝不出现在阈值选择数据中）：
  A. 历史(<=2024)全局最优固定阈值 -> 应用到2025
  B. 滚动窗口校准: 对每个2025样本, 仅用其日期前W天的OOS预测选局部最优th
  C. 参照: 当前per-fold val_acc阈值(原始结果) + 全表阈值扫描(描述性)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BASE = Path(r"f:\43_华泰证券预处理\43_华泰证券预处理\model_nn")
PRED = BASE / "out" / "H10_S40_FINAL5_nn_test_predictions.csv"
OUT = BASE / "out" / "H10_S40_FINAL5_threshold_calib.txt"

THS = np.round(np.arange(0.35, 0.6501, 0.01), 2)
WINDOWS = [120, 250]  # 滚动校准回看窗口（OOS预测样本数）


def acc_at(sub: pd.DataFrame, th: float) -> float:
    return accuracy_score(sub["y_true"], (sub["proba_down"] >= th).astype(int))


def f1_at(sub: pd.DataFrame, th: float) -> float:
    return f1_score(sub["y_true"], (sub["proba_down"] >= th).astype(int),
                    average="macro", zero_division=0)


def best_th(sub: pd.DataFrame) -> float:
    return float(max(THS, key=lambda t: acc_at(sub, t)))


def main() -> None:
    df = pd.read_csv(PRED, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    cur = df[df["year"] == 2025]
    hist = df[df["year"] <= 2024]
    lines: list[str] = []

    def w(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    w("=" * 64)
    w("阈值后校准: H10_S40_FINAL5 (10天, 5seed, OOS概率)")
    w(f"样本: 全局={len(df)}  2025={len(cur)}  (<=2024={len(hist)})")
    w("=" * 64)

    # ---- C. 阈值扫描（描述性，含2025信息，仅看敏感性） ----
    w("\n[1] 阈值扫描 (th | 全局Acc/F1 | 2025 Acc/F1)")
    w(f"{'th':>5} | {'全局Acc':>7} {'全局F1':>7} | {'2025Acc':>7} {'2025F1':>7}")
    for t in THS:
        w(f"{t:5.2f} | {acc_at(df, t):7.4f} {f1_at(df, t):7.4f} | "
          f"{acc_at(cur, t):7.4f} {f1_at(cur, t):7.4f}")

    # ---- A. 历史固定阈值 ----
    w("\n[2] 方案A: 用<=2024历史选最优固定阈值, 应用到2025 (无泄漏)")
    th_hist = best_th(hist)
    w(f"    历史最优th={th_hist:.2f} (历史Acc={acc_at(hist, th_hist):.4f}, "
      f"原始th均值={df['threshold'].mean():.3f})")
    w(f"    -> 2025: Acc={acc_at(cur, th_hist):.4f}  F1={f1_at(cur, th_hist):.4f}")
    w(f"    -> 全局: Acc={acc_at(df, th_hist):.4f}  F1={f1_at(df, th_hist):.4f}")

    # ---- B. 滚动窗口校准（向量化：前缀和求任意窗口内各阈值正确数） ----
    w("\n[3] 方案B: 滚动窗口校准 (每个2025样本仅用其日期前W天OOS预测选th)")
    p = df["proba_down"].to_numpy()
    yv = df["y_true"].to_numpy()
    # correct[k, i] = 样本i在阈值THS[k]下是否判对; cum[k, i] = 前 i 个样本判对数
    correct = ((p[None, :] >= THS[:, None]).astype(int) == yv[None, :]).astype(int)
    cum = np.zeros((len(THS), len(df) + 1), dtype=int)
    cum[:, 1:] = np.cumsum(correct, axis=1)
    for W in WINDOWS:
        preds, used_th, ys = [], [], []
        for i in np.where(df["year"].to_numpy() == 2025)[0]:
            hi = max(0, i - 10)            # 右端回退10行: 10天horizon标签才落地
            lo = max(0, hi - W)
            if hi - lo < 30:
                continue
            accs = (cum[:, hi] - cum[:, lo]) / (hi - lo)
            k = int(np.argmax(accs))
            used_th.append(float(THS[k]))
            preds.append(int(p[i] >= THS[k]))   # 预测标签（非判对标志）
            ys.append(int(yv[i]))
        if preds:
            w(f"    W={W}: 覆盖{len(preds)}/{len(cur)}  "
              f"2025Acc={accuracy_score(ys, preds):.4f}  "
              f"th均值={np.mean(used_th):.3f}±{np.std(used_th):.3f}")

    # ---- 汇总 ----
    w("\n[4] 对比汇总")
    base_acc = accuracy_score(cur["y_true"], cur["pred"])
    w(f"    原始(per-fold val_acc th): 2025 Acc={base_acc:.4f} | "
      f"全局 Acc={accuracy_score(df['y_true'], df['pred']):.4f}")
    w(f"    方案A(历史th={th_hist:.2f}):   2025 Acc={acc_at(cur, th_hist):.4f} | "
      f"全局 Acc={acc_at(df, th_hist):.4f}")

    # ---- B' 全样本滚动校准（验证方法普适性，非2025专用） ----
    # 注意标签可用性: 样本j的标签在date_j+10天才落地, 故窗口右端需再回退LAG行防前视
    LAG = 10
    w(f"\n[5] 全样本滚动校准 B' (每样本用其前W天OOS预测选th, 全期适用, lag={LAG}行防标签前视)")
    yr = df["year"].to_numpy()
    for W in WINDOWS:
        preds, ys, yy = [], [], []
        for i in range(len(df)):
            hi = max(0, i - LAG)          # 右端回退: 只用标签已落地的样本
            lo = max(0, hi - W)
            if hi - lo < 30:
                continue
            accs = (cum[:, hi] - cum[:, lo]) / (hi - lo)
            k = int(np.argmax(accs))
            preds.append(int(p[i] >= THS[k]))   # 预测标签（非判对标志）
            ys.append(int(yv[i]))
            yy.append(int(yr[i]))
        preds, ys, yy = np.array(preds), np.array(ys), np.array(yy)
        g = accuracy_score(ys, preds)
        w(f"    W={W}: 覆盖{len(ys)}/{len(df)}  全局Acc={g:.4f}  预测1占比={preds.mean():.3f}")
        for y in np.unique(yy):
            m = yy == y
            w(f"        {y}: n={m.sum():4d}  acc={accuracy_score(ys[m], preds[m]):.4f}  "
              f"pred1={preds[m].mean():.3f}  true1={ys[m].mean():.3f}")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n输出: {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
