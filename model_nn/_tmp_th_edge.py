# -*- coding: utf-8 -*-
"""临时验证：阈值边界处的押注退化程度（th 贴下限/上限时是否变成无信息押注）"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r"f:\43_华泰证券预处理\43_华泰证券预处理\model_nn")
df = pd.read_csv(BASE / "out" / "H10_S40_FINAL5_nn_test_predictions.csv",
                 parse_dates=["date"]).sort_values("date").reset_index(drop=True)
p = df["proba_down"].to_numpy()
y = df["y_true"].to_numpy()

print("===== 概率分布（全样本 OOS）=====")
qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
print({q: round(float(np.quantile(p, q)), 3) for q in qs})
print(f"p<0.30 占比 {(p<0.30).mean():.1%} | p<0.36 占比 {(p<0.36).mean():.1%} | "
      f"p>0.65 占比 {(p>0.65).mean():.1%} | p>0.70 占比 {(p>0.70).mean():.1%}")

print("\n===== 2025年1-2月（th=0.36 贴下限期间）=====")
m = (df["date"] >= "2025-01-01") & (df["date"] <= "2025-02-28")
print(f"窗口内真实下行占比 = {y[m].mean():.1%}")
print(f"按 th=0.36 判：报下行天数占比 = {(p[m]>=0.36).mean():.1%}")
for th in (0.30, 0.35, 0.36, 0.40):
    print(f"  th={th:.2f} -> 报下行占比 {(p[m]>=th).mean():.1%}, "
          f"acc {((p[m]>=th).astype(int)==y[m]).mean():.3f}")

print("\n===== 2025年10-12月（th≈0.65 贴上限期间）=====")
m2 = df["date"] >= "2025-10-01"
print(f"窗口内真实下行占比 = {y[m2].mean():.1%}")
print(f"按 th=0.65 判：报下行天数占比 = {(p[m2]>=0.65).mean():.1%}")
for th in (0.65, 0.70, 0.75):
    print(f"  th={th:.2f} -> 报下行占比 {(p[m2]>=th).mean():.1%}, "
          f"acc {((p[m2]>=th).astype(int)==y[m2]).mean():.3f}")

print("\n===== 参照：押多数类（无信息）基线 =====")
print(f"2025年1-2月 全押下行 acc = {y[m].mean():.3f}")
print(f"2025年10-12月 全押上行 acc = {1-y[m2].mean():.3f}")
