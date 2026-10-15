# -*- coding: utf-8 -*-
"""对比不同阈值下2024年三分类标签分布"""
import pandas as pd
import numpy as np

rate = pd.read_csv("data/43_rate.csv")
rate["time"] = pd.to_datetime(rate["time"])
rate = rate.set_index("time")["rate"]
delta5 = rate.shift(-5) - rate  # 未来5日变化
d = delta5[delta5.index.year >= 2024].dropna()

with open("_threshold_compare.txt", "w", encoding="utf-8") as f:
    for threshold_bp in [5, 3, 2]:
        thr = threshold_bp / 100  # BP转小数
        l0 = (d <= -thr).sum()     # 下行>=thr → 偏多
        l2 = (d >= thr).sum()      # 上行>=thr → 偏空
        l1 = len(d) - l0 - l2       # 其余 → 震荡

        # 震荡内部分布
        osc = d[(d > -thr) & (d < thr)]
        osc_down = (osc < 0).sum()
        osc_up = (osc >= 0).sum()

        f.write(f"=== 阈值 ±{threshold_bp}BP ===\n")
        f.write(f"偏多(≤-{threshold_bp}BP): {l0:3d}  ({l0/len(d):5.1%})\n")
        f.write(f"震荡(±{threshold_bp}BP): {l1:3d}  ({l1/len(d):5.1%})\n")
        f.write(f"偏空(≥+{threshold_bp}BP): {l2:3d}  ({l2/len(d):5.1%})\n")
        f.write(f"震荡内: 小幅下行 {osc_down} ({osc_down/len(osc):.1%}) | 小幅上行 {osc_up} ({osc_up/len(osc):.1%})\n")
        f.write("\n")

    # 整体年份对比
    f.write("=== 各阈值下各类占比变化 ===\n")
    for threshold_bp in [5, 3, 2]:
        thr = threshold_bp / 100
        l0 = (d <= -thr).sum()
        l1 = ((d > -thr) & (d < thr)).sum()
        l2 = (d >= thr).sum()
        f.write(f"±{threshold_bp}BP: 偏多={l0/len(d):.1%}  震荡={l1/len(d):.1%}  偏空={l2/len(d):.1%}\n")

    # 分析结论
    f.write("\n=== 结论 ===\n")
    thr3 = 0.03
    thr2 = 0.02
    l2_3 = ((d > -thr3) & (d < thr3)).sum()
    l2_2 = ((d > -thr2) & (d < thr2)).sum()
    f.write(f"±3BP: 震荡={l2_3} ({l2_3/len(d):.1%})\n")
    f.write(f"±2BP: 震荡={l2_2} ({l2_2/len(d):.1%})\n")
    f.write(f"阈值缩小后: 震荡从 {l2_3/len(d):.1%} 降到 {l2_2/len(d):.1%}\n")
    f.write(f"偏多占比从 {(d<=-0.03).sum()/len(d):.1%} 升到 {(d<=-0.02).sum()/len(d):.1%}\n")

print("DONE - see _threshold_compare.txt")
