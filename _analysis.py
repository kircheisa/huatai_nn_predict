# -*- coding: utf-8 -*-
"""分析2024年起5日变化方向与幅度分布"""
import pandas as pd
import numpy as np

rate = pd.read_csv("data/43_rate.csv")
rate["time"] = pd.to_datetime(rate["time"])
rate = rate.set_index("time")["rate"]

# 未来5日变化 = rate[t+5] - rate[t]，正=上行，负=下行
delta5 = rate.shift(-5) - rate

# 2024年起
d = delta5[delta5.index >= "2024-01-01"].dropna()
chg_bp = d * 100

with open("_analysis.txt", "w", encoding="utf-8") as f:
    f.write("=== 2024年起 未来5日变化方向 ===\n")
    f.write(f"总样本: {len(d)}\n")
    f.write(f"下行(<0): {(d<0).sum()}  ({(d<0).mean():.1%})\n")
    f.write(f"上行(>=0): {(d>=0).sum()}  ({(d>=0).mean():.1%})\n\n")

    # 三分类
    l0 = (d < -0.03).sum()
    l1 = ((d >= -0.03) & (d < 0.03)).sum()
    l2 = (d >= 0.03).sum()
    f.write("=== 三分类标签分布 ===\n")
    f.write(f"偏多(下行>=3BP): {l0}  ({l0/len(d):.1%})\n")
    f.write(f"震荡(±3BP内):   {l1}  ({l1/len(d):.1%})\n")
    f.write(f"偏空(上行>=3BP): {l2}  ({l2/len(d):.1%})\n\n")

    # 震荡类内部
    osc = d[(d >= -0.03) & (d < 0.03)]
    f.write("=== 震荡类内部构成 ===\n")
    f.write(f"  小幅下行(-3~0BP): {(osc<0).sum()}  ({(osc<0).mean():.1%})\n")
    f.write(f"  小幅上行(0~+3BP): {(osc>=0).sum()}  ({(osc>=0).mean():.1%})\n")
    f.write(f"  均值: {osc.mean()*100:.2f}BP\n\n")

    # 幅度分布
    f.write("=== 全部5日变化幅度分布(BP) ===\n")
    bins = [-50,-10,-5,-3,-1,0,1,3,5,10,50]
    cuts = pd.cut(chg_bp, bins=bins)
    f.write(cuts.value_counts().sort_index().to_string())
    f.write("\n\n")

    # 对比不同年份
    f.write("=== 各年5日变化均值(BP) ===\n")
    for year in [2021, 2022, 2023, 2024, 2025]:
        dy = delta5[delta5.index.year == year].dropna()
        if len(dy) > 0:
            f.write(f"  {year}: 均值={dy.mean()*100:.2f}BP  中位数={dy.median()*100:.2f}BP  下行占比={ (dy<0).mean():.1%}\n")

print("DONE - see _analysis.txt")
