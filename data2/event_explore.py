# -*- coding: utf-8 -*-
"""
事件识别与断点探索：放宽参数，寻找更多"穿越100"事件
并初步查看断点回归（RDD）视角
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE, "美元指数相关性分析.txt")

def load_data():
    df = pd.read_csv(DATA_FILE, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    num_cols = ["美元指数", "上证A股指数", "上证所:股票成交金额", "上证所:主板A股:成交金额"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c].str.replace(",", "", regex=False).str.strip(), errors="coerce")
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    return df.dropna(subset=["日期"] + num_cols).sort_values("日期").reset_index(drop=True)

def identify_events(df, L_look, L_hold, threshold=100.0):
    dxy = df["美元指数"].values
    n = len(df)
    events = []
    i = 0
    while i < n:
        if dxy[i] >= threshold:
            look_start = max(0, i - L_look)
            pre_ok = np.all(dxy[look_start:i] < threshold)
            if pre_ok:
                j = i
                while j < n and dxy[j] >= threshold:
                    j += 1
                run_len = j - i
                if run_len >= L_hold:
                    events.append((i, run_len))
                i = j
                continue
            else:
                j = i
                while j < n and dxy[j] >= threshold:
                    j += 1
                i = j
                continue
        i += 1
    return events

def main():
    df = load_data()

    print("=== 极宽松参数扫描 ===")
    print(f"{'L_look':>7} {'L_hold':>7} {'事件数':>7}")
    for L_look in [0, 1, 2, 3, 5]:
        for L_hold in [3, 5]:
            evts = identify_events(df, L_look, L_hold)
            print(f"{L_look:>7} {L_hold:>7} {len(evts):>7}")

    # 列出所有"穿越点"（不要求前置期，只要连续>=3天）
    print("\n=== 所有上穿100且连续>=3天的事件（L_look=1, L_hold=3）===")
    evts = identify_events(df, 1, 3)
    for idx, (ei, rl) in enumerate(evts):
        d0 = df.iloc[ei]["日期"].date()
        # 前一日A股
        prev_sh = df.iloc[ei-1]["上证A股指数"] if ei > 0 else np.nan
        sh0 = df.iloc[ei]["上证A股指数"]
        print(f"  事件{idx+1}: t0={d0} 前日A股={prev_sh:.1f} -> t0={sh0:.1f}, 持续{rl}天")

    # 断点回归视角：100附近±2区间的A股水平
    print("\n=== 断点回归视角：美元指数100附近A股均值 ===")
    for lo, hi in [(97,99.5),(99.5,100),(100,100.5),(100.5,103)]:
        m = (df["美元指数"]>=lo) & (df["美元指数"]<hi)
        if m.sum()>0:
            sh = df.loc[m,"上证A股指数"]
            print(f"  美元[{lo},{hi}): n={m.sum()}, 上证A股 均值={sh.mean():.1f}, 中位数={sh.median():.1f}")

if __name__ == "__main__":
    main()
