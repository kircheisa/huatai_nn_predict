# -*- coding: utf-8 -*-
"""
第一步：识别美元指数"从未超过100 -> 突破且连续>=5天维持在100以上"的事件
事件定义：
  - 突破前 L_look 天美元指数全部 < 100（确保"从未超过"的基准期）
  - 突破日 t=0：美元指数首次 >= 100
  - 突破后连续 >= L_hold 天美元指数 >= 100（过滤假突破）
"""
import os
import pandas as pd
import numpy as np

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

def identify_events(df, L_look=10, L_hold=5, threshold=100.0):
    """
    L_look: 突破前需要连续<100的最少天数
    L_hold: 突破后需要连续>=100的最少天数
    返回事件突破日（t=0）的索引列表
    """
    dxy = df["美元指数"].values
    n = len(df)
    events = []
    i = 0
    while i < n:
        if dxy[i] >= threshold:
            # 前置：之前 L_look 天全部 < 100
            look_start = max(0, i - L_look)
            pre_ok = np.all(dxy[look_start:i] < threshold)
            if pre_ok:
                # 后置：连续 >= threshold 的天数
                j = i
                while j < n and dxy[j] >= threshold:
                    j += 1
                run_len = j - i
                if run_len >= L_hold:
                    events.append((i, run_len))
                # 跳过整段，避免在同一段内重复触发
                i = j
                continue
            else:
                # 前置不满足，跳过本段
                j = i
                while j < n and dxy[j] >= threshold:
                    j += 1
                i = j
                continue
        i += 1
    return events

def main():
    df = load_data()
    print(f"样本 {len(df)} 行，{df['日期'].min().date()} ~ {df['日期'].max().date()}")

    # 探查不同参数下的事件数量
    print("\n=== 不同参数组合下的事件数量 ===")
    print(f"{'L_look':>8} {'L_hold':>8} {'事件数':>8}")
    for L_look in [5, 10, 15]:
        for L_hold in [3, 5, 7, 10]:
            evts = identify_events(df, L_look, L_hold)
            print(f"{L_look:>8} {L_hold:>8} {len(evts):>8}")

    # 默认参数详细列出事件
    L_look, L_hold = 10, 5
    evts = identify_events(df, L_look, L_hold)
    print(f"\n=== 默认参数 L_look={L_look}, L_hold={L_hold} 识别到 {len(evts)} 个事件 ===")
    for idx, (ei, rl) in enumerate(evts):
        d0 = df.iloc[ei]["日期"].date()
        d_end = df.iloc[min(ei + rl - 1, len(df)-1)]["日期"].date()
        sh0 = df.iloc[ei]["上证A股指数"]
        print(f"  事件{idx+1}: t0={d0} (A股={sh0:.1f}), 持续{rl}天, 结束={d_end}")

if __name__ == "__main__":
    main()
