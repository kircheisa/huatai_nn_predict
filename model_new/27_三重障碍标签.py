# -*- coding: utf-8 -*-
"""
三重障碍标签（triple-barrier）构建：交易视角的标签方案

规则（全部因果）：
  - 观察窗 N=15 个交易日；障碍幅度 B_t = k × σ252(15日Δrate)，k=1.0，
    下限3bp（σ为t日往前252日的15日差分标准差，不含未来）
  - 未来N日内先触 rate+B（上行）-> label=0；先触 rate-B（下行）-> label=1
  - N日内未触碰任何障碍 -> label=NaN（剔除，天然消灭死区噪声）
  - 与触碰时点无关、样本权重恒为1

输出（与21脚本兼容的格式）:
  data_new/43v2_tb_labels.csv  (label / rate_delta_5d(占用位,=触碰方向×B) / sample_weight)
  data_new/43v2_tb_meta.csv    (barrier_bp / hit_days / label，供PnL代理分析)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "data_new" / "国债10年基础数据.xlsx"
OUT_DIR = BASE / "data_new"

N_DAYS = 15
K_BARRIER = 1.0
MIN_BARRIER_BP = 3.0
SIGMA_WIN = 252


def load_rate() -> pd.Series:
    raw = pd.ExcelFile(SRC).parse("滚动稳健标准化日频", header=None)
    hi = raw.index[raw[0] == "指标名称"][0]
    names = list(raw.iloc[hi, 1:])
    df = raw.iloc[hi + 1:].copy()
    df.columns = ["date"] + names
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    r = pd.to_numeric(df["中债国债到期收益率:10年"]).mask(lambda s: s == 0)
    return r.dropna()


def main():
    r = load_rate()
    grid = r.index
    # 完全因果的滚动σ：back15[j]=r[j]-r[j-15]，t日只聚合 j<=t 的历史差分
    back15 = r.diff(N_DAYS)
    hist_sigma = back15.rolling(SIGMA_WIN, min_periods=120).std()
    barrier = (K_BARRIER * hist_sigma).clip(lower=MIN_BARRIER_BP / 100.0)

    n = len(r)
    labels = np.full(n, np.nan)
    hit_days = np.full(n, np.nan)
    barrier_bp = np.full(n, np.nan)
    rv = r.values
    bv = barrier.values
    for i in range(n):
        if np.isnan(bv[i]) or i + N_DAYS >= n:
            continue
        up, dn = rv[i] + bv[i], rv[i] - bv[i]
        for j in range(i + 1, i + N_DAYS + 1):
            if rv[j] >= up:
                labels[i], hit_days[i] = 0.0, j - i
                break
            if rv[j] <= dn:
                labels[i], hit_days[i] = 1.0, j - i
                break
        barrier_bp[i] = bv[i] * 100

    lab = pd.DataFrame({
        "label": labels,
        "rate_delta_5d": np.where(labels == 1, -barrier_bp, barrier_bp),
        "sample_weight": np.where(np.isnan(labels), np.nan, 1.0),
    }, index=grid)
    meta = pd.DataFrame({"barrier_bp": barrier_bp, "hit_days": hit_days,
                         "label": labels}, index=grid)

    valid = lab["label"].notna()
    print(f"样本网格 {n} 天 | TB有效标签 {int(valid.sum())} 天 "
          f"({valid.mean():.1%})")
    print(f"标签分布: 下行(触下障) {(labels == 1).sum()} / "
          f"上行 {(labels == 0).sum()}  (下行占比 "
          f"{np.nanmean(labels):.3f})")
    print(f"触发时点: 中位 {np.nanmedian(hit_days):.0f} 日 / "
          f"均值 {np.nanmean(hit_days):.1f} 日")
    print(f"障碍幅度: 中位 {np.nanmedian(barrier_bp):.1f}bp / "
          f"2018后中位 {np.nanmedian(barrier_bp[grid >= '2018-01-01']):.1f}bp")
    lab.to_csv(OUT_DIR / "43v2_tb_labels.csv", encoding="utf-8-sig")
    meta.to_csv(OUT_DIR / "43v2_tb_meta.csv", encoding="utf-8-sig")
    print("输出:", OUT_DIR / "43v2_tb_labels.csv")


if __name__ == "__main__":
    main()
