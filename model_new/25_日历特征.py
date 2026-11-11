# -*- coding: utf-8 -*-
"""
日历/流动性时点特征：在 43v2_features.csv 基础上追加因果日历特征

资金面利率的月末/季末效应、缴税大月、跨季考核等是硬逻辑，且完全无泄漏
（只依赖日期本身）。特征族：
  cal__距月末交易日数 / cal__距月初交易日数（资金面日历压力）
  cal__月末3日 / cal__月初3日 / cal__季末3日 / cal__年末半月 dummy
  cal__缴税大月(1/4/5/7/10) / cal__春节月(1-2月) dummy
  cal__周一 / cal__周五 dummy（利率周内效应）

输出: data_new/43v2cal_features.csv（281+12=293列）
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
NEW_DIR = BASE / "data_new"


def month_end_distances(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """基于交易日网格的月末/月初距离（自然月内逐日递减/递增计数）"""
    ym = idx.to_period("M")
    # 每个自然月内的序号与总数
    ser = pd.Series(range(len(idx)), index=idx)
    grp = ser.groupby(ym.values)
    pos_in_month = grp.cumcount()          # 月内第几个交易日(0起)
    cnt_in_month = grp.transform("count")  # 月内交易日总数
    return pos_in_month, cnt_in_month


def main() -> None:
    feats = pd.read_csv(NEW_DIR / "43v2_features.csv", index_col=0,
                        parse_dates=True)
    idx = feats.index
    pos, cnt = month_end_distances(idx)
    cal = pd.DataFrame(index=idx)
    cal["cal__距月末交易日数"] = cnt - pos - 1
    cal["cal__距月初交易日数"] = pos
    cal["cal__月末3日"] = (cal["cal__距月末交易日数"] <= 2).astype(int)
    cal["cal__月初3日"] = (pos <= 2).astype(int)

    # 季末：距季度最后一个交易日 <= 2
    q = idx.to_period("Q")
    ser = pd.Series(range(len(idx)), index=idx)
    gq = ser.groupby(q.values)
    pos_q = gq.cumcount()
    cnt_q = gq.transform("count")
    cal["cal__季末3日"] = ((cnt_q - pos_q - 1) <= 2).astype(int)
    cal["cal__年末半月"] = ((idx.month == 12) & (idx.day >= 16)).astype(int)

    month = idx.month
    cal["cal__缴税大月"] = month.isin([1, 4, 5, 7, 10]).astype(int)
    cal["cal__春节月"] = month.isin([1, 2]).astype(int)
    dow = idx.dayofweek
    cal["cal__周一"] = (dow == 0).astype(int)
    cal["cal__周五"] = (dow == 4).astype(int)

    out = feats.join(cal)
    print(f"日历特征 {cal.shape[1]} 个追加 -> {out.shape[1]} 列")
    out.to_csv(NEW_DIR / "43v2cal_features.csv", encoding="utf-8-sig")
    print("输出:", NEW_DIR / "43v2cal_features.csv")
    print(cal.tail(3).to_string())


if __name__ == "__main__":
    main()
