# -*- coding: utf-8 -*-
"""
从预测项目原始数据 2026_ori_data_new.xlsx 导出干净版多sheet Excel，
供 bond_show 仪表盘"相关数据"模块展示。

输出: data/ori_data_new.xlsx（4 sheet：日频+买卖行为 / 日频 / 周频 / 月频）
      data/10_bond_rate_ori.xlsx（主数据：10Y收益率两列格式，覆盖旧版；
        全部总览/历史/统计/模拟模块共用，已剔除 0=缺失值，更新至最新交易日）
  - 首行 = 日期 + 指标名（去掉 Wind 行与"指标名称"表头行）
  - 日期升序、格式 YYYY-MM-DD；0 值保留原样（主文件除外）

用法: python export_ori_data.py
  （默认从 ../43_华泰证券预处理/data_ori/2026_ori_data_new.xlsx 读取，
   可用环境变量 ORI_DATA_SOURCE 覆盖）
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = os.environ.get("ORI_DATA_SOURCE") or str(
    (HERE / ".." / "data_ori" / "2026_ori_data_new.xlsx"))
OUT = HERE / "data" / "ori_data_new.xlsx"
OUT_MAIN = HERE / "data" / "10_bond_rate_ori.xlsx"
SHEETS = ["滚动稳健标准化日频+买卖行为", "滚动稳健标准化日频",
          "滚动稳健标准化周频", "滚动稳健标准化月频"]
SHORT = {"滚动稳健标准化日频+买卖行为": "日频+买卖行为",
         "滚动稳健标准化日频": "日频", "滚动稳健标准化周频": "周频",
         "滚动稳健标准化月频": "月频"}


def clean_sheet(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    raw = xl.parse(sheet, header=None)
    hi = raw.index[raw[0] == "指标名称"][0]
    names = ["日期"] + [str(n) for n in raw.iloc[hi, 1:]]
    df = raw.iloc[hi + 1:].copy()
    df.columns = names
    df["日期"] = pd.to_datetime(df["date" if "date" in df.columns else "日期"],
                               errors="coerce")
    df = df.dropna(subset=["日期"]).sort_values("日期")
    df["日期"] = df["日期"].dt.strftime("%Y-%m-%d")
    order = ["日期"] + [c for c in df.columns if c != "日期"]
    return df[order]


def main() -> None:
    xl = pd.ExcelFile(SRC)
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        for s in SHEETS:
            df = clean_sheet(xl, s)
            df.to_excel(writer, sheet_name=SHORT[s][:31], index=False)
            print(f"  [{SHORT[s]}] {df.shape[0]}行×{df.shape[1]}列")
    size = OUT.stat().st_size / 1e6
    print(f"输出: {OUT}  ({size:.1f} MB)")

    # 主数据文件：10Y收益率两列（0=缺失剔除），供总览/历史/统计/模拟模块
    daily = clean_sheet(xl, "滚动稳健标准化日频")
    col = "中国:国债到期收益率:10年"
    s = daily[["日期", col]].copy()
    s[col] = pd.to_numeric(s[col], errors="coerce")
    s = s[s[col].notna() & (s[col] != 0)]
    with pd.ExcelWriter(OUT_MAIN, engine="openpyxl") as writer:
        frame = pd.DataFrame({"指标名称": [col] * 0})  # 占位
        pd.DataFrame({col: []}).to_excel(writer, sheet_name="Sheet1", index=False)
        # Wind 风格两列：首行标记 + 表头行 + 数据（解析器跳过含"指标名称"的首行）
        raw = pd.DataFrame([["=[1]!edb()", None],
                            ["指标名称", col]] +
                           [[r["日期"], r[col]] for r in s.to_dict("records")])
        raw.to_excel(writer, sheet_name="Sheet1", index=False, header=False)
    print(f"输出: {OUT_MAIN}  主数据 {len(s)} 天 "
          f"({s['日期'].iloc[0]} ~ {s['日期'].iloc[-1]})")


if __name__ == "__main__":
    main()
