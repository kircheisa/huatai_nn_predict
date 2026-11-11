# -*- coding: utf-8 -*-
"""
提取 data_ori/2026_origin_data.xlsx 中的原始列为 CSV

输出目录: data_new/origin_csv/
  origin2026_日频.csv / 周频 / 月频 / 季频 / 差值          —— 全量列，日期升序索引
  origin2026_月频发布日期.csv / 季频发布日期.csv            —— 发布日期表
  origin2026_rate.csv                                        —— 10Y收益率
  origin2026_有用列_日频.csv                                 —— 仅特征有用性梳理
                                                              第六节映射到的日频列
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "data_ori" / "2026_origin_data.xlsx"
RATE_SRC = BASE / "data_ori" / "2026_origin_rate.csv"
OUT = BASE / "data_new" / "origin_csv"
OUT.mkdir(exist_ok=True)

# 特征有用性梳理第六节映射到的日频原始列（旧模型Top16来源 + 新模型族的origin来源）
USEFUL_DAILY = [
    # 资金面
    "DR001", "DR007", "R001", "R007", "1Y存单发行利率", "FR007:1年",
    "成交量:R001", "R001成交量占比",
    # 汇率/海外
    "美元指数", "中间价", "即期汇率", "2年美债", "10年美债", "标普500",
    # 利差/期货
    "1年信用利差", "3年信用利差", "5年信用利差",
    "国债期货成交量：10年", "国债期货持仓量:10年", "国债期货收盘价(活跃合约):10年",
    # 实体/商品（旧Top16与新模型Top35来源）
    "精对苯二甲酸开工率", "PTA负荷率", "玻璃市场价", "波罗的海", "水泥价格指数",
    "阴极铜价格", "COMEX金价", "ICE布油", "南华工业品指数",
    # 权益/股债
    "股债相对", "沪深300",
    # 机构行为（origin独有）
    "债券净融资", "净买入：券商自营", "净买入：保险", "净买入：基金",
    "净买入：理财", "净买入：货币基金", "十年国债活跃券成交量",
]

SHEETS = {
    "滚动稳健标准化日频": "origin2026_日频.csv",
    "滚动稳健标准化周频": "origin2026_周频.csv",
    "滚动稳健标准化月频": "origin2026_月频.csv",
    "滚动稳健标准化季频": "origin2026_季频.csv",
    "差值历史稳健标准化": "origin2026_差值.csv",
    "月频数据发布日期": "origin2026_月频发布日期.csv",
    "季频数据发布日期": "origin2026_季频发布日期.csv",
}


def main() -> None:
    xl = pd.ExcelFile(SRC)
    for sheet, fname in SHEETS.items():
        df = xl.parse(sheet, header=0, index_col=0)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.index.name = "date"
        df.to_csv(OUT / fname, encoding="utf-8-sig")
        print(f"[{sheet}] {df.shape[0]}行×{df.shape[1]}列 -> {fname}"
              f"  ({df.index.min().date()}~{df.index.max().date()})")

    # 10Y收益率
    rate = pd.read_csv(RATE_SRC)
    rate.columns = ["date", "中国国债收益率10年"]
    rate["date"] = pd.to_datetime(rate["date"])
    rate = rate.sort_values("date").set_index("date")
    rate.to_csv(OUT / "origin2026_rate.csv", encoding="utf-8-sig")
    print(f"[rate] {len(rate)}行 -> origin2026_rate.csv "
          f"({rate.index.min().date()}~{rate.index.max().date()})")

    # 有用列子集（日频）
    daily = xl.parse("滚动稳健标准化日频", header=0, index_col=0)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    daily.index.name = "date"
    missing = [c for c in USEFUL_DAILY if c not in daily.columns]
    have = [c for c in USEFUL_DAILY if c in daily.columns]
    daily[have].to_csv(OUT / "origin2026_有用列_日频.csv", encoding="utf-8-sig")
    print(f"[有用列] {len(have)}/{len(USEFUL_DAILY)}列导出 -> "
          f"origin2026_有用列_日频.csv")
    if missing:
        print("  未找到的列:", missing)


if __name__ == "__main__":
    main()
