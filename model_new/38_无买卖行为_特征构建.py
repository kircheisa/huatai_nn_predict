# -*- coding: utf-8 -*-
"""
无买卖行为数据的特征构建（消融：剔除5类机构现券净买入）

模拟"数据源中没有净买入原始列"的场景：
  排除原始列: 净买入：券商自营 / 保险 / 基金 / 理财 / 货币基金
  连带排除特征族:
    F5 净买入流量族   净买入：X__{z,sum5_z,sum20_z,sum5_chg5}  (20列)
    F9 机构深化族     机构__{交易盘净买入,配置盘净买入,交易配置差额,分歧度,
                     corr_基金净买入利率20}                  (5列)

由于净买入列在 30/32 管线中只进入上述两族（交易日网格、F1-F4、F6-F8、
F10-F13 均与其无关），从 43v4 中精确删除这 25 列 == 无该数据重建的完整结果，
且保证与含净买入版（43v4）唯一差异就是买卖行为，构成严格受控对比。

输出: data_new/43v4nb_features.csv（371列，index与43v4一致；三个horizon共用，
      标签复用 43v3_labels_bin{,_h10,_h15,_h20}.csv）
      data_new/43v4nb_build_meta.txt
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "data_ori" / "2026_ori_data_new.xlsx"
DATA = BASE / "data_new"

NETBUY_RAW = ["净买入：券商自营", "净买入：保险", "净买入：基金",
              "净买入：理财", "净买入：货币基金"]


def main() -> None:
    # 确认原始数据中确实存在这5列（仅为校验来源定义，不参与构建）
    raw = pd.read_excel(SRC, sheet_name="滚动稳健标准化日频+买卖行为",
                        header=None, nrows=2)
    header = [str(x) for x in raw.iloc[0, 1:]]
    missing = [c for c in NETBUY_RAW if c not in header]
    assert not missing, f"原始表缺少预期净买入列: {missing}"

    f4 = pd.read_csv(DATA / "43v4_features.csv", index_col=0, parse_dates=True)
    drop = [c for c in f4.columns
            if "净买入" in c or c.startswith("机构__")]
    # 城投债/地方债净融资（"净融资"）不属于净买入，必须保留
    keep_flow = [c for c in drop if "净融资" in c]
    assert not keep_flow, f"误伤净融资列: {keep_flow}"
    out = f4.drop(columns=drop)
    out.to_csv(DATA / "43v4nb_features.csv", encoding="utf-8-sig")

    lines = ["=" * 60,
             f"无买卖行为特征 43v4nb: {out.shape[1]}列 "
             f"(= 43v4的{f4.shape[1]}列 - 剔除{len(drop)}列)",
             f"网格 {out.index.min().date()}~{out.index.max().date()} "
             f"{len(out)}行",
             "剔除清单:"]
    lines += [f"  - {c}" for c in drop]
    lines += ["标签复用 43v3_labels_bin{,_h10,_h15,_h20}.csv",
              "=" * 60]
    report = "\n".join(lines)
    print(report)
    (DATA / "43v4nb_build_meta.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
