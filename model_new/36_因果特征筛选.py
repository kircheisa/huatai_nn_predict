# -*- coding: utf-8 -*-
"""
因果特征筛选：仅用测试期(2018-01)之前的样本拟合 XGB 选 Top-N 特征

动机：直接用全期重要性表筛特征会把测试期信息泄入特征选择（与 NN 全局
Top-K 同理）。本脚本只用 pre-2018 数据（2010~2017，约2000样本）拟合，
产出的子集对全部测试 fold 均因果。

输出: data_new/43v4f{150,200}_features.csv（列子集，index不变）
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data_new"
CUT = "2018-01-01"
TOPS = [150, 200]
PARAMS = dict(max_depth=3, n_estimators=600, learning_rate=0.05,
              min_child_weight=8.0, reg_alpha=0.5, reg_lambda=2.0,
              subsample=0.8, colsample_bytree=0.8, tree_method="hist",
              random_state=42, eval_metric="logloss", n_jobs=4)


def main() -> None:
    feats = pd.read_csv(DATA / "43v4_features.csv", index_col=0,
                        parse_dates=True)
    lab = pd.read_csv(DATA / "43v3_labels_bin_h10.csv", index_col=0,
                      parse_dates=True)
    df = feats.join(lab, how="inner")
    df = df.loc[df["label"].notna() & df["sample_weight"].notna()]
    pre = df[df.index < CUT]
    print(f"pre-2018 训练样本: {len(pre)}  ({pre.index.min().date()}~"
          f"{pre.index.max().date()})  下行占比 {pre['label'].mean():.4f}")
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(pre[feats.columns], pre["label"].astype(int),
          sample_weight=pre["sample_weight"])
    gain = pd.Series(m.feature_importances_, index=feats.columns)
    ranked = gain.sort_values(ascending=False)
    print("pre-2018 Top15:", dict(ranked.head(15).round(4)))
    for n in TOPS:
        cols = ranked.head(n).index.tolist()
        feats[cols].to_csv(DATA / f"43v4f{n}_features.csv",
                           encoding="utf-8-sig")
        print(f"输出 43v4f{n}_features.csv ({n}列)")


if __name__ == "__main__":
    main()
