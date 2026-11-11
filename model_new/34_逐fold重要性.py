# -*- coding: utf-8 -*-
"""
逐 fold 训练窗内 XGB 特征重要性表（供 17 脚本 --fold-importance-file 使用）

动机：17 脚本原用全测试段聚合的 XGB 重要性选 NN Top-K 特征，重要性本身
含测试期信息（轻微选择泄漏）且方差大（v4 上 NN 因此反而变差）。本脚本
对每个 fold 只用其训练窗（与 NN 相同的 max_train_days=1260）拟合 XGB，
产出该 fold 自己的因果重要性排名。

输出: data_new/out/xgbv4h10_fold_importance.csv  (fold, feature, gain)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data_new"
OUT = DATA / "out"

FEATS = "43v4_features_h10.csv"
LABELS = "43v3_labels_bin_h10.csv"
FOLDS = "xgbv4h10_fold_metrics.csv"
MAX_TRAIN_DAYS = 1260          # 与 17 脚本 NN 默认训练窗一致

PARAMS = dict(max_depth=3, n_estimators=300, learning_rate=0.05,
              min_child_weight=8.0, reg_alpha=0.5, reg_lambda=2.0,
              subsample=0.8, colsample_bytree=0.8, tree_method="hist",
              random_state=42, eval_metric="logloss", n_jobs=4)


def main() -> None:
    feats = pd.read_csv(DATA / FEATS, index_col=0, parse_dates=True)
    lab = pd.read_csv(DATA / LABELS, index_col=0, parse_dates=True)
    df = feats.join(lab, how="inner")
    df = df.loc[df["label"].notna() & df["sample_weight"].notna()]
    feat_cols = list(feats.columns)
    X, y, w = df[feat_cols], df["label"].astype(int), df["sample_weight"]
    fm = pd.read_csv(OUT / FOLDS)
    fm["date_start"] = pd.to_datetime(fm["date_start"])

    rows = []
    for _, fd in fm.iterrows():
        pos = df.index.searchsorted(fd["date_start"])
        tr_lo = max(0, pos - MAX_TRAIN_DAYS)
        tr_idx = np.arange(tr_lo, pos)
        if len(tr_idx) < 300:
            continue
        m = xgb.XGBClassifier(**PARAMS)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=w.iloc[tr_idx])
        for f, g in zip(feat_cols, m.feature_importances_):
            rows.append({"fold": int(fd["fold"]), "feature": f,
                         "gain": float(g)})
        print(f"  fold {int(fd['fold'])} 训练窗[{df.index[tr_lo].date()}~"
              f"{df.index[pos - 1].date()}] n={len(tr_idx)}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "xgbv4h10_fold_importance.csv", index=False,
               encoding="utf-8-sig")
    print(f"输出: {OUT / 'xgbv4h10_fold_importance.csv'}  "
          f"{out['fold'].nunique()} folds × {len(feat_cols)} 特征")


if __name__ == "__main__":
    main()
