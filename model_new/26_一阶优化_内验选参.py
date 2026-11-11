# -*- coding: utf-8 -*-
"""
一阶优化内验选参：日历特征版(cal) + h10标签上选 类权重×死区权重，深度guard

候选（depth=2, lr=0.03 固定，此前内验已选）：
  spw ∈ {0(关), -1(逐fold自动 n_neg/n_pos), 0.5} × dz ∈ {0.0, 0.3, 0.6}
  另加2个深度guard: (d3,lr0.03) 与 (d2,lr0.05)，防新特征改变深度最优

选择标准：训练窗内验证段（末63日，purge10行）的 Acc（本轮要修的就是Acc），
AUC 同时报告作护栏（选择时 AUC 相差>1.5pp 则优先 AUC）。
测试段不参与选择。

输出: data_new/out/xgb_stage1_inner_val.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data_new"
OUT = DATA / "out"

START_DATE = "2018-01-01"
TRAIN_WINDOW = 756
STEP = 63
INNER_VAL = 63
PURGE = 10
BASE_PARAMS = dict(
    n_estimators=600, min_child_weight=8.0, reg_alpha=0.5, reg_lambda=2.0,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
    random_state=42, eval_metric="logloss", n_jobs=4,
)

CONFIGS = []
for spw in (0.0, -1.0, 0.5):
    for dz in (0.0, 0.3, 0.6):
        CONFIGS.append(dict(depth=2, lr=0.03, spw=spw, dz=dz))
CONFIGS.append(dict(depth=3, lr=0.03, spw=0.0, dz=0.3))
CONFIGS.append(dict(depth=2, lr=0.05, spw=0.0, dz=0.3))


def main():
    import sys
    feats_file = sys.argv[1] if len(sys.argv) > 1 else "43v2cal_features.csv"
    feats = pd.read_csv(DATA / feats_file, index_col=0,
                        parse_dates=True)
    print(f"[26内验选参] 特征文件: {feats_file}")
    lab = pd.read_csv(DATA / "43v2_labels_bin_h10.csv", index_col=0,
                      parse_dates=True)
    df = feats.join(lab, how="inner")
    df = df.loc[df["label"].notna()]
    feat_cols = list(feats.columns)
    delta = df["rate_delta_5d"]

    test_idx_all = df.index[df.index >= START_DATE]
    folds = [test_idx_all[i: i + STEP] for i in range(0, len(test_idx_all), STEP)]
    print(f"样本{len(df)} 特征{len(feat_cols)} folds={len(folds)} "
          f"候选{len(CONFIGS)}")

    rows = []
    for ci, cfg in enumerate(CONFIGS, 1):
        accs, aucs = [], []
        for seg in folds:
            pos = df.index.get_indexer(seg)
            tr_lo = max(0, pos[0] - TRAIN_WINDOW)
            cut = pos[0] - INNER_VAL
            itr = np.arange(tr_lo, max(cut - PURGE, tr_lo + 100))
            iva = np.arange(cut, pos[0])
            if len(iva) < 30:
                continue
            ytr = df["label"].iloc[itr]
            # 死区权重按配置重算
            if cfg["dz"] >= 0:
                dtr = delta.iloc[itr]
                wtr = pd.Series(
                    np.where(dtr.abs() * 100 < 2.0, cfg["dz"], 1.0),
                    index=dtr.index)
            else:
                wtr = df["sample_weight"].iloc[itr]
            fp = dict(BASE_PARAMS)
            fp["max_depth"] = cfg["depth"]
            fp["learning_rate"] = cfg["lr"]
            if cfg["spw"] < 0:
                fp["scale_pos_weight"] = \
                    float((ytr == 0).sum()) / max(int((ytr == 1).sum()), 1)
            elif cfg["spw"] > 0:
                fp["scale_pos_weight"] = cfg["spw"]
            m = xgb.XGBClassifier(**fp)
            m.fit(df.iloc[itr][feat_cols], ytr, sample_weight=wtr)
            p = m.predict_proba(df.iloc[iva][feat_cols])[:, 1]
            yv = df["label"].iloc[iva]
            accs.append(accuracy_score(yv, (p >= 0.5).astype(int)))
            try:
                aucs.append(roc_auc_score(yv, p))
            except ValueError:
                pass
        rows.append({"cfg": f"d{cfg['depth']}_lr{cfg['lr']}_spw{cfg['spw']}"
                           f"_dz{cfg['dz']}",
                     "val_acc": np.mean(accs), "val_acc_std": np.std(accs),
                     "val_auc": np.mean(aucs)})
        print(f"  [{ci:>2}/{len(CONFIGS)}] {rows[-1]['cfg']}: "
              f"valAcc={rows[-1]['val_acc']:.4f} "
              f"valAUC={rows[-1]['val_auc']:.4f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "xgb_stage1_inner_val.csv", index=False,
               encoding="utf-8-sig")
    print("\n按内验Acc排序（AUC护栏：若榜首AUC比Acc次席低>1.5pp则换次席）:")
    print(res.sort_values("val_acc", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
