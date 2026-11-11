# -*- coding: utf-8 -*-
"""
XGB 超参内验证调参（新库 h10）：训练窗口内选参，测试段只做最终报告

协议：
  - 每个fold的训练窗 [s-756, s) 再切内验证：内训练 [s-756, s-73)，
    内验证 [s-63, s)，中间隔10行purge（=label horizon，防标签重叠泄漏）
  - 配置得分 = 各fold内验证AUC的均值（不看测试段，避免用测试集调参）
  - 选出最优配置后，用完整窗口重训，在测试段出最终指标（与21脚本同口径）

网格: depth{2,3,4} × lr{0.03,0.05,0.1}，其余超参沿用现行（n=600, mcw=8等）
输出: data_new/out/xgbtune_inner_val.csv + xgbtune_final_metrics.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data_new"
OUT = DATA / "out"
OUT.mkdir(exist_ok=True)

START_DATE = "2018-01-01"
TRAIN_WINDOW = 756
STEP = 63
INNER_VAL = 63          # 内验证段长度（交易日）
PURGE = 10              # 内训练/内验证间隔（=horizon，防标签重叠）

GRID = [(d, lr) for d in (2, 3, 4) for lr in (0.03, 0.05, 0.1)]
BASE_PARAMS = dict(
    n_estimators=600, min_child_weight=8.0, reg_alpha=0.5, reg_lambda=2.0,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
    random_state=42, eval_metric="logloss", n_jobs=4,
)


def load():
    feats = pd.read_csv(DATA / "43v2_features.csv", index_col=0,
                        parse_dates=True)
    lab = pd.read_csv(DATA / "43v2_labels_bin_h10.csv", index_col=0,
                      parse_dates=True)
    df = feats.join(lab, how="inner")
    df = df.loc[df["label"].notna() & df["sample_weight"].notna()]
    return df, list(feats.columns)


def main():
    df, feat_cols = load()
    test_idx_all = df.index[df.index >= START_DATE]
    folds = [test_idx_all[i: i + STEP] for i in range(0, len(test_idx_all), STEP)]
    print(f"样本{len(df)} 特征{len(feat_cols)} folds={len(folds)}")

    # ---- 阶段1：内验证网格 ----
    rows = []
    for depth, lr in GRID:
        val_aucs = []
        for seg in folds:
            pos = df.index.get_indexer(seg)
            tr_lo = max(0, pos[0] - TRAIN_WINDOW)
            cut = pos[0] - INNER_VAL
            itr = np.arange(tr_lo, max(cut - PURGE, tr_lo + 100))
            iva = np.arange(cut, pos[0])
            if len(iva) < 30:
                continue
            m = xgb.XGBClassifier(max_depth=depth, learning_rate=lr,
                                  **BASE_PARAMS)
            m.fit(df.iloc[itr][feat_cols], df["label"].iloc[itr],
                  sample_weight=df["sample_weight"].iloc[itr])
            p = m.predict_proba(df.iloc[iva][feat_cols])[:, 1]
            try:
                val_aucs.append(roc_auc_score(df["label"].iloc[iva], p))
            except ValueError:
                pass
        rows.append({"depth": depth, "lr": lr,
                     "val_auc_mean": np.mean(val_aucs),
                     "val_auc_std": np.std(val_aucs),
                     "n_val_folds": len(val_aucs)})
        print(f"  depth={depth} lr={lr}: 内验证AUC "
              f"{rows[-1]['val_auc_mean']:.4f} ±{rows[-1]['val_auc_std']:.4f}",
              flush=True)
    tune = pd.DataFrame(rows).sort_values("val_auc_mean", ascending=False)
    tune.to_csv(OUT / "xgbtune_inner_val.csv", index=False,
                encoding="utf-8-sig")
    best = tune.iloc[0]
    print(f"\n内验证最优: depth={int(best.depth)} lr={best.lr} "
          f"(val AUC {best.val_auc_mean:.4f})")

    # ---- 阶段2：最优配置完整窗口最终评估 ----
    for tag, depth, lr in [("默认(depth3,lr0.05)", 3, 0.05),
                           ("内验最优", int(best.depth), float(best.lr))]:
        preds, fms = [], []
        for k, seg in enumerate(folds, 1):
            pos = df.index.get_indexer(seg)
            tr_idx = np.arange(max(0, pos[0] - TRAIN_WINDOW), pos[0])
            if len(tr_idx) < 300:
                continue
            m = xgb.XGBClassifier(max_depth=depth, learning_rate=lr,
                                  **BASE_PARAMS)
            m.fit(df.iloc[tr_idx][feat_cols], df["label"].iloc[tr_idx],
                  sample_weight=df["sample_weight"].iloc[tr_idx])
            proba = m.predict_proba(df.iloc[pos][feat_cols])[:, 1]
            pred = (proba >= 0.5).astype(int)
            yt = df["label"].iloc[pos].values
            try:
                auc = roc_auc_score(yt, proba)
            except ValueError:
                auc = np.nan
            fms.append({"fold": k, "date_start": seg[0].date().isoformat(),
                        "date_end": seg[-1].date().isoformat(),
                        "accuracy": accuracy_score(yt, pred),
                        "auc": auc})
            for d, p, pb, yy in zip(seg, pred, proba, yt):
                preds.append({"date": d.date().isoformat(), "y_true": int(yy),
                              "proba_down": float(pb), "pred": int(p)})
        pr = pd.DataFrame(preds)
        g_acc = accuracy_score(pr["y_true"], pr["pred"])
        g_f1 = f1_score(pr["y_true"], pr["pred"], average="macro",
                        zero_division=0)
        g_auc = roc_auc_score(pr["y_true"], pr["proba_down"])
        pr["year"] = pd.to_datetime(pr["date"]).dt.year
        y25 = pr[pr.year == 2025]
        y26 = pr[pr.year == 2026]
        print(f"\n[{tag}] depth={depth} lr={lr}")
        print(f"  全局: Acc={g_acc:.4f} F1={g_f1:.4f} AUC={g_auc:.4f}")
        print(f"  2025: Acc={accuracy_score(y25.y_true, y25.pred):.4f} "
              f"AUC={roc_auc_score(y25.y_true, y25.proba_down):.4f}")
        print(f"  2026: Acc={accuracy_score(y26.y_true, y26.pred):.4f} "
              f"AUC={roc_auc_score(y26.y_true, y26.proba_down):.4f}")
        pd.DataFrame(fms).to_csv(
            OUT / f"xgbtune_final_{tag.split('(')[0].strip()}_folds.csv",
            index=False, encoding="utf-8-sig")
        pr.to_csv(
            OUT / f"xgbtune_final_{tag.split('(')[0].strip()}_preds.csv",
            index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
