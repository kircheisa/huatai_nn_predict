# -*- coding: utf-8 -*-
"""
新基础数据（data_new/43v2_*.csv）XGB 二分类训练：10Y收益率未来5日方向（下行=1）

评估协议与旧管线（03_XGB训练_二分类.py）对齐：
  - 滚动窗口: 窗口756交易日(约3年)，步长63交易日(约季度)重训
  - 测试段: 2018-01-01 起（与旧基线同期可比）
  - 死区: |Δ|<2bp 样本权重0.3（训练加权，评估全量）
  - 超参: depth=3, n_estimators=600, lr=0.05, min_child_weight=8,
          reg_alpha=0.5, reg_lambda=2.0（旧管线"原始正则"档）

输出（data_new/out/）:
  xgbv2_fold_metrics.csv      逐fold指标
  xgbv2_test_predictions.csv  逐日OOS预测
  xgbv2_feature_importance.csv 重要性(逐foldgain聚合) + 稳定性
  xgbv2_metrics.txt           全局/分年汇总（含旧基线对照）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data_new"
OUT = DATA / "out"
OUT.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features-file", default="43v2_features.csv")
    p.add_argument("--labels-file", default="43v2_labels_bin.csv")
    p.add_argument("--out-prefix", default="xgbv2")
    # 超参覆盖（0=沿用PARAMS默认；由内验选参后传入）
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.0)
    # 类权重：0=关；-1=自动(逐fold 负类/正类)；>0显式
    p.add_argument("--scale-pos-weight", type=float, default=0.0)
    # 死区权重：-1=沿用标签文件；>=0 覆盖（|Δ|<2bp 样本的训练权重）
    p.add_argument("--deadzone-weight", type=float, default=-1.0)
    # seed-bagging：>1 时逐fold训练多个seed平均概率
    p.add_argument("--n-seeds", type=int, default=1)
    # 协议修复：训练窗尾部剔除 horizon 行（其标签跨入测试期，部署时不可得）
    p.add_argument("--purge", type=int, default=0,
                   help="训练窗尾部剔除样本数（建议=horizon）")
    # 近期样本半衰期加权（年）；>0 时训练权重 x 0.5^(age/HL)
    p.add_argument("--recency-hl", type=float, default=0.0,
                   help="近期样本半衰期（年），0=不加权")
    return p.parse_args()

START_DATE = "2018-01-01"      # 测试段起点（训练窗=起点前756交易日）
TRAIN_WINDOW = 756             # 滚动训练窗口（交易日）
STEP = 63                      # 重训步长（交易日）
RANDOM_STATE = 42

PARAMS = dict(
    max_depth=3, n_estimators=600, learning_rate=0.05,
    min_child_weight=8.0, reg_alpha=0.5, reg_lambda=2.0,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", random_state=RANDOM_STATE,
    eval_metric="logloss", n_jobs=4,
)


def main() -> None:
    args = parse_args()
    feats = pd.read_csv(DATA / args.features_file, index_col=0,
                        parse_dates=True)
    lab = pd.read_csv(DATA / args.labels_file, index_col=0,
                      parse_dates=True)
    df = feats.join(lab, how="inner")
    df = df.loc[df["label"].notna() & df["sample_weight"].notna()]
    # 死区权重覆盖（按标签文件里的原始delta重新计算）
    if args.deadzone_weight >= 0:
        dz = (df["rate_delta_5d"].abs() * 100 < 2.0).fillna(False)
        df["sample_weight"] = np.where(dz, args.deadzone_weight, 1.0)
    feat_cols = [c for c in feats.columns]
    X, y, w = df[feat_cols], df["label"].astype(int), df["sample_weight"]

    params = dict(PARAMS)
    if args.depth:
        params["max_depth"] = args.depth
    if args.lr:
        params["learning_rate"] = args.lr

    # 测试fold切分：2018起，每STEP个交易日为一个测试段
    test_idx_all = df.index[df.index >= START_DATE]
    folds = []
    i = 0
    while i < len(test_idx_all):
        seg = test_idx_all[i: i + STEP]
        folds.append(seg)
        i += STEP
    print(f"样本 {len(df)} | 特征 {len(feat_cols)} | 测试fold {len(folds)} "
          f"({folds[0][0].date()} ~ {folds[-1][-1].date()})")

    rows, preds, imps = [], [], []
    for k, seg in enumerate(folds, 1):
        pos = df.index.get_indexer(seg)
        tr_lo = max(0, pos[0] - TRAIN_WINDOW)
        tr_hi = max(tr_lo + 1, pos[0] - args.purge)
        tr_idx = np.arange(tr_lo, tr_hi)
        if len(tr_idx) < 300:
            continue
        w_tr = w.iloc[tr_idx]
        if args.recency_hl > 0:
            age_d = (seg[0] - df.index[tr_idx]).days.values
            w_tr = w_tr * 0.5 ** (age_d / (365.0 * args.recency_hl))
        fold_params = dict(params)
        if args.scale_pos_weight < 0:       # 自动：逐fold按训练窗类频
            n_pos = max(int((y.iloc[tr_idx] == 1).sum()), 1)
            fold_params["scale_pos_weight"] = \
                float((y.iloc[tr_idx] == 0).sum()) / n_pos
        elif args.scale_pos_weight > 0:
            fold_params["scale_pos_weight"] = args.scale_pos_weight
        # seed-bagging：多seed训练平均概率与重要性
        proba_sum, imp_sum = None, np.zeros(len(feat_cols))
        for sd in range(max(1, args.n_seeds)):
            fp = dict(fold_params)
            fp["random_state"] = RANDOM_STATE + sd
            model = xgb.XGBClassifier(**fp)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                      sample_weight=w_tr)
            p = model.predict_proba(X.iloc[pos])[:, 1]
            proba_sum = p.copy() if proba_sum is None else proba_sum + p
            imp_sum += model.feature_importances_
        proba = proba_sum / max(1, args.n_seeds)
        pred = (proba >= 0.5).astype(int)
        yt = y.iloc[pos].values
        try:
            auc = roc_auc_score(yt, proba)
        except ValueError:
            auc = np.nan
        rows.append({"fold": k, "date_start": seg[0].date().isoformat(),
                     "date_end": seg[-1].date().isoformat(), "n_test": len(seg),
                     "accuracy": accuracy_score(yt, pred),
                     "macro_f1": f1_score(yt, pred, average="macro",
                                          zero_division=0),
                     "auc": auc})
        for d, p, pb, yy in zip(seg, pred, proba, yt):
            preds.append({"date": d.date().isoformat(), "y_true": int(yy),
                          "proba_down": float(pb), "pred": int(p)})
        gain = imp_sum / max(1, args.n_seeds)
        imps.append(pd.Series(gain, index=feat_cols))
        print(f"  fold {k:>2} [{seg[0].date()}~{seg[-1].date()}] "
              f"acc={rows[-1]['accuracy']:.4f} auc={auc:.4f}", flush=True)

    fm = pd.DataFrame(rows)
    pr = pd.DataFrame(preds)
    imp = pd.concat(imps, axis=1)
    imp_out = pd.DataFrame({
        "feature": imp.index,
        "gain_mean": imp.mean(axis=1).values,
        "gain_std": imp.std(axis=1).values,
        "appear_ratio": (imp > 0).mean(axis=1).values,
    }).sort_values("gain_mean", ascending=False)

    g_acc = accuracy_score(pr["y_true"], pr["pred"])
    g_f1 = f1_score(pr["y_true"], pr["pred"], average="macro",
                    zero_division=0)
    g_auc = roc_auc_score(pr["y_true"], pr["proba_down"])
    maj = max(pr["y_true"].mean(), 1 - pr["y_true"].mean())
    pr["year"] = pd.to_datetime(pr["date"]).dt.year
    year_stats = pr.groupby("year").apply(
        lambda g: pd.Series({
            "n": len(g),
            "acc": accuracy_score(g["y_true"], g["pred"]),
            "auc": roc_auc_score(g["y_true"], g["proba_down"])
            if g["y_true"].nunique() > 1 else np.nan,
        }), include_groups=False)

    lines = [
        "=" * 60,
        f"新基础数据XGB[{args.out_prefix}]（{len(feat_cols)}特征 | "
        f"窗口{TRAIN_WINDOW} | 步长{STEP} | depth={params.get('max_depth')} "
        f"lr={params.get('learning_rate')} | spw={args.scale_pos_weight} "
        f"dz={args.deadzone_weight} | seeds={max(1, args.n_seeds)}）",
        "-" * 60,
        f"全局: Acc={g_acc:.4f}  MacroF1={g_f1:.4f}  AUC={g_auc:.4f}",
        f"多数类基线: {maj:.4f}  (下行占比 {pr['y_true'].mean():.4f})",
        "参考: 5天版XGB Acc=0.5236 AUC=0.5277 | 旧数据XGB 0.5635/0.5822",
        "-" * 60,
        "分年:",
        year_stats.round(4).to_string(),
        "-" * 60,
        "Top20特征(gain):",
        imp_out.head(20)[["feature", "gain_mean", "appear_ratio"]]
        .to_string(index=False),
        "-" * 60,
        f"folds={len(fm)}  测试样本={len(pr)}",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print(report)
    fm.to_csv(OUT / f"{args.out_prefix}_fold_metrics.csv", index=False,
              encoding="utf-8-sig")
    pr.to_csv(OUT / f"{args.out_prefix}_test_predictions.csv", index=False,
              encoding="utf-8-sig")
    imp_out.to_csv(OUT / f"{args.out_prefix}_feature_importance.csv",
                   index=False, encoding="utf-8-sig")
    (OUT / f"{args.out_prefix}_metrics.txt").write_text(report,
                                                        encoding="utf-8")
    print(f"\n输出: {OUT / (args.out_prefix + '_metrics.txt')}")


if __name__ == "__main__":
    main()
