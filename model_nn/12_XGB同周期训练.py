# -*- coding: utf-8 -*-
"""
12_XGB同周期训练.py
======================================================================
P1：与 NN 完全同日历、同折边界、同标签(down_ex2)、同样本切分协议的
XGBoost 二分类，为概率融合提供异质模型。

协议（与 09_方向优化实验.run_task 逐条对齐）:
  - build_dataset(variant, horizon=10, history_start=2010-01-01)
  - 每折: train = idx[:-(n_val+gap)], val = idx[-n_val:], 测试 = 折内交易日
  - purge gap = max(seq_len, horizon); val 首日前标签必须已成熟
  - 特征不做标准化（树模型不需要，NaN 由 XGBoost 原生处理）
输出:
  out/OPTX_xgbE_predictions.csv / OPTX_xgbF_predictions.csv + _metrics.json
运行: & "D:\\soft\\Anaconda3\\python.exe" -u -B "12_XGB同周期训练.py"
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE / "_libs"))
import xgboost as xgb
OUT = HERE / "out"
CUTOFF = pd.Timestamp("2024-12-31")
DEAD_BP = 2.0


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp9 = load_module("exp9", HERE / "09_方向优化实验.py")
core = exp9.core


def seg_metrics(y, pred_label, proba):
    y = np.asarray(y); pred_label = np.asarray(pred_label)
    m = {"n": len(y), "acc": float(accuracy_score(y, pred_label)),
         "macro_f1": float(f1_score(y, pred_label, average="macro", zero_division=0)),
         "pred_down": float(pred_label.mean()), "true_down": float(y.mean()),
         "majority_baseline": float(max(y.mean(), 1 - y.mean()))}
    m["auc"] = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan")
    m["vs_majority"] = m["acc"] - m["majority_baseline"]
    return m


def run_variant(variant, args, folds):
    frame, delta, dates, maturity = exp9.build_dataset(
        variant, args.horizon, args.history_start)
    y = (delta < -DEAD_BP / 100.0).astype(int)
    X = frame.to_numpy(dtype=float)
    tag = f"OPTX_xgb{variant}"
    print(f"[{tag}] {len(y)}行 {X.shape[1]}列 "
          f"{dates.min().date()}~{dates.max().date()}", flush=True)

    predictions = []
    for _, fd in folds.iterrows():
        s = dates.searchsorted(fd["date_start"])
        e = dates.searchsorted(fd["date_end"], side="right")
        if s >= len(y) or e <= s:
            continue
        hi = int(maturity.searchsorted(dates.iloc[s], side="left")) - 1
        idx = core.seq_indices(len(y), 0, hi, args.seq_len)
        n_val = max(8, int(len(idx) * args.val_rate))
        gap = max(args.seq_len, args.horizon)
        train_idx = idx[:-(n_val + gap)]
        va_idx = idx[-n_val:]
        assert maturity.iloc[train_idx[-1]] < dates.iloc[va_idx[0]]
        assert maturity.iloc[hi] < dates.iloc[s]
        end_idx = np.arange(max(s, args.seq_len - 1), e)

        clf = xgb.XGBClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.learning_rate, min_child_weight=args.min_child_weight,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda, gamma=0.1,
            objective="binary:logistic", eval_metric="auc",
            early_stopping_rounds=args.early_stopping,
            tree_method="hist", n_jobs=4, random_state=42)
        clf.fit(X[train_idx], y[train_idx], eval_set=[(X[va_idx], y[va_idx])], verbose=False)
        p = clf.predict_proba(X[end_idx])[:, 1]
        for j, row in enumerate(end_idx):
            predictions.append({
                "date": str(dates.iloc[row].date()), "y_true": int(y[row]),
                "proba_down": float(p[j]), "pred": int(p[j] >= 0.5),
                "threshold": 0.5, "fold": int(fd["fold"]),
                "seed0": float(p[j]),
                "best_iter": int(clf.best_iteration) if hasattr(clf, "best_iteration") else -1})
        print(f"[{tag}] fold={fd['fold']} rows={len(train_idx)} "
              f"best_iter={predictions[-1]['best_iter']}", flush=True)
    pr = pd.DataFrame(predictions)
    pr["date"] = pd.to_datetime(pr["date"])
    pr = pr.sort_values("date").reset_index(drop=True)
    pr.to_csv(OUT / f"{tag}_predictions.csv", index=False, encoding="utf-8-sig")

    mat_map = dict(zip(dates, maturity))
    pr["maturity"] = pr["date"].map(mat_map)
    dev_m = ((pr["date"] <= CUTOFF) & (pr["maturity"] <= CUTOFF)).to_numpy()
    m25 = (pr["date"].dt.year == 2025).to_numpy()
    yv = pr["y_true"].to_numpy(); pv = pr["proba_down"].to_numpy()
    predv = (pv >= 0.5).astype(int)
    # 原始方向口径
    rate = pd.read_csv(HERE.parent / "data_ori" / "2026_origin_rate.csv").iloc[:, :2]
    rate.columns = ["date", "rate"]
    rate["date"] = pd.to_datetime(rate["date"])
    rate["delta"] = pd.to_numeric(rate["rate"], errors="coerce").shift(-args.horizon) \
        - pd.to_numeric(rate["rate"], errors="coerce")
    ydir = (pr["date"].map(rate.set_index("date")["delta"]).to_numpy() < 0).astype(int)
    report = {
        "task": tag, "variant": variant, "label": "down_ex2",
        "features": int(X.shape[1]),
        "params": {k: getattr(args, k) for k in
                   ("n_estimators", "max_depth", "learning_rate", "min_child_weight",
                    "reg_alpha", "reg_lambda", "early_stopping")},
        "dev": seg_metrics(yv[dev_m], predv[dev_m], pv[dev_m]),
        "full": seg_metrics(yv, predv, pv),
        "2025": seg_metrics(yv[m25], predv[m25], pv[m25]),
        "original_dev": seg_metrics(ydir[dev_m], predv[dev_m], pv[dev_m]),
        "original_2025": seg_metrics(ydir[m25], predv[m25], pv[m25]),
    }
    (OUT / f"{tag}_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for seg in ("dev", "full", "2025", "original_dev", "original_2025"):
        m = report[seg]
        print(f"[{tag}] {seg:14s} Acc={m['acc']:.4f}(vs多数类{m['vs_majority']:+.4f}) "
              f"AUC={m['auc']:.4f} n={m['n']}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=40)
    ap.add_argument("--val-rate", type=float, default=0.2)
    ap.add_argument("--history-start", default="2010-01-01")
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-child-weight", type=float, default=8.0)
    ap.add_argument("--reg-alpha", type=float, default=1.0)
    ap.add_argument("--reg-lambda", type=float, default=2.0)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--variants", default="E,F")
    args = ap.parse_args()

    cfg = core.parse_args()
    folds = core.load_folds(cfg)
    all_reports = {}
    for v in args.variants.split(","):
        all_reports[f"xgb{v}"] = run_variant(v.strip(), args, folds)
    (OUT / "OPTX_对比汇总.json").write_text(
        json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
