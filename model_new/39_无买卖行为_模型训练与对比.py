# -*- coding: utf-8 -*-
"""
无买卖行为数据（43v4nb）模型训练与对比：量化机构净买入的边际价值

流程（单文件端到端）:
  1. XGB 训练（21脚本同协议: 窗口756/步长63/测试2018起）: h10 + h20
  2. 逐fold训练窗重要性表（34脚本同逻辑，供NN逐fold选特征）
  3. BiGRU NN_B 训练（subprocess调17脚本）: 全局Top30 + 逐fold Top30
  4. 对比评估: 无净买入(nb) vs 含净买入(43v4) 的 XGB/NN/融合/2026逐月

输出:
  data_new/out/xgbv4nbh10_* / xgbv4nbh20_* / xgbv4nbh10_fold_importance.csv
  model_nn/out/NNV4NBH10{B,PF}_*
  data_new/out/39_无买卖行为对比.txt
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data_new"
OUT = DATA / "out"
NN_DIR = BASE / "model_nn"
NN_OUT = NN_DIR / "out"
NN17 = NN_DIR / "17_BiGRU训练_新基础数据_二分类.py"

FEATS_NB = "43v4nb_features.csv"
START_DATE = "2018-01-01"
TRAIN_WINDOW = 756
STEP = 63
RANDOM_STATE = 42
PARAMS = dict(max_depth=3, n_estimators=600, learning_rate=0.05,
              min_child_weight=8.0, reg_alpha=0.5, reg_lambda=2.0,
              subsample=0.8, colsample_bytree=0.8, tree_method="hist",
              random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=4)

L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    L.append(s)
    print(s, flush=True)


# ---------------- 1. XGB（21脚本核心协议） ----------------
def train_xgb(feats_file: str, labels_file: str, out_prefix: str) -> None:
    feats = pd.read_csv(DATA / feats_file, index_col=0, parse_dates=True)
    lab = pd.read_csv(DATA / labels_file, index_col=0, parse_dates=True)
    df = feats.join(lab, how="inner")
    df = df.loc[df["label"].notna() & df["sample_weight"].notna()]
    feat_cols = list(feats.columns)
    X, y, w = df[feat_cols], df["label"].astype(int), df["sample_weight"]

    test_idx_all = df.index[df.index >= START_DATE]
    folds, i = [], 0
    while i < len(test_idx_all):
        folds.append(test_idx_all[i: i + STEP])
        i += STEP

    rows, preds, imps = [], [], []
    for k, seg in enumerate(folds, 1):
        pos = df.index.get_indexer(seg)
        tr_idx = np.arange(max(0, pos[0] - TRAIN_WINDOW), pos[0])
        if len(tr_idx) < 300:
            continue
        model = xgb.XGBClassifier(**PARAMS)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=w.iloc[tr_idx])
        proba = model.predict_proba(X.iloc[pos])[:, 1]
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
                                          zero_division=0), "auc": auc})
        preds += [{"date": d.date().isoformat(), "y_true": int(yy),
                   "proba_down": float(pb), "pred": int(pp)}
                  for d, pp, pb, yy in zip(seg, pred, proba, yt)]
        imps.append(pd.Series(model.feature_importances_, index=feat_cols))
        if k % 8 == 0:
            P(f"  [{out_prefix}] fold {k}/{len(folds)}")

    pr = pd.DataFrame(preds)
    imp = pd.concat(imps, axis=1)
    imp_out = pd.DataFrame({
        "feature": imp.index, "gain_mean": imp.mean(axis=1).values,
        "gain_std": imp.std(axis=1).values,
        "appear_ratio": (imp > 0).mean(axis=1).values,
    }).sort_values("gain_mean", ascending=False)
    g_acc = accuracy_score(pr["y_true"], pr["pred"])
    g_auc = roc_auc_score(pr["y_true"], pr["proba_down"])
    P(f"[XGB {out_prefix}] 全局 Acc={g_acc:.4f} AUC={g_auc:.4f} "
      f"样本={len(pr)} 特征={len(feat_cols)}")
    pd.DataFrame(rows).to_csv(OUT / f"{out_prefix}_fold_metrics.csv",
                              index=False, encoding="utf-8-sig")
    pr.to_csv(OUT / f"{out_prefix}_test_predictions.csv", index=False,
              encoding="utf-8-sig")
    imp_out.to_csv(OUT / f"{out_prefix}_feature_importance.csv", index=False,
                   encoding="utf-8-sig")


# ---------------- 2. 逐fold训练窗重要性（34脚本同逻辑） ----------------
def fold_importance(feats_file: str, labels_file: str, folds_file: str,
                    out_file: str) -> None:
    feats = pd.read_csv(DATA / feats_file, index_col=0, parse_dates=True)
    lab = pd.read_csv(DATA / labels_file, index_col=0, parse_dates=True)
    df = feats.join(lab, how="inner")
    df = df.loc[df["label"].notna() & df["sample_weight"].notna()]
    feat_cols = list(feats.columns)
    X, y, w = df[feat_cols], df["label"].astype(int), df["sample_weight"]
    fm = pd.read_csv(OUT / folds_file)
    rows = []
    for _, fd in fm.iterrows():
        pos = df.index.searchsorted(pd.Timestamp(fd["date_start"]))
        tr_idx = np.arange(max(0, pos - 1260), pos)
        if len(tr_idx) < 300:
            continue
        m = xgb.XGBClassifier(**{**PARAMS, "n_estimators": 300})
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=w.iloc[tr_idx])
        rows += [{"fold": int(fd["fold"]), "feature": f, "gain": float(g)}
                 for f, g in zip(feat_cols, m.feature_importances_)]
    pd.DataFrame(rows).to_csv(OUT / out_file, index=False,
                              encoding="utf-8-sig")
    P(f"[fold-importance] {out_file}")


# ---------------- 3. NN（subprocess 调 17 脚本） ----------------
def run_nn(tag: str, extra: list[str]) -> None:
    cmd = [sys.executable, "-X", "utf8", str(NN17),
           "--features-file", FEATS_NB,
           "--labels-file", "43v3_labels_bin_h10.csv",
           "--folds-file", "xgbv4nbh10_fold_metrics.csv",
           "--hidden-size", "64", "--lr", "5e-4", "--tag", tag] + extra
    log = (NN_OUT / f"{tag}_run.log").open("w", encoding="utf-8")
    P(f"[NN] {tag}: {' '.join(extra)}")
    rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    log.close()
    if rc != 0:
        P(f"[NN] {tag} 失败 rc={rc}，见 {NN_OUT / (tag + '_run.log')}")


# ---------------- 4. 对比评估 ----------------
def load(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["date"] = pd.to_datetime(d["date"])
    return d


def fuse(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    na = a[["date", "y_true"]].assign(p=a["proba_down"])
    nb = b[["date", "y_true"]].assign(p=b["proba_down"])
    m = na.merge(nb, on="date", suffixes=("_a", "_b"))
    m["y_true"] = m["y_true_a"]
    m["proba_down"] = (m["p_a"] + m["p_b"]) / 2
    m["pred"] = (m["proba_down"] >= 0.5).astype(int)
    return m


def show(name: str, df: pd.DataFrame) -> None:
    def seg(g):
        y = g["y_true"]
        auc = roc_auc_score(y, g["proba_down"]) if y.nunique() > 1 else float("nan")
        return (f"Acc={accuracy_score(y, g['pred']):.4f} AUC={auc:.4f} "
                f"基线={max(y.mean(), 1 - y.mean()):.4f}")
    P(f"{name:<32} 全局 n={len(df):<5} {seg(df)}")
    for yr in (2025, 2026):
        g = df[pd.DatetimeIndex(df["date"]).year == yr]
        if len(g):
            P(f"{'':<32} {yr}  n={len(g):<4} {seg(g)}")


def evaluate() -> None:
    P("\n" + "=" * 84)
    P("对比评估: 无买卖行为(nb) vs 含买卖行为(43v4)")
    P("=" * 84)
    pairs = [
        ("XGB h10", "xgbv4nbh10", "xgbv4h10"),
        ("XGB h20", "xgbv4nbh20", "xgbv4h20"),
        ("NN 全局Top30", "NNV4NBH10B", "NNV4H10B"),
        ("NN 逐foldTop30", "NNV4NBH10PF", "NNV4H10PF"),
    ]
    for name, nb_tag, v4_tag in pairs:
        P(f"\n[{name}]")
        nb = load(OUT / f"{nb_tag}_test_predictions.csv") \
            if not nb_tag.startswith("NNV") else \
            load(NN_OUT / f"{nb_tag}_nnv2_test_predictions.csv")
        v4 = load(OUT / f"{v4_tag}_test_predictions.csv") \
            if not v4_tag.startswith("NNV") else \
            load(NN_OUT / f"{v4_tag}_nnv2_test_predictions.csv")
        if nb is not None:
            show(f"{name} 无净买入", nb)
        if v4 is not None:
            show(f"{name} 含净买入", v4)

    P("\n[融合 h10: XGB+NN全局K30 / XGB+NN逐fold]")
    nbx = load(OUT / "xgbv4nbh10_test_predictions.csv")
    nng = load(NN_OUT / "NNV4NBH10B_nnv2_test_predictions.csv")
    nnp = load(NN_OUT / "NNV4NBH10PF_nnv2_test_predictions.csv")
    if nbx is not None and nng is not None:
        show("融合(nb) XGB+NNK30", fuse(nbx, nng))
    if nbx is not None and nnp is not None:
        show("融合(nb) XGB+NN逐fold", fuse(nbx, nnp))

    P("\n[2026年逐月Acc: XGB h20 无净买入 vs 含净买入]")
    v4h20 = load(OUT / "xgbv4h20_test_predictions.csv")
    for name, df in [("XGBh20无净买入", load(
            OUT / "xgbv4nbh20_test_predictions.csv")),
            ("XGBh20含净买入", v4h20)]:
        if df is None:
            continue
        g = df[pd.DatetimeIndex(df["date"]) >= "2026-01-01"].copy()
        parts = [f"{mo}:{accuracy_score(x['y_true'], x['pred']):.2f}"
                 for mo, x in g.groupby(
                     pd.DatetimeIndex(g["date"]).strftime("%m"))]
        P(f"  {name:<16} " + " ".join(parts))

    (OUT / "39_无买卖行为对比.txt").write_text("\n".join(L), encoding="utf-8")
    P(f"\n输出: {OUT / '39_无买卖行为对比.txt'}")


def main() -> None:
    P("=" * 84)
    P("无买卖行为数据(43v4nb)端到端训练: XGB h10/h20 + NN(B/PF) + 对比")
    P("=" * 84)

    P("\n[1/4] XGB 训练 ...")
    train_xgb(FEATS_NB, "43v3_labels_bin_h10.csv", "xgbv4nbh10")
    train_xgb(FEATS_NB, "43v3_labels_bin_h20.csv", "xgbv4nbh20")

    P("\n[2/4] 逐fold训练窗重要性 ...")
    fold_importance(FEATS_NB, "43v3_labels_bin_h10.csv",
                    "xgbv4nbh10_fold_metrics.csv",
                    "xgbv4nbh10_fold_importance.csv")

    P("\n[3/4] BiGRU NN_B ...")
    run_nn("NNV4NBH10B", ["--importance-file",
                          "xgbv4nbh10_feature_importance.csv"])
    run_nn("NNV4NBH10PF", ["--fold-importance-file",
                           "xgbv4nbh10_fold_importance.csv"])

    P("\n[4/4] 对比评估 ...")
    evaluate()


if __name__ == "__main__":
    main()
