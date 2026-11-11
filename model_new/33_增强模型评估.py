# -*- coding: utf-8 -*-
"""
43v4 增强特征模型评估：43v3 vs 43v4 全配置对照 + 跨周期共识融合 + 2026探查

评估口径与 31 脚本一致（34 folds 滚动样本外 2018-01 起）。
输出: data_new/out/43v4增强评估.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data_new" / "out"
NN_OUT = BASE / "model_nn" / "out"
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    L.append(s)
    print(s)


def load_pred(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"])
    return d


def seg(g: pd.DataFrame, proba: str, pred: str) -> str:
    y = g["y_true"]
    auc = roc_auc_score(y, g[proba]) if y.nunique() > 1 else np.nan
    return (f"n={len(g):<5} Acc={accuracy_score(y, g[pred]):.4f} "
            f"AUC={auc:.4f} F1={f1_score(y, g[pred], average='macro', zero_division=0):.4f} "
            f"基线={max(y.mean(), 1 - y.mean()):.4f}")


def show(name: str, df: pd.DataFrame, proba: str, pred: str) -> None:
    P(f"{name:<26} 全局 {seg(df, proba, pred)}")
    for yr in (2025, 2026):
        g = df[pd.DatetimeIndex(df["date"]).year == yr]
        if len(g):
            P(f"{'':<26} {yr}  {seg(g, proba, pred)}")


def fuse(a: pd.DataFrame, b: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """概率平均融合；输入可为原始预测帧(proba_down)或已融合帧(proba)"""
    def norm(d: pd.DataFrame) -> pd.DataFrame:
        p = d["proba"] if "proba" in d.columns else d["proba_down"]
        return d[["date", "y_true"]].assign(p=p)
    m = norm(a).merge(norm(b), on="date", suffixes=("_a", "_b"))
    m["y_true"] = m["y_true_a"]
    m["proba"] = (m["p_a"] + m["p_b"]) / 2
    m["pred"] = (m["proba"] >= 0.5).astype(int)
    return m


def main() -> None:
    P("=" * 84)
    P("43v4 增强特征（+45列派生族）模型评估  对照: 43v3")
    P("=" * 84)

    runs = {
        "XGB h10 默认(v3)": (OUT / "xgbv3h10_test_predictions.csv", "v3"),
        "XGB h10 默认(v4)": (OUT / "xgbv4h10_test_predictions.csv", "v4"),
        "XGB h10 调参(v3)": (OUT / "xgbv3tuneh10_test_predictions.csv", "v3"),
        "XGB h10 调参(v4)": (OUT / "xgbv4tuneh10_test_predictions.csv", "v4"),
        "XGB h5 默认(v3)": (OUT / "xgbv3_test_predictions.csv", "v3"),
        "XGB h5 默认(v4)": (OUT / "xgbv4_test_predictions.csv", "v4"),
    }
    nn_runs = {
        "NN_B h10(v3)": ("NNV3H10B", "v3"),
        "NN_B h10(v4)": ("NNV4H10B", "v4"),
        "NN_B h10(v4)扩展窗": ("NNV4H10E", "v4"),
        "NN_B h5(v3)": ("NNV3H5B", "v3"),
        "NN_B h5(v4)": ("NNV4H5B", "v4"),
    }
    preds: dict[str, pd.DataFrame] = {}
    for k, (p, _) in runs.items():
        d = load_pred(p)
        if d is not None:
            preds[k] = d
    for k, (tag, _) in nn_runs.items():
        d = load_pred(NN_OUT / f"{tag}_nnv2_test_predictions.csv")
        if d is not None:
            preds[k] = d

    P("\n[单模型对照]")
    for k, d in preds.items():
        show(k, d, "proba_down", "pred")

    P("\n[同周期融合]")
    combos = [
        ("融合 XGB默认+NN h10(v4)", "XGB h10 默认(v4)", "NN_B h10(v4)"),
        ("融合 调参XGB+NN h10(v4)", "XGB h10 调参(v4)", "NN_B h10(v4)"),
        ("融合 调参XGB+NN扩展 h10(v4)", "XGB h10 调参(v4)", "NN_B h10(v4)扩展窗"),
        ("融合 XGB+NN h5(v4)", "XGB h5 默认(v4)", "NN_B h5(v4)"),
    ]
    for name, ka, kb in combos:
        if ka in preds and kb in preds:
            m = fuse(preds[ka], preds[kb], name)
            preds[name] = m
            show(name, m, "proba", "pred")

    P("\n[跨周期共识融合: h5与h10模型概率平均，按h10标签评估]")
    for name, ka, kb in [
        ("跨周期 XGB: h5+h10(v4)", "XGB h5 默认(v4)", "XGB h10 默认(v4)"),
        ("跨周期 NN: h5+h10(v4)", "NN_B h5(v4)", "NN_B h10(v4)"),
        ("跨周期 融合+融合(v4)", "融合 XGB默认+NN h10(v4)", "融合 XGB+NN h5(v4)"),
    ]:
        if ka in preds and kb in preds:
            m = fuse(preds[ka], preds[kb], name)
            preds[name] = m
            show(name, m, "proba", "pred")

    P("\n[新特征被采用情况(v4 h10 XGB重要性)]")
    imp_path = OUT / "xgbv4h10_feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        base_cols = set(pd.read_csv(OUT.parent / "43v3_features.csv",
                                    index_col=0, nrows=2).columns)
        new_imp = imp[~imp["feature"].isin(base_cols)].sort_values(
            "gain_mean", ascending=False)
        P(f"新增45列中进重要性表 {len(new_imp)} 列; Top12:")
        P(new_imp.head(12).to_string(index=False))

    # 2026 逐月：最优两三个配置
    P("\n[2026年逐月] (Acc, 关键配置)")
    best = [k for k in ["XGB h10 默认(v4)", "XGB h10 调参(v4)", "NN_B h10(v4)",
                        "融合 调参XGB+NN h10(v4)", "跨周期 融合+融合(v4)"]
            if k in preds]
    for k in best:
        df = preds[k]
        df = df[pd.DatetimeIndex(df["date"]) >= "2026-01-01"].copy()
        df["month"] = pd.DatetimeIndex(df["date"]).strftime("%Y-%m")
        proba = "proba" if "proba" in df.columns else "proba_down"
        pred = "pred"
        parts = []
        for mo, g in df.groupby("month"):
            y = g["y_true"]
            parts.append(f"{mo[5:]}:{accuracy_score(y, g[pred]):.2f}")
        P(f"  {k:<28} " + " ".join(parts))

    (OUT / "43v4增强评估.txt").write_text("\n".join(L), encoding="utf-8")
    P(f"\n输出: {OUT / '43v4增强评估.txt'}")


if __name__ == "__main__":
    main()
