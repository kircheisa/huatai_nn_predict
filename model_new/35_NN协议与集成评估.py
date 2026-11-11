# -*- coding: utf-8 -*-
"""
NN 特征选择协议改进与多子集集成评估（第二轮优化）

对比对象（全部 34 folds，2018-01~2026-10，h10）：
  NN特征协议: 全局Top30(NNV4H10B) vs 逐fold训练窗Top30(NNV4H10PF)
  NN多子集:   Top20/Top40 与 K20+K30+K40 概率平均集成
  跨版本融合: XGB(v4特征) + NN(v3特征NNV3H10B——最强NN)
  XGB变体:   5-seed bagging vs 单seed；斜坡权重(证伪记录)
输出: data_new/out/35_NN协议与集成评估.txt
"""

from __future__ import annotations

from pathlib import Path

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


def load(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["date"] = pd.to_datetime(d["date"])
    return d


def norm(d: pd.DataFrame) -> pd.DataFrame:
    return d[["date", "y_true"]].assign(
        p=d["proba"] if "proba" in d.columns else d["proba_down"])


def fuse(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    m = norm(a).merge(norm(b), on="date", suffixes=("_a", "_b"))
    m["y_true"] = m["y_true_a"]
    m["proba"] = (m["p_a"] + m["p_b"]) / 2
    m["pred"] = (m["proba"] >= 0.5).astype(int)
    return m


def show(name: str, df: pd.DataFrame) -> None:
    if "proba" not in df.columns:
        df = df.copy()
        df["proba"] = df["proba_down"]

    def seg(g):
        y = g["y_true"]
        auc = roc_auc_score(y, g["proba"]) if y.nunique() > 1 else float("nan")
        return (f"Acc={accuracy_score(y, g['pred']):.4f} AUC={auc:.4f} "
                f"F1={f1_score(y, g['pred'], average='macro', zero_division=0):.4f} "
                f"基线={max(y.mean(), 1 - y.mean()):.4f}")
    P(f"{name:<30} 全局 n={len(df):<5} {seg(df)}")
    for yr in (2025, 2026):
        g = df[pd.DatetimeIndex(df["date"]).year == yr]
        if len(g):
            P(f"{'':<30} {yr}  n={len(g):<4} {seg(g)}")


def main() -> None:
    P("=" * 84)
    P("第二轮优化评估：NN特征协议 / 多子集集成 / 跨版本融合 / XGB变体（h10）")
    P("=" * 84)

    xgb_v4 = load(OUT / "xgbv4h10_test_predictions.csv")
    xgb_v4s5 = load(OUT / "xgbv4h10s5_test_predictions.csv")
    nn = {t: load(NN_OUT / f"{t}_nnv2_test_predictions.csv")
          for t in ["NNV4H10B", "NNV4H10PF", "NNV4H10K20", "NNV4H10K40",
                    "NNV3H10B"]}

    P("\n[1] NN 特征选择协议: 全局Top30 vs 逐fold训练窗Top30")
    if nn["NNV4H10B"] is not None:
        show("NN 全局Top30(v4)", nn["NNV4H10B"])
    if nn["NNV4H10PF"] is not None:
        show("NN 逐fold Top30(v4)", nn["NNV4H10PF"])
    if nn["NNV3H10B"] is not None:
        show("NN 全局Top30(v3特征)", nn["NNV3H10B"])

    P("\n[2] NN 多子集成型与集成")
    for k in ["NNV4H10K20", "NNV4H10K40"]:
        if nn[k] is not None:
            show(f"NN {k[7:]}", nn[k])
    ens = [nn[k] for k in ["NNV4H10K20", "NNV4H10B", "NNV4H10K40"]
           if nn[k] is not None]
    if len(ens) == 3:
        m = norm(ens[0]).merge(norm(ens[1]), on="date",
                               suffixes=("_1", "_2"))
        m = m.merge(norm(ens[2]).rename(columns={"p": "p_3"}), on="date")
        m["y_true"] = m["y_true_1"]
        m["proba"] = m[["p_1", "p_2", "p_3"]].mean(axis=1)
        m["pred"] = (m["proba"] >= 0.5).astype(int)
        show("NN K20+K30+K40 集成", m)

    P("\n[3] XGB 变体")
    show("XGB h10 单seed(v4)", xgb_v4)
    if xgb_v4s5 is not None:
        show("XGB h10 5-seed(v4)", xgb_v4s5)

    P("\n[4] 融合（关键组合）")
    combos = [
        ("融合 XGBv4+NN全局K30", xgb_v4, nn["NNV4H10B"]),
        ("融合 XGBv4+NN逐fold", xgb_v4, nn["NNV4H10PF"]),
        ("融合 XGBv4+NN(v3特征)", xgb_v4, nn["NNV3H10B"]),
    ]
    if xgb_v4s5 is not None and nn["NNV3H10B"] is not None:
        combos.append(("融合 XGBv4s5+NN(v3特征)", xgb_v4s5, nn["NNV3H10B"]))
    for name, a, b in combos:
        if a is not None and b is not None:
            show(name, fuse(a, b))

    P("\n[5] 2026年逐月Acc（重点配置）")
    named = {
        "XGBv4": xgb_v4, "NN逐fold": nn["NNV4H10PF"],
        "NN(v3特征)": nn["NNV3H10B"],
    }
    if len(ens) == 3:
        named["NN三K集成"] = m
    for name, a, b in combos:
        if a is not None and b is not None:
            named[name] = fuse(a, b)
    for name, df in named.items():
        if df is None:
            continue
        g = df[pd.DatetimeIndex(df["date"]) >= "2026-01-01"].copy()
        g["month"] = pd.DatetimeIndex(g["date"]).strftime("%m")
        parts = [f"{mo}:{accuracy_score(x['y_true'], x['pred']):.2f}"
                 for mo, x in g.groupby("month")]
        P(f"  {name:<28} " + " ".join(parts))

    (OUT / "35_NN协议与集成评估.txt").write_text("\n".join(L), encoding="utf-8")
    P(f"\n输出: {OUT / '35_NN协议与集成评估.txt'}")


if __name__ == "__main__":
    main()
