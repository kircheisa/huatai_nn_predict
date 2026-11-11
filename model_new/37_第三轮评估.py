# -*- coding: utf-8 -*-
"""第三轮评估：h15/h20 长周期、融合组合与诚实口径（purge）对照"""

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
                f"F1={f1_score(y, g['pred'], average='macro', zero_division=0):.4f} "
                f"基线={max(y.mean(), 1 - y.mean()):.4f}")
    P(f"{name:<30} 全局 n={len(df):<5} {seg(df)}")
    for yr in (2025, 2026):
        g = df[pd.DatetimeIndex(df["date"]).year == yr]
        if len(g):
            P(f"{'':<30} {yr}  n={len(g):<4} {seg(g)}")


def main() -> None:
    P("=" * 84)
    P("第三轮：h15/h20 长周期与诚实口径（purge）")
    P("=" * 84)

    xgb = {k: load(OUT / f"{k}_test_predictions.csv") for k in
           ["xgbv4h15", "xgbv4h20", "xgbv4h20p20", "xgbv4h20rec15",
            "xgbv4h10f200", "xgbv4h10", "xgbv4h10p10"]}
    nn20 = {t: load(NN_OUT / f"{t}_nnv2_test_predictions.csv")
            for t in ["NNV3H20B", "NNV4H20PF"]}

    P("\n[1] 长周期 XGB 与口径对照")
    for k in ["xgbv4h10", "xgbv4h15", "xgbv4h20", "xgbv4h20p20",
              "xgbv4h20rec15", "xgbv4h10p10"]:
        if xgb[k] is not None:
            show(k, xgb[k])

    P("\n[2] h20 融合")
    for name, a, b in [
        ("融合 XGBh20+NNv3h20", xgb["xgbv4h20"], nn20["NNV3H20B"]),
        ("融合 XGBh20+NNv4PFh20", xgb["xgbv4h20"], nn20["NNV4H20PF"]),
        ("融合 XGBh20rec+NNv4PFh20", xgb["xgbv4h20rec15"], nn20["NNV4H20PF"]),
        ("融合 XGBh20+XGBh10", xgb["xgbv4h20"], xgb["xgbv4h10"]),
    ]:
        if a is not None and b is not None:
            m = fuse(a, b)
            show(name, m)

    P("\n[3] 2026年逐月Acc（h20系）")
    for name, df in [("XGBh20", xgb["xgbv4h20"]),
                     ("XGBh20rec", xgb["xgbv4h20rec15"]),
                     ("XGBh20p20", xgb["xgbv4h20p20"])]:
        if df is None:
            continue
        g = df[pd.DatetimeIndex(df["date"]) >= "2026-01-01"].copy()
        parts = [f"{mo}:{accuracy_score(x['y_true'], x['pred']):.2f}"
                 for mo, x in g.groupby(pd.DatetimeIndex(g["date"]).strftime("%m"))]
        P(f"  {name:<16} " + " ".join(parts))

    (OUT / "37_第三轮评估.txt").write_text("\n".join(L), encoding="utf-8")
    P(f"\n输出: {OUT / '37_第三轮评估.txt'}")


if __name__ == "__main__":
    main()
