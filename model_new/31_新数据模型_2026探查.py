# -*- coding: utf-8 -*-
"""
新版原始数据（43v3）XGB/BiGRU 模型 2026年1-11月准确率探查

读取 21_XGB 与 17_BiGRU 在 43v3 上的样本外逐日预测：
  h5 : xgbv3              + NNV3H5B  (NN_B: hidden64/lr5e-4/s60)
  h10: xgbv3h10 / 调参xgbv3tuneh10 + NNV3H10B
输出:
  data_new/out/新数据模型_2026探查.txt   全局/分年/2026逐月/融合/方向偏差
  data_new/out/新数据模型_2026逐月.csv
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


def load(prefix_xgb: str, tag_nn: str,
         prefix_tuned: str | None = None) -> dict[str, pd.DataFrame]:
    x = pd.read_csv(OUT / f"{prefix_xgb}_test_predictions.csv")
    x["date"] = pd.to_datetime(x["date"])
    out = {"xgb": x}
    if prefix_tuned:
        t = pd.read_csv(OUT / f"{prefix_tuned}_test_predictions.csv")
        t["date"] = pd.to_datetime(t["date"])
        out["xgb_tuned"] = t
    if tag_nn:
        n = pd.read_csv(NN_OUT / f"{tag_nn}_nnv2_test_predictions.csv")
        n["date"] = pd.to_datetime(n["date"])
        out["nn"] = n
        for key, xg in [("fuse", x), ("fuse_tuned", out.get("xgb_tuned"))]:
            if xg is None:
                continue
            m = xg.merge(n, on="date", suffixes=("_x", "_n"))
            # 同一套标签文件，理论上一致；不一致以XGB侧为准
            m["y_true"] = m["y_true_x"]
            m["proba_fuse"] = (m["proba_down_x"] + m["proba_down_n"]) / 2
            m["pred_fuse"] = (m["proba_fuse"] >= 0.5).astype(int)
            out[key] = m
    return out


def metrics(g: pd.DataFrame, proba: str, pred: str) -> dict:
    y = g["y_true"]
    return {
        "n": len(g),
        "acc": accuracy_score(y, g[pred]),
        "auc": roc_auc_score(y, g[proba]) if y.nunique() > 1 else np.nan,
        "f1": f1_score(y, g[pred], average="macro", zero_division=0),
        "base": max(y.mean(), 1 - y.mean()),
        "实际下行占比": y.mean(),
        "预测下行占比": g[pred].mean(),
    }


def report(name: str, df: pd.DataFrame, proba: str, pred: str) -> None:
    g = metrics(df, proba, pred)
    P(f"{name:<28} n={g['n']:<5} Acc={g['acc']:.4f} AUC={g['auc']:.4f} "
      f"F1={g['f1']:.4f} 多数类基线={g['base']:.4f} "
      f"实际下行{g['实际下行占比']:.3f}/预测下行{g['预测下行占比']:.3f}")


def main() -> None:
    P("=" * 78)
    P("新版原始数据（43v3）模型 2026年1-11月准确率探查")
    P("XGB: 21脚本协议(窗口756/步长63/测试2018起)  NN: 17脚本NN_B(hidden64/lr5e-4/s60)")
    P("=" * 78)

    monthly_rows = []
    for hz, xgb_pref, nn_tag in [("h5", "xgbv3", "NNV3H5B"),
                                 ("h10", "xgbv3h10", "NNV3H10B")]:
        P("\n" + "#" * 78)
        P(f"### horizon={hz}")
        data = load(xgb_pref, nn_tag,
                    prefix_tuned="xgbv3tuneh10" if hz == "h10" else None)

        P("\n[全局 & 分年]")
        combos = [("xgb", "proba_down", "pred", "XGB默认"),
                  ("fuse", "proba_fuse", "pred_fuse", "XGB默认+NN_B融合"),
                  ("nn", "proba_down", "pred", "NN_B")]
        if hz == "h10":
            combos += [("xgb_tuned", "proba_down", "pred", "XGB调参(d2/lr0.03)"),
                       ("fuse_tuned", "proba_fuse", "pred_fuse",
                        "XGB调参+NN_B融合")]
        for key, proba, pred, nm in combos:
            if key not in data:
                continue
            df = data[key]
            report(f"{nm} 全局", df, proba, pred)
            yr = df.assign(year=pd.DatetimeIndex(df["date"]).year)
            for y, g in yr.groupby("year"):
                r = metrics(g, proba, pred)
                P(f"    {y}: n={r['n']:<4} Acc={r['acc']:.4f} AUC={r['auc']:.4f} "
                  f"基线={r['base']:.4f} 实际下行{r['实际下行占比']:.3f}/"
                  f"预测下行{r['预测下行占比']:.3f}")

        # 2026 逐月（以融合为主线，附XGB与NN单模型）
        P(f"\n[2026年逐月] horizon={hz}")
        for key, proba, pred, nm in [("xgb", "proba_down", "pred", "XGB"),
                                     ("nn", "proba_down", "pred", "NN_B"),
                                     ("fuse", "proba_fuse", "pred_fuse", "融合"),
                                     ("fuse_tuned", "proba_fuse", "pred_fuse", "调参融合")]:
            if key not in data:
                continue
            df = data[key].copy()
            df = df[df["date"] >= "2026-01-01"]
            if df.empty:
                continue
            df["month"] = df["date"].dt.to_period("M").astype(str)
            P(f"  -- {nm} --")
            cum_y, cum_p = [], []
            for mo, g in df.groupby("month"):
                r = metrics(g, proba, pred)
                cum_y += g["y_true"].tolist()
                cum_p += g[pred].tolist()
                r["月份"], r["模型"], r["horizon"] = mo, nm, hz
                monthly_rows.append(r)
                P(f"    {mo}: n={r['n']:<3} Acc={r['acc']:.3f} AUC={r['auc']:.3f} "
                  f"基线={r['base']:.3f} 实际下行{r['实际下行占比']:.2f}/"
                  f"预测下行{r['预测下行占比']:.2f}  "
                  f"累计Acc={accuracy_score(cum_y, cum_p):.3f}")

    md = pd.DataFrame(monthly_rows)
    md.to_csv(OUT / "新数据模型_2026逐月.csv", index=False, encoding="utf-8-sig")
    (OUT / "新数据模型_2026探查.txt").write_text("\n".join(L), encoding="utf-8")
    P(f"\n输出: {OUT / '新数据模型_2026探查.txt'}")


if __name__ == "__main__":
    main()
