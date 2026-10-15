# -*- coding: utf-8 -*-
"""
08_预测诊断分析.py
============================================================
对最终模型(RAW_OPT_FINAL_S5_E40)的逐日预测做结构化诊断，
目标是回答：还有没有更好的方法提高准确率，尤其是2025年。

诊断维度：
  A 分年 / 2025分月  指标，定位误差集中的时间段
  B 按真实涨跌幅分层 |Δbp|，检验模型是否只在"大变动"时才有判别力
  C 固定阈值扫描（仅在开发期≤2024选择），再套用到2025，检验阈值是否被浪费
  D 概率分布与方向偏置（是否结构性偏向某一类）
  E proba_down 与实际涨跌幅的秩相关（模型到底有没有排序能力）
所有"选择"只在开发期完成，2025仅作结果查看。

运行: & "D:\\soft\\Anaconda3\\python.exe" -u -B "08_预测诊断分析.py"
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ROOT = HERE.parent

TAG = "RAW_OPT_FINAL_S5_E40"
CUTOFF = pd.Timestamp("2024-12-31")
TH_GRID = np.round(np.arange(0.30, 0.705, 0.025), 3)


def load():
    pred = pd.read_csv(OUT / f"{TAG}_predictions.csv", parse_dates=["date"]).sort_values("date")
    rate = pd.read_csv(ROOT / "data_ori" / "2026_origin_rate.csv").iloc[:, :2]
    rate.columns = ["date", "rate"]
    rate["date"] = pd.to_datetime(rate["date"], errors="coerce")
    rate = rate.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    rate["maturity"] = rate["date"].shift(-10)
    rate["future_rate"] = rate["rate"].shift(-10)
    pred = pred.merge(rate[["date", "rate", "maturity", "future_rate"]], on="date", how="left")
    pred = pred.dropna(subset=["maturity"]).reset_index(drop=True)
    pred["change_bp"] = (pred["future_rate"] - pred["rate"]) * 100
    pred["abs_bp"] = pred["change_bp"].abs()
    pred["year"] = pred["date"].dt.year
    pred["dev"] = (pred["date"] <= CUTOFF) & (pred["maturity"] <= CUTOFF)
    return pred


def metrics(y, pred_label, proba=None):
    y = np.asarray(y); pred_label = np.asarray(pred_label)
    m = {
        "n": len(y),
        "acc": float(accuracy_score(y, pred_label)),
        "macro_f1": float(f1_score(y, pred_label, average="macro", zero_division=0)),
        "pred_down": float(pred_label.mean()),
        "true_down": float(y.mean()),
        "majority": float(max(y.mean(), 1 - y.mean())),
    }
    if proba is not None and len(np.unique(y)) > 1:
        m["auc"] = float(roc_auc_score(y, proba))
    else:
        m["auc"] = float("nan")
    return m


def main():
    p = load()
    y = p["y_true"].to_numpy()
    proba = p["proba_down"].to_numpy()
    pred = p["pred"].to_numpy()

    print(f"[数据] {len(p)}条  开发期{int(p['dev'].sum())}条  "
          f"预测日 {p['date'].min():%Y-%m-%d}~{p['date'].max():%Y-%m-%d}")

    # ---------- A 分年 ----------
    print("\n[A] 分年指标（阈值0.5）")
    for yr, g in p.groupby("year"):
        m = metrics(g["y_true"], g["pred"], g["proba_down"])
        gap = m["acc"] - m["majority"]
        print(f"  {yr}: n={m['n']:4d} Acc={m['acc']:.4f} AUC={m['auc']:.4f} "
              f"MacroF1={m['macro_f1']:.4f} 预测下行={m['pred_down']:.3f} 真实下行={m['true_down']:.3f} "
              f"多数类={m['majority']:.4f} 相对多数类={gap:+.4f}")

    # ---------- A2 2025分月 ----------
    print("\n[A2] 2025年分月")
    g25 = p[p["year"] == 2025].copy()
    for mo, g in g25.groupby(g25["date"].dt.to_period("M")):
        m = metrics(g["y_true"], g["pred"], g["proba_down"])
        print(f"  {mo}: n={m['n']:3d} Acc={m['acc']:.4f} "
              f"预测下行={m['pred_down']:.3f} 真实下行={m['true_down']:.3f} 多数类={m['majority']:.4f}")

    # ---------- B 按涨跌幅分层 ----------
    print("\n[B] 按|真实涨跌|分层（全期 / 2025）")
    edges = [0, 2, 5, 10, 20, 1e9]
    labels = ["0-2bp", "2-5bp", "5-10bp", "10-20bp", ">20bp"]
    p["bp_bin"] = pd.cut(p["abs_bp"], bins=edges, labels=labels)
    rows_b = []
    g25b = p[p["year"] == 2025]
    for seg_name, seg in [("全期", p), ("2025", g25b)]:
        for lb in labels:
            g = seg[seg["bp_bin"] == lb]
            if len(g) == 0 or len(np.unique(g["y_true"])) < 2:
                continue
            m = metrics(g["y_true"], g["pred"], g["proba_down"])
            rows_b.append({"区间": seg_name, "幅度": lb, **m})
            print(f"  {seg_name} {lb:>8s}: n={m['n']:4d} Acc={m['acc']:.4f} AUC={m['auc']:.4f} "
                  f"真实下行={m['true_down']:.3f} 多数类={m['majority']:.4f} 相对={m['acc']-m['majority']:+.4f}")
    pd.DataFrame(rows_b).to_csv(OUT / "DIAG_幅度分层.csv", index=False, encoding="utf-8-sig")

    # ---------- C 固定阈值扫描（开发期选，2025看） ----------
    print("\n[C] 固定阈值扫描（开发期≤2024选最优，再套用）")
    dev = p[p["dev"]]
    scan = []
    for th in TH_GRID:
        pl = (dev["proba_down"] >= th).astype(int)
        scan.append({"阈值": th, "开发期Acc": accuracy_score(dev["y_true"], pl)})
    scan = pd.DataFrame(scan)
    best_th = float(scan.loc[scan["开发期Acc"].idxmax(), "阈值"])
    print(f"  开发期最优固定阈值 = {best_th}")
    for th in [0.5, best_th]:
        for seg_name, seg in [("开发期", dev), ("全期", p), ("2025", g25)]:
            pl = (seg["proba_down"] >= th).astype(int)
            m = metrics(seg["y_true"], pl, seg["proba_down"])
            print(f"  th={th:.3f} {seg_name:>4s}: Acc={m['acc']:.4f} 预测下行={m['pred_down']:.3f} "
                  f"多数类={m['majority']:.4f} 相对={m['acc']-m['majority']:+.4f}")
    scan.to_csv(OUT / "DIAG_阈值扫描.csv", index=False, encoding="utf-8-sig")

    # ---------- D 概率分布 / 方向偏置 ----------
    print("\n[D] 概率分布（下行概率分位）")
    for seg_name, seg in [("全期", p), ("2025", g25)]:
        q = seg["proba_down"].quantile([.05, .25, .5, .75, .95]).round(3).to_dict()
        print(f"  {seg_name}: {q}  真实下行占比={seg['y_true'].mean():.3f}")

    # ---------- E 秩相关 ----------
    print("\n[E] proba_down 与真实涨跌幅(Δbp)的秩相关（应为负）")
    for seg_name, seg in [("全期", p), ("2025", g25), ("开发期", dev)]:
        rho, pv = spearmanr(seg["proba_down"], seg["change_bp"])
        print(f"  {seg_name}: Spearman={rho:.4f} (p={pv:.3g}, n={len(seg)})")

    # ---------- F 事前波动率能否识别"死区" ----------
    # 用预测日之前20个交易日的日度收益率变动标准差（bp）作为事前波动，绝不含未来信息
    rs = p[["date", "rate"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    rs["dchg_bp"] = rs["rate"].diff() * 100
    rs["vol20"] = rs["dchg_bp"].rolling(20, min_periods=10).std()
    vol_map = rs.set_index("date")["vol20"]
    p["vol20"] = p["date"].map(vol_map)
    dev = p[p["dev"]]        # 重新派生，确保带上vol20列
    print("\n[F] 事前波动率 vol20 与未来|Δ|的关系，及分档准确率")
    for seg_name, seg in [("全期", p), ("2025", p[p["year"] == 2025])]:
        s = seg.dropna(subset=["vol20"])
        rho, pv = spearmanr(s["vol20"], s["abs_bp"])
        print(f"  {seg_name}: corr(vol20, |Δ|) Spearman={rho:.3f} (p={pv:.3g})")
    # 用开发期的vol20三分位作为分档边界（不调参，纯诊断）
    q = dev["vol20"].dropna().quantile([1 / 3, 2 / 3]).to_numpy()
    print(f"  开发期vol20三分位边界(bp): {np.round(q,3)}")
    rows_f = []
    for seg_name, seg in [("开发期", dev), ("2025", p[p["year"] == 2025])]:
        s = seg.dropna(subset=["vol20"])
        for lb, sub in [("低波动", s[s["vol20"] <= q[0]]),
                        ("中波动", s[(s["vol20"] > q[0]) & (s["vol20"] <= q[1])]),
                        ("高波动", s[s["vol20"] > q[1]])]:
            if len(sub) < 5 or len(np.unique(sub["y_true"])) < 2:
                continue
            m = metrics(sub["y_true"], sub["pred"], sub["proba_down"])
            rows_f.append({"区间": seg_name, "波动档": lb, "平均绝对涨跌bp": sub["abs_bp"].mean(), **m})
            print(f"  {seg_name} {lb}: n={m['n']:3d} Acc={m['acc']:.4f} AUC={m['auc']:.4f} "
                  f"平均|Δ|={sub['abs_bp'].mean():.1f}bp 多数类={m['majority']:.4f} 相对={m['acc']-m['majority']:+.4f}")
    pd.DataFrame(rows_f).to_csv(OUT / "DIAG_波动分层.csv", index=False, encoding="utf-8-sig")

    # ---------- G 剔除"死区"后的准确率上限 ----------
    print("\n[G] 剔除|Δ|<2bp（死区）后的准确率（事后口径，代表可实现的收益上限）")
    for seg_name, seg in [("开发期", dev), ("全期", p), ("2025", p[p["year"] == 2025])]:
        nd = seg[seg["abs_bp"] < 2]
        nz = seg[seg["abs_bp"] >= 2]
        md = metrics(nd["y_true"], nd["pred"], nd["proba_down"])
        mz = metrics(nz["y_true"], nz["pred"], nz["proba_down"])
        print(f"  {seg_name}: 死区n={md['n']:3d} Acc={md['acc']:.4f}(真实下行{md['true_down']:.2f}) | "
              f"非死区n={mz['n']:3d} Acc={mz['acc']:.4f} AUC={mz['auc']:.4f} 多数类={mz['majority']:.4f} 相对={mz['acc']-mz['majority']:+.4f}")

    # ---------- 汇总导出 ----------
    yr_rows = []
    for yr, g in p.groupby("year"):
        yr_rows.append({"年份": int(yr), **metrics(g["y_true"], g["pred"], g["proba_down"])})
    pd.DataFrame(yr_rows).to_csv(OUT / "DIAG_分年指标.csv", index=False, encoding="utf-8-sig")
    print(f"\n[导出] DIAG_幅度分层.csv / DIAG_阈值扫描.csv / DIAG_分年指标.csv")


if __name__ == "__main__":
    main()