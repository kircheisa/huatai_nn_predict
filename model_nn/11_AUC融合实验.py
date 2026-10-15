# -*- coding: utf-8 -*-
"""
11_AUC融合实验.py
======================================================================
P0：零训练融合。把 OPT9 已有 15 组 OOS 预测（含逐 seed 概率）作为模型池，
在开发期(<=2024)选择融合方案，2025 作验证查看。

融合方式：
  mean      : 等权概率平均
  rank      : 秩平均（对概率标度不敏感）
  devw      : 按 dev AUC 加权（softmax 温度可调）
  greedy    : 开发期 AUC 贪心前向选择
评估口径：
  native   : y_true 为各 run 自身标签（down_ex2 主线）
  direction: 原始方向 Δ<0（从利率序列重建，与标签无关）
选择协议：所有"选择"只用开发期；2025列仅作报告。
运行: & "D:\\soft\\Anaconda3\\python.exe" -u -B "11_AUC融合实验.py"
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
CUTOFF = pd.Timestamp("2024-12-31")
HORIZON = 10

RUNS = [
    "OPT9_base_D", "OPT9_relabel_D", "OPT9_twostage_D",
    "OPT9_features_E", "OPT9_featrelab_E",
    "OPT9_featrelab_E_W1260", "OPT9_featrelab_E_W756", "OPT9_featrelab_E_W504",
    "OPT9_featrelab_E_R252", "OPT9_featrelab_E_S20",
    "OPT9_featF_relab", "OPT9_featF_down",
    "OPT9_event_objective_E", "OPT9_event_direction_E",
    "RAW_OPT_FINAL_S5_E40",
]
# down_ex2 主线（融合 native AUC 的默认池）
EX2_RUNS = [r for r in RUNS if r not in
            ("OPT9_base_D", "OPT9_features_E", "OPT9_featF_down", "OPT9_twostage_D")]


def load_rate():
    rate = pd.read_csv(ROOT / "data_ori" / "2026_origin_rate.csv").iloc[:, :2]
    rate.columns = ["date", "rate"]
    rate["date"] = pd.to_datetime(rate["date"])
    rate["rate"] = pd.to_numeric(rate["rate"], errors="coerce")
    rate = rate.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
    rate["delta"] = rate["rate"].shift(-HORIZON) - rate["rate"]
    rate["maturity"] = rate["date"].shift(-HORIZON)
    return rate.set_index("date")


def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def seg(y, p):
    pred = (np.asarray(p) >= 0.5).astype(int)
    maj = max(np.mean(y), 1 - np.mean(y))
    return {"n": len(y), "acc": float(accuracy_score(y, pred)), "majority": float(maj),
            "vs_majority": float(accuracy_score(y, pred) - maj), "auc": auc(y, p)}


def main():
    rate = load_rate()
    base = pd.read_csv(OUT / f"{RUNS[0]}_predictions.csv", parse_dates=["date"])
    key = base[["date", "fold"]].copy()
    delta = key["date"].map(rate["delta"]).to_numpy()
    maturity = key["date"].map(rate["maturity"])
    assert np.isfinite(delta).all() and maturity.notna().all()
    y_dir = (delta < 0).astype(int)
    y_ex2 = (delta < -0.02).astype(int)
    dev = ((key["date"] <= CUTOFF) & (maturity <= CUTOFF)).to_numpy()
    m25 = (key["date"].dt.year == 2025).to_numpy()

    # ---- 模型池：每个 run 的 mean 概率 + 每个 seed 单模型 ----
    pool, meta = {}, {}
    for run in RUNS:
        pr = pd.read_csv(OUT / f"{run}_predictions.csv", parse_dates=["date"])
        assert (pr["date"].to_numpy() == key["date"].to_numpy()).all(), run
        pool[f"{run}:mean"] = pr["proba_down"].to_numpy()
        meta[f"{run}:mean"] = {"run": run, "kind": "mean"}
        for c in [c for c in pr.columns if c.startswith("seed")]:
            pool[f"{run}:{c}"] = pr[c].to_numpy()
            meta[f"{run}:{c}"] = {"run": run, "kind": "seed"}
    # 同周期 XGB（因果特征 E/F）
    for run in sorted(OUT.glob("OPTX_xgb*_predictions.csv")):
        run = run.stem.replace("_predictions", "")
        path = OUT / f"{run}_predictions.csv"
        pr = pd.read_csv(path, parse_dates=["date"])
        if len(pr) != len(key) or not (pr["date"].to_numpy() == key["date"].to_numpy()).all():
            print(f"跳过 {run}: 日期不对齐")
            continue
        pool[f"{run}:mean"] = pr["proba_down"].to_numpy()
        meta[f"{run}:mean"] = {"run": run, "kind": "mean"}
    # 第四轮新增 NN 任务（featG / S6 / PW）自动发现
    new_runs = []
    for run in sorted(OUT.glob("OPT9_feat*_predictions.csv")):
        run = run.stem.replace("_predictions", "")
        if run in [m["run"] for m in meta.values()] or run in RUNS:
            continue
        pr = pd.read_csv(OUT / f"{run}_predictions.csv", parse_dates=["date"])
        if len(pr) != len(key) or not (pr["date"].to_numpy() == key["date"].to_numpy()).all():
            print(f"跳过 {run}: 日期不对齐")
            continue
        new_runs.append(run)
        pool[f"{run}:mean"] = pr["proba_down"].to_numpy()
        meta[f"{run}:mean"] = {"run": run, "kind": "mean"}
        for c in [c for c in pr.columns if c.startswith("seed")]:
            pool[f"{run}:{c}"] = pr[c].to_numpy()
            meta[f"{run}:{c}"] = {"run": run, "kind": "seed"}
    if new_runs:
        print("新增NN任务:", ", ".join(new_runs))
    # 缓存特征 XGB H10（29折旧管线；2025覆盖完整；注意特征可得性待审计）
    x10 = pd.read_csv(ROOT / "model_new" / "data" / "xgb_test_predictions_10d.csv",
                      parse_dates=["date"])
    x10 = x10.rename(columns={"prob_1_下行": "proba"})[["date", "proba"]]
    merged = key.merge(x10, on="date", how="left")
    cover = merged["proba"].notna()
    print(f"XGB10D 日期覆盖: {int(cover.sum())}/{len(key)} (2025内"
          f"{int(cover[m25].sum())}/{int(m25.sum())})")
    pool["XGB10D:mean"] = merged["proba"].fillna(0.5).to_numpy()
    meta["XGB10D:mean"] = {"run": "XGB10D", "kind": "mean"}
    mask_x10 = merged["proba"].notna().to_numpy()
    names = list(pool)
    P = np.column_stack([pool[n] for n in names])
    rank = pd.DataFrame(P).rank(pct=True).to_numpy()

    def report(tag, p, note="", keep=None):
        r = {}
        for label_name, y in (("native", y_ex2), ("direction", y_dir)):
            for seg_name, m in (("dev", dev), ("2025", m25), ("full", np.ones(len(y), bool))):
                seg_keep = m if keep is None else (m & keep)
                r[f"{label_name}_{seg_name}"] = seg(y[seg_keep], p[seg_keep])
        rows.append({"方案": tag, "说明": note,
                     "dev_auc_native": r["native_dev"]["auc"],
                     "dev_auc_dir": r["direction_dev"]["auc"],
                     "2025_auc_native": r["native_2025"]["auc"],
                     "2025_auc_dir": r["direction_2025"]["auc"],
                     "2025_acc_native": r["native_2025"]["acc"],
                     "2025_vs_maj_native": r["native_2025"]["vs_majority"],
                     "2025_acc_dir": r["direction_2025"]["acc"],
                     "2025_vs_maj_dir": r["direction_2025"]["vs_majority"],
                     "dev_acc_native": r["native_dev"]["acc"],
                     "dev_vs_maj_native": r["native_dev"]["vs_majority"]})
        return r

    rows = []
    print(f"模型池: {len(names)} 个 (15 run, 含逐seed)", flush=True)
    print(f"样本: dev={dev.sum()}  2025={m25.sum()}", flush=True)

    # ---- 单模型基线 ----
    singles = []
    for i, n in enumerate(names):
        singles.append((auc(y_ex2[dev], P[dev, i]),
                        auc(y_ex2[m25], P[m25, i]), n))
    singles.sort(reverse=True)
    print("\n[单模型 dev AUC(native) Top10]")
    for d, t, n in singles[:10]:
        print(f"  {n:42s} dev={d:.4f}  2025={t:.4f}")
    best_single = singles[0][2]
    report("single_best_dev", P[:, names.index(best_single)],
           f"单模型 {best_single}")
    report("single_XGB10D", P[:, names.index("XGB10D:mean")],
           "缓存特征XGB H10（29折旧管线）", keep=mask_x10)

    # ---- 融合方案 ----
    ex2_idx = [i for i, n in enumerate(names) if meta[n]["run"] in EX2_RUNS]
    all_idx = list(range(len(names)))

    report("mean_ex2_runs", P[:, ex2_idx].mean(1), "7个down_ex2 run等权")
    report("mean_all_runs_mean", P[:, [i for i, n in enumerate(names) if meta[n]['kind'] == 'mean']].mean(1), "15个run的mean概率等权")
    report("mean_all_pool", P[:, all_idx].mean(1), "全部seed+mean等权")
    report("rank_all_pool", rank[:, all_idx].mean(1), "全部秩平均")

    # dev AUC 加权（每个模型先在 dev 上算 AUC，softmax 权重）
    dev_aucs = np.array([auc(y_ex2[dev], P[dev, i]) for i in all_idx])
    for temp in (0.01, 0.02, 0.05):
        wgt = np.exp((dev_aucs - dev_aucs.max()) / temp)
        wgt /= wgt.sum()
        report(f"devw_t{temp}", P[:, all_idx] @ wgt, "devAUC softmax加权")

    # 贪心前向（开发期 native AUC），只对 run 级 mean 概率做，控制规模
    run_names = [n for n in names if meta[n]["kind"] == "mean"]
    selected, cand = [], list(run_names)
    cur_p = None
    while cand:
        best_gain, best_n = -1, None
        for n in cand:
            p = P[:, names.index(n)] if cur_p is None else (cur_p * len(selected) + P[:, names.index(n)]) / (len(selected) + 1)
            g = auc(y_ex2[dev], p[dev])
            if g > best_gain:
                best_gain, best_n, best_p = g, n, p
        if selected and best_gain <= auc(y_ex2[dev], cur_p[dev]) - 1e-5:
            break
        selected.append(best_n)
        cand.remove(best_n)
        cur_p = best_p
        print(f"  greedy +{best_n:40s} devAUC={best_gain:.4f}", flush=True)
        if len(selected) >= 8:
            break
    report("greedy_dev", cur_p, "+".join(s.split(":")[0].replace("OPT9_", "") for s in selected))

    # 简单两两：最强单 run 与 featF 融合
    for a, b in [("OPT9_featF_relab:mean", "OPT9_featrelab_E_R252:mean"),
                 ("OPT9_featF_relab:mean", "OPT9_featrelab_E:mean"),
                 ("OPT9_featF_relab:mean", "RAW_OPT_FINAL_S5_E40:mean"),
                 ("OPT9_featF_relab:mean", "OPT9_relabel_D:mean"),
                 ("RAW_OPT_FINAL_S5_E40:mean", "OPT9_featrelab_E_R252:mean")]:
        report(f"pair:{a.split(':')[0][5:]}+{b.split(':')[0][5:] if 'RAW' not in b else 'E40'}",
               (P[:, names.index(a)] + P[:, names.index(b)]) / 2)

    # NN x XGB 加权融合：权重只在开发期网格选择
    nn_p = P[:, names.index("OPT9_featF_relab:mean")]
    xg_p = P[:, names.index("XGB10D:mean")]
    best_w, best_dev_auc = None, -1
    for w in np.arange(0.0, 1.0001, 0.05):
        p = w * nn_p + (1 - w) * xg_p
        a = auc(y_ex2[dev & mask_x10], p[dev & mask_x10])
        if a > best_dev_auc:
            best_dev_auc, best_w = a, float(w)
    report(f"fuse:featF*{best_w:.2f}+XGB10D*{1-best_w:.2f}",
           best_w * nn_p + (1 - best_w) * xg_p,
           f"开发期选权 w={best_w:.2f} (devAUC={best_dev_auc:.4f})", keep=mask_x10)
    for w in (0.5, 0.7):
        report(f"fuse_fix{w:.1f}", w * nn_p + (1 - w) * xg_p,
               f"固定 w={w:.1f}", keep=mask_x10)
    # 三方: featF + E40 + XGB10D (dev 上对 NN 间权重粗网格)
    e40_p = P[:, names.index("RAW_OPT_FINAL_S5_E40:mean")]
    best3 = None
    for w1 in np.arange(0.0, 1.0001, 0.1):
        for w2 in np.arange(0.0, 1.0001 - w1 + 1e-9, 0.1):
            p = (w1 * nn_p + w2 * e40_p + (1 - w1 - w2) * xg_p)
            a = auc(y_ex2[dev & mask_x10], p[dev & mask_x10])
            if best3 is None or a > best3[0]:
                best3 = (float(a), float(w1), float(w2))
    a3, w1, w2 = best3
    report(f"fuse3:{w1:.1f}*featF+{w2:.1f}*E40+{1-w1-w2:.1f}*XGB10D",
           w1 * nn_p + w2 * e40_p + (1 - w1 - w2) * xg_p,
           f"开发期三方选权 devAUC={a3:.4f}", keep=mask_x10)

    # ---- 权重选择准则对比（全部只用开发期，2025列仅报告） ----
    yr = key["date"].dt.year.to_numpy()
    recent = dev & np.isin(yr, [2023, 2024])
    dev_years = [y for y in np.unique(yr[dev]) ]
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    crit_rows = []
    for w in grid:
        p = w * nn_p + (1 - w) * xg_p
        row = {"w_nn": w,
               "dev_auc": auc(y_ex2[dev & mask_x10], p[dev & mask_x10]),
               "dev2324_auc": auc(y_ex2[recent & mask_x10], p[recent & mask_x10]),
               "2025_auc": auc(y_ex2[m25 & mask_x10], p[m25 & mask_x10])}
        row["dev_minyear_auc"] = min(
            auc(y_ex2[dev & mask_x10 & (yr == y)], p[dev & mask_x10 & (yr == y)])
            for y in dev_years)
        crit_rows.append(row)
    crit = pd.DataFrame(crit_rows)
    crit.to_csv(OUT / "OPTB_权重准则.csv", index=False, encoding="utf-8-sig")
    w_rec = float(crit.loc[crit["dev2324_auc"].idxmax(), "w_nn"])
    w_min = float(crit.loc[crit["dev_minyear_auc"].idxmax(), "w_nn"])
    report(f"fuse_recency:{w_rec:.2f}featF", w_rec * nn_p + (1 - w_rec) * xg_p,
           f"2023-24近期段选权 w={w_rec:.2f}", keep=mask_x10)
    report(f"fuse_minyear:{w_min:.2f}featF", w_min * nn_p + (1 - w_min) * xg_p,
           f"开发期逐年最小AUC准则 w={w_min:.2f}", keep=mask_x10)
    # logit 空间融合（同权重的另一种标度）
    def logit(x):
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))
    lg = 0.5 * logit(nn_p) + 0.5 * logit(xg_p)
    from scipy.special import expit
    report("fuse_logit50", expit(lg), "logit空间等权", keep=mask_x10)
    lg7 = 0.7 * logit(nn_p) + 0.3 * logit(xg_p)
    report("fuse_logit70", expit(lg7), "logit空间 0.7/0.3", keep=mask_x10)

    print("\n[权重准则敏感性] w=NN权重 (0=纯XGB10D, 1=纯featF)")
    print(crit.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- 第四轮：新NN组件 vs XGB10D 的开发期选权融合 ----
    for run in new_runs:
        if run.endswith("_PW") or "_W" in run:
            continue
        nn2 = P[:, names.index(f"{run}:mean")]
        bw, ba = None, -1
        for w in grid:
            p = w * nn2 + (1 - w) * xg_p
            a = auc(y_ex2[dev & mask_x10], p[dev & mask_x10])
            if a > ba:
                ba, bw = a, float(w)
        p = bw * nn2 + (1 - bw) * xg_p
        report(f"fuse4:{run.replace('OPT9_', '')}*{bw:.2f}+XGB10D", p,
               f"开发期选权 w={bw:.2f} devAUC={ba:.4f}", keep=mask_x10)
        # 该NN单独
        report(f"single4:{run.replace('OPT9_', '')}", nn2, "第四轮新NN单独")

    res = pd.DataFrame(rows).sort_values("dev_auc_native", ascending=False)
    res.to_csv(OUT / "OPTB_融合对比.csv", index=False, encoding="utf-8-sig")

    # ---- 导出关键方案逐日预测（含逐年指标摘要） ----
    wgt_t001 = np.exp((dev_aucs - dev_aucs.max()) / 0.01)
    wgt_t001 /= wgt_t001.sum()
    export = {
        "single_featF": nn_p,
        "single_XGB10D": xg_p,
        "CHAMPION_devw_t001": P[:, all_idx] @ wgt_t001,
        "fuse_w0.40": 0.40 * nn_p + 0.60 * xg_p,
        "fuseC_w0.70(2025informed)": 0.70 * nn_p + 0.30 * xg_p,
    }
    out = key.copy()
    out["y_ex2"] = y_ex2
    out["y_dir"] = y_dir
    out["dev"] = dev
    for name, p in export.items():
        out[f"p_{name}"] = np.round(p, 6)
    out.to_csv(OUT / "OPTB_final_predictions.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    for name, p in export.items():
        for seg_name, m in (("dev", dev), ("2025", m25)):
            summary_rows.append({
                "方案": name, "段": seg_name,
                "n": int(m.sum()),
                "acc_native": seg(y_ex2[m], p[m])["acc"],
                "auc_native": seg(y_ex2[m], p[m])["auc"],
                "acc_dir": seg(y_dir[m], p[m])["acc"],
                "auc_dir": seg(y_dir[m], p[m])["auc"]})
    pd.DataFrame(summary_rows).to_csv(OUT / "OPTB_final_摘要.csv",
                                      index=False, encoding="utf-8-sig")
    cols = ["方案", "dev_auc_native", "2025_auc_native", "dev_auc_dir", "2025_auc_dir",
            "2025_acc_native", "2025_acc_dir", "dev_acc_native"]
    print("\n[融合方案对比] (按 dev_auc_native 排序)")
    print(res[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n输出: {OUT / 'OPTB_融合对比.csv'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
