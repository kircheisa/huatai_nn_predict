# -*- coding: utf-8 -*-
"""
07_规则过滤器回测.py
============================================================
在最终基础特征模型（RAW_OPT_FINAL_S5_E40）的逐日预测之上，
评估一层"决策规则"，用于回答：多天预测失败后能否用规则提高准确率。

四类规则（全部严格防前视）：
  R1 置信度过滤     —— 概率贴近0.5的不出手
  R2 seed一致性过滤 —— 5个seed方向分歧大的不出手
  R3 滚动表现保护   —— 只用"标签已成熟"的预测计算近期准确率，低于阈值则停手
  R4 滚动阈值调整   —— 用"标签已成熟"的预测动态选阈值再判方向

关键时间约束：
  预测日期 t 的标签要到 t 之后第10个交易日才成熟，
  因此 t 时刻只能使用 maturity <= t 的预测结果，绝不能使用尚未成熟的标签。

选参口径：
  仅在开发期（预测日期与标签成熟日期都 <= 2024-12-31）选择规则参数，
  再原样套用到全期与2025年；2025年此前已被历史迭代观察过，本轮不作为选型依据。

运行：
  & "D:\\soft\\Anaconda3\\python.exe" -u -B "07_规则过滤器回测.py"
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ROOT = HERE.parent

TAG = "RAW_OPT_FINAL_S5_E40"
SEED_COLS = ["seed0", "seed1", "seed2", "seed3", "seed4"]
CUTOFF = pd.Timestamp("2024-12-31")     # 开发期截止（同时限制预测日与成熟日）
TH_GRID = np.round(np.arange(0.30, 0.705, 0.05), 2)

# 选参约束：开发期至少保留这么多信号、覆盖率不低于此值，避免退化成"只留极少数样本"
MIN_DEV_KEPT = 150
MIN_DEV_COVERAGE = 0.40


# ============================================================
# 数据加载：预测 + 标签成熟日期
# ============================================================
def load():
    pred = pd.read_csv(OUT / f"{TAG}_predictions.csv", parse_dates=["date"])
    pred = pred.sort_values("date").reset_index(drop=True)

    rate = pd.read_csv(ROOT / "data_ori" / "2026_origin_rate.csv").iloc[:, :2]
    rate.columns = ["date", "rate"]
    rate["date"] = pd.to_datetime(rate["date"], errors="coerce")
    rate = rate.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    rate["maturity"] = rate["date"].shift(-10)
    mat_map = rate.set_index("date")["maturity"]

    pred["maturity"] = pred["date"].map(mat_map)
    missing = pred["maturity"].isna()
    if missing.any():
        # 末端不足10个交易日的样本标签未成熟，直接剔除（不参与任何评估）
        pred = pred.loc[~missing].reset_index(drop=True)

    assert (pred["maturity"] > pred["date"]).all()
    # 校验标签与10交易日后收益方向一致（与导出脚本同一口径）
    rate2 = rate.set_index("date")
    cur = pred["date"].map(rate2["rate"])
    fut = pred["maturity"].map(rate2["rate"])
    assert np.array_equal((fut < cur).astype(int).to_numpy(), pred["y_true"].to_numpy())
    return pred


# ============================================================
# 规则生成：返回 keep(bool数组) 或 新预测标签(int数组)
# ============================================================
def keep_margin(pred, m):
    """R1 置信度过滤：概率与0.5的距离不足 m 则不出手"""
    return (np.abs(pred["proba_down"].to_numpy() - 0.5) >= m)


def seed_agreement(pred):
    """每个seed自身方向与集成方向一致的占比"""
    seed_pred = (pred[SEED_COLS].to_numpy() >= 0.5).astype(int)
    ens = pred["pred"].to_numpy()
    return (seed_pred == ens[:, None]).mean(axis=1)


def keep_agreement(pred, a):
    """R2 seed一致性过滤：一致占比低于 a 则不出手"""
    return seed_agreement(pred) >= a


def keep_rolling_guard(pred, W, thr, min_hist):
    """R3 滚动表现保护：用已成熟预测的近期准确率决定是否停手"""
    y = pred["y_true"].to_numpy()
    p = pred["pred"].to_numpy()
    d = pred["date"].to_numpy()
    mat = pred["maturity"].to_numpy()
    corr = (p == y).astype(float)
    n = len(pred)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        avail = np.where(mat <= d[i])[0]      # 只用到i为止已成熟的预测
        if len(avail) < min_hist:
            keep[i] = True                    # 历史不足，默认不干预
            continue
        recent = avail[-W:]
        keep[i] = corr[recent].mean() >= thr
    return keep


def rolling_threshold_pred(pred, W, min_hist):
    """R4 滚动阈值：用已成熟预测在候选阈值上选近期最优阈值，再判当前方向"""
    y = pred["y_true"].to_numpy()
    proba = pred["proba_down"].to_numpy()
    d = pred["date"].to_numpy()
    mat = pred["maturity"].to_numpy()
    n = len(pred)
    out = np.zeros(n, dtype=int)
    for i in range(n):
        avail = np.where(mat <= d[i])[0]
        if len(avail) < min_hist:
            th = 0.5
        else:
            recent = avail[-W:]
            best_th, best_acc = 0.5, -1.0
            for th in TH_GRID:
                acc = ((proba[recent] >= th).astype(int) == y[recent]).mean()
                if acc > best_acc:
                    best_acc, best_th = acc, th
            th = best_th
        out[i] = int(proba[i] >= th)
    return out


def keep_consecutive_fail(pred, k, min_hist=40):
    """R5 连续失败停手：最近 k 个已成熟预测全部错误时，当前预测不出手"""
    y = pred["y_true"].to_numpy()
    p = pred["pred"].to_numpy()
    d = pred["date"].to_numpy()
    mat = pred["maturity"].to_numpy()
    corr = (p == y)
    n = len(pred)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        avail = np.where(mat <= d[i])[0]
        if len(avail) < max(min_hist, k):
            continue
        recent = avail[-k:]
        if not corr[recent].any():          # 最近k个成熟预测全错
            keep[i] = False
    return keep


def reverse_consecutive_fail(pred, k, min_hist=40):
    """R7 连续失败反向：最近 k 个已成熟预测全部错误时，把当前预测方向取反"""
    y = pred["y_true"].to_numpy()
    p = pred["pred"].to_numpy()
    d = pred["date"].to_numpy()
    mat = pred["maturity"].to_numpy()
    corr = (p == y)
    n = len(pred)
    out = p.copy()
    for i in range(n):
        avail = np.where(mat <= d[i])[0]
        if len(avail) < max(min_hist, k):
            continue
        if not corr[avail[-k:]].any():
            out[i] = 1 - p[i]
    return out


# ============================================================
# 指标
# ============================================================
def seg_metrics(y, pred_label, proba, keep=None):
    """在 keep 掩码内计算指标；proba 用于AUC（阈值无关），pred_label 用于Acc/F1"""
    y = np.asarray(y)
    pred_label = np.asarray(pred_label)
    proba = np.asarray(proba)
    if keep is None:
        keep = np.ones(len(y), dtype=bool)
    yk, pk, bk = y[keep], pred_label[keep], proba[keep]
    n = int(keep.sum())
    if n == 0:
        return {"n": 0}
    out = {
        "n": n,
        "coverage": float(n / len(y)),
        "acc": float(accuracy_score(yk, pk)),
        "macro_f1": float(f1_score(yk, pk, average="macro", zero_division=0)),
        "pred_down": float(pk.mean()),
        "true_down": float(yk.mean()),
        "majority_baseline": float(max(yk.mean(), 1 - yk.mean())),
    }
    out["auc"] = float(roc_auc_score(yk, bk)) if len(np.unique(yk)) > 1 else None
    return out


def evaluate(rule_pred, keep, pred, dev_mask, y, proba):
    """返回开发期/全期/2025三个口径的指标"""
    res = {}
    res["dev"] = seg_metrics(y[dev_mask], rule_pred[dev_mask], proba[dev_mask], keep[dev_mask])
    res["full"] = seg_metrics(y, rule_pred, proba, keep)
    y2025 = (pred["date"].dt.year == 2025).to_numpy()
    res["2025"] = seg_metrics(y[y2025], rule_pred[y2025], proba[y2025], keep[y2025])
    return res


# ============================================================
# 主流程
# ============================================================
def main():
    pred = load()
    y = pred["y_true"].to_numpy()
    proba = pred["proba_down"].to_numpy()
    base_pred = pred["pred"].to_numpy()
    dev_mask = ((pred["date"] <= CUTOFF) & (pred["maturity"] <= CUTOFF)).to_numpy()

    print(f"[数据] 预测 {len(pred)} 条，{pred['date'].min():%Y-%m-%d} ~ {pred['date'].max():%Y-%m-%d}")
    print(f"[数据] 开发期样本 {int(dev_mask.sum())} 条（预测日与成熟日均<=2024-12-31）")

    base = evaluate(base_pred, np.ones(len(pred), bool), pred, dev_mask, y, proba)
    print(f"[基准] 全期 Acc={base['full']['acc']:.4f} AUC={base['full']['auc']:.4f} 多数类={base['full']['majority_baseline']:.4f} | "
          f"2025 Acc={base['2025']['acc']:.4f} AUC={base['2025']['auc']:.4f} 多数类={base['2025']['majority_baseline']:.4f} | "
          f"开发期 Acc={base['dev']['acc']:.4f} 覆盖={base['dev']['coverage']:.2f}")

    rows = []
    cache = {}      # (家族, 规则) -> (keep, rule_pred)，供导出明细复用

    def add(family, name, keep, rule_pred):
        r = evaluate(rule_pred, keep, pred, dev_mask, y, proba)
        cache[(family, name)] = (keep, rule_pred)
        rows.append({
            "家族": family, "规则": name,
            "开发期_覆盖率": r["dev"]["coverage"], "开发期_保留准确率": r["dev"]["acc"],
            "开发期_保留MacroF1": r["dev"]["macro_f1"], "开发期_保留AUC": r["dev"]["auc"],
            "开发期_保留n": r["dev"]["n"], "开发期_多数类": r["dev"]["majority_baseline"],
            "全期_覆盖率": r["full"]["coverage"], "全期_保留准确率": r["full"]["acc"],
            "全期_保留MacroF1": r["full"]["macro_f1"], "全期_保留AUC": r["full"]["auc"],
            "全期_保留n": r["full"]["n"], "全期_多数类": r["full"]["majority_baseline"],
            "2025_覆盖率": r["2025"]["coverage"], "2025_保留准确率": r["2025"]["acc"],
            "2025_保留MacroF1": r["2025"]["macro_f1"], "2025_保留AUC": r["2025"]["auc"],
            "2025_保留n": r["2025"]["n"], "2025_多数类": r["2025"]["majority_baseline"],
        })

    # ---- R1 置信度过滤 ----
    for m in np.round(np.arange(0.0, 0.325, 0.025), 3):
        add("R1置信度", f"margin>={m:.3f}", keep_margin(pred, m), base_pred)

    # ---- R2 seed一致性 ----
    for a in [0.6, 0.8, 1.0]:
        add("R2种子一致性", f"agree>={a:.1f}", keep_agreement(pred, a), base_pred)

    # ---- R3 滚动表现保护 ----
    for W in [60, 120, 250]:
        for thr in [0.45, 0.50]:
            add("R3滚动保护", f"W={W},acc>={thr:.2f}",
                keep_rolling_guard(pred, W, thr, min_hist=40), base_pred)

    # ---- R4 滚动阈值 ----
    for W in [120, 250]:
        rp = rolling_threshold_pred(pred, W, min_hist=60)
        add("R4滚动阈值", f"W={W}", np.ones(len(pred), bool), rp)

    # ---- R5 连续失败停手 ----
    for k in [3, 5, 8]:
        add("R5连续失败停手", f"连续{k}次全错", keep_consecutive_fail(pred, k), base_pred)

    # ---- R7 连续失败反向 ----
    for k in [3, 5]:
        add("R7连续失败反向", f"连续{k}次全错→反向",
            np.ones(len(pred), bool), reverse_consecutive_fail(pred, k))

    # ---- R6 组合：置信度 + 种子一致性 ----
    for m in [0.10, 0.125, 0.15]:
        for a in [0.6, 0.8]:
            add("R6组合", f"margin>={m:.3f}&agree>={a:.1f}",
                keep_margin(pred, m) & keep_agreement(pred, a), base_pred)

    res = pd.DataFrame(rows)
    # 保留子集的准确率相对"该子集多数类比例"的增益：<=0 说明过滤没有带来真实方向判别力
    for seg in ["开发期", "全期", "2025"]:
        res[f"{seg}_相对多数类"] = res[f"{seg}_保留准确率"] - res[f"{seg}_多数类"]

    # ---- 用开发期选参 ----
    eligible = res[(res["开发期_保留n"] >= MIN_DEV_KEPT) &
                   (res["开发期_覆盖率"] >= MIN_DEV_COVERAGE)].copy()
    if eligible.empty:
        eligible = res.copy()
    eligible = eligible.sort_values(
        ["开发期_保留准确率", "开发期_覆盖率"], ascending=[False, False]
    )
    best_by_family = (eligible.sort_values("开发期_保留准确率", ascending=False)
                      .groupby("家族", as_index=False).first())

    print("\n[开发期选参] 各家族最优（按开发期保留准确率）：")
    for _, r in best_by_family.iterrows():
        print(f"  {r['家族']:<12s} {r['规则']:<22s} "
              f"开发期Acc={r['开发期_保留准确率']:.4f}(相对多数类{r['开发期_相对多数类']:+.4f}) 覆盖={r['开发期_覆盖率']:.2f} | "
              f"全期Acc={r['全期_保留准确率']:.4f}({r['全期_相对多数类']:+.4f})/{r['全期_覆盖率']:.2f} | "
              f"2025Acc={r['2025_保留准确率']:.4f}({r['2025_相对多数类']:+.4f})/{r['2025_覆盖率']:.2f}")

    res.to_csv(OUT / "RAW_RULE_筛选结果.csv", index=False, encoding="utf-8-sig")
    best_by_family.to_csv(OUT / "RAW_RULE_各家族最优.csv", index=False, encoding="utf-8-sig")

    # ---- 导出"各家族最优"规则在每一天的决策明细（直接从cache取，避免重复解析规则串）----
    detail_frames = []
    agree_all = seed_agreement(pred)
    for _, br in best_by_family.iterrows():
        fam, name = br["家族"], br["规则"]
        kp, rp = cache[(fam, name)]
        df = pd.DataFrame({
            "规则家族": fam,
            "规则": name,
            "日期": pred["date"].dt.strftime("%Y-%m-%d"),
            "折号": pred["fold"],
            "下行概率": pred["proba_down"],
            "原始预测方向": pd.Series(base_pred).map({0: "上行或不变", 1: "下行"}),
            "规则后预测方向": pd.Series(rp).map({0: "上行或不变", 1: "下行"}),
            "是否出手": kp,
            "真实方向": pred["y_true"].map({0: "上行或不变", 1: "下行"}),
            "原始是否正确": base_pred == y,
            "规则后是否正确": rp == y,
            "seed一致占比": agree_all,
            "标签成熟日期": pred["maturity"].dt.strftime("%Y-%m-%d"),
            "开发期内": dev_mask,
        })
        detail_frames.append(df)
    detail = pd.concat(detail_frames, ignore_index=True)
    detail.to_csv(OUT / "RAW_RULE_决策明细.csv", index=False, encoding="utf-8-sig")

    # ---- 规则后仍错误的日期（按"是否出手"区分：出手且错=决策错误；不出手=弃权）----
    err = detail[(detail["是否出手"]) & (~detail["规则后是否正确"])].copy()
    err.to_csv(OUT / "RAW_RULE_规则后错误日期.csv", index=False, encoding="utf-8-sig")

    print(f"\n[导出] 筛选结果 RAW_RULE_筛选结果.csv（{len(res)}条配置）")
    print(f"[导出] 各家族最优 RAW_RULE_各家族最优.csv")
    print(f"[导出] 决策明细 RAW_RULE_决策明细.csv（{len(detail)}行）")
    print(f"[导出] 规则后错误日期 RAW_RULE_规则后错误日期.csv（{len(err)}行）")


if __name__ == "__main__":
    main()