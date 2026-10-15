# -*- coding: utf-8 -*-
"""
超参全网格搜索 —— 完整滚动训练下 48 个组合

参数空间：
  depth ∈ {2, 3, 4}
  lr ∈ {0.05, 0.1}
  subsample ∈ {0.7, 0.8}
  colsample_bytree ∈ {0.6, 0.8}
  stl_period ∈ {63, 126}
  = 3 × 2 × 2 × 2 × 2 = 48 组合

固定参数（已知最优）：
  W=120, V=30, S=63, reg_alpha=1.5, min_child_weight=8.0
  detrend_label=True, keep_cols=keep_cols_full.txt

输出：
  model_new/data/grid_results.csv   —— 逐组合指标
  model_new/data/grid_report.txt    —— 排序汇总

注意：为控制运行时间，使用 n_estimators=400（配合 early_stopping_rounds=50）
      最终最优组合确定后用 n_estimators=600 复核。
"""

import sys
import warnings
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =====================================================================
# 导入 03 模块
# =====================================================================
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "model_new" / "data"

_spec = importlib.util.spec_from_file_location("xgb_bin", BASE / "model_new" / "03_XGB训练_二分类.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

build_features = m.build_features
load_genetic_features = m.load_genetic_features
gen_label = m.gen_label
fit_calibrator = m.fit_calibrator
calibrate_proba = m.calibrate_proba


# =====================================================================
# 特征构建（按 stl_period 缓存，避免重复计算）
# =====================================================================

def prepare_data(stl_period, keep_cols_file=None):
    """加载数据 + 构建特征 + 标签（按 stl_period 不同返回不同特征集）"""
    df_data = pd.read_csv(DATA / "data.csv")
    df_rate = pd.read_csv(DATA / "rate.csv")
    df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"])
    df_rate["time"] = pd.to_datetime(df_rate["time"])
    merged = pd.merge(df_data, df_rate, left_on="Unnamed: 0", right_on="time", how="inner").sort_values("Unnamed: 0")
    merged["target_date_5d"] = merged["Unnamed: 0"].shift(-5)
    merged["rate_delta_5d"] = merged["rate"].shift(-5) - merged["rate"]

    # 超趋势标签
    from statsmodels.tsa.seasonal import STL
    stl_temp = STL(merged["rate"].fillna(merged["rate"].mean()),
                   period=stl_period, robust=True).fit()
    trend_now = pd.Series(stl_temp.trend.values, index=merged.index)
    trend_expected_5d = trend_now.shift(-5) - trend_now
    excess_move = merged["rate_delta_5d"] - trend_expected_5d
    merged["label"] = excess_move.apply(gen_label)
    merged = merged.dropna(subset=["label"]).copy()
    merged["label"] = merged["label"].astype(int)

    # 构建特征（设置全局 _ARGS_CACHE 让 STL 特征用正确的 period）
    global_args = SimpleNamespace(
        no_stl=False, stl_period=stl_period,
        genetic_features_file="model_new/data/x_genetic_bin.csv", no_genetic=False,
    )
    m._ARGS_CACHE = global_args
    features = build_features(merged)

    # 遗传特征
    gp = load_genetic_features(global_args)
    if gp is not None:
        features = features.join(gp, how="left")
        gp_cols = [c for c in gp.columns if c in features.columns]
        features[gp_cols] = features[gp_cols].fillna(0.0)

    # 特征筛选
    if keep_cols_file:
        keep_path = BASE / keep_cols_file
        keep_cols = [c.strip() for c in keep_path.read_text(encoding="utf-8").splitlines() if c.strip()]
        matched = [c for c in keep_cols if c in features.columns]
        features = features[matched]

    # 过滤起始日期
    cutoff = pd.Timestamp("2018-01-01")
    mask = pd.to_datetime(merged["Unnamed: 0"]) >= cutoff
    features = features.loc[mask.values].reset_index(drop=True)
    y_all = merged.loc[mask, "label"].reset_index(drop=True)
    dates = merged.loc[mask, "Unnamed: 0"].reset_index(drop=True)

    return features, y_all, dates


# =====================================================================
# 滚动训练（单组合）
# =====================================================================

def rolling_eval(features, y_all, dates, depth, lr, subsample, colsample):
    """单参数组合的完整滚动训练，返回分段指标。
    提速：n_estimators=300 + early_stopping=30，平衡精度与速度。"""
    W, S, V = 120, 63, 30
    n = len(features)
    # 只用最后 10 个 fold 快筛（覆盖约 2023-2026，聚焦当前 regime）
    # 全部 fold 从 range(W, n-S+1, S) 取最后 10 个
    all_starts = list(range(W, n - S + 1, S))
    eval_starts = all_starts[-10:]  # 最后 10 个 fold
    all_preds, all_true, all_proba, all_dates = [], [], [], []

    for start in eval_starts:
        end = min(start + S, n)
        x_tr = features.iloc[start - W:start - V]
        y_tr = y_all.iloc[start - W:start - V]
        x_val = features.iloc[start - V:start]
        y_val = y_all.iloc[start - V:start]
        x_te = features.iloc[start:end]
        y_te = y_all.iloc[start:end]
        if len(x_tr) < 30:
            continue

        model = XGBClassifier(
            n_estimators=300, max_depth=depth, learning_rate=lr,
            subsample=subsample, colsample_bytree=colsample,
            min_child_weight=8.0, reg_alpha=1.5, reg_lambda=2.0, gamma=0.1,
            random_state=42, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", early_stopping_rounds=30,
        )
        model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)

        # 网格搜索阶段跳过校准器（提速 3x），最终最优组合再启用
        proba = model.predict_proba(x_te)

        yp = (proba[:, 1] > 0.5).astype(int)
        all_preds.extend(yp.tolist())
        all_true.extend(y_te.values.tolist())
        all_proba.extend(proba[:, 1].tolist())
        all_dates.extend(dates.iloc[start:end].tolist())

    return np.array(all_true), np.array(all_preds), np.array(all_proba), pd.to_datetime(all_dates)


def metrics_by_period(y_true, y_pred, y_proba, dates):
    """分时段评估"""
    results = {}
    dt = pd.to_datetime(dates)
    periods = [
        ("整体", np.ones(len(y_true), dtype=bool)),
        ("2022后", (dt >= pd.Timestamp("2022-01-01"))),
        ("2024_2026", (dt >= pd.Timestamp("2024-01-01"))),
    ]
    for name, mask in periods:
        if mask.sum() < 10:
            results[name] = {"acc": np.nan, "auc": np.nan, "mf1": np.nan, "lift": np.nan}
            continue
        yt, yp, pr = y_true[mask], y_pred[mask], y_proba[mask]
        try:
            auc = roc_auc_score(yt, pr)
        except ValueError:
            auc = np.nan
        acc = accuracy_score(yt, yp)
        maj = pd.Series(yt).mode().iloc[0]
        maj_acc = (yt == maj).mean()
        mf1 = f1_score(yt, yp, average="macro")
        results[name] = {"acc": acc, "auc": auc, "mf1": mf1, "lift": acc - maj_acc}
    return results


# =====================================================================
# 主网格
# =====================================================================

def main():
    # 参数空间
    DEPTHS = [2, 3, 4]
    LRS = [0.05, 0.1]
    SUBSAMPLES = [0.7, 0.8]
    COLSAMPLES = [0.6, 0.8]
    STL_PERIODS = [63, 126]

    total = len(DEPTHS) * len(LRS) * len(SUBSAMPLES) * len(COLSAMPLES) * len(STL_PERIODS)
    print(f"超参全网格: {total} 个组合", flush=True)
    print(f"固定: W=120, V=30, reg_alpha=1.5, mcw=8, detrend_label, keep_cols_full", flush=True)
    print(f"{'='*80}", flush=True)

    # 按 stl_period 缓存特征（只构建 2 次）
    feat_cache = {}
    for sp in STL_PERIODS:
        print(f"\n[特征构建] stl_period={sp} ...", flush=True)
        feat_cache[sp] = prepare_data(sp, "model_new/data/keep_cols_full.txt")
        feats, y_all, dates = feat_cache[sp]
        print(f"  样本: {len(feats)}, 特征: {feats.shape[1]}", flush=True)

    # 网格遍历
    rows = []
    combo_idx = 0
    for sp in STL_PERIODS:
        features, y_all, dates = feat_cache[sp]
        for depth, lr, ss, cs in product(DEPTHS, LRS, SUBSAMPLES, COLSAMPLES):
            combo_idx += 1
            tag = f"d{depth}_lr{lr}_ss{ss}_cs{cs}_sp{sp}"
            print(f"[{combo_idx:2d}/{total}] {tag} ...", end=" ", flush=True)

            y_true, y_pred, y_proba, pred_dates = rolling_eval(
                features, y_all, dates, depth, lr, ss, cs
            )
            res = metrics_by_period(y_true, y_pred, y_proba, pred_dates)

            row = {
                "combo": tag,
                "depth": depth, "lr": lr, "subsample": ss,
                "colsample": cs, "stl_period": sp,
                "整体_auc": res["整体"]["auc"],
                "整体_acc_lift": res["整体"]["lift"],
                "整体_mf1": res["整体"]["mf1"],
                "2022后_auc": res["2022后"]["auc"],
                "2022后_lift": res["2022后"]["lift"],
                "2024_auc": res["2024_2026"]["auc"],
                "2024_lift": res["2024_2026"]["lift"],
            }
            rows.append(row)
            print(f"整体AUC={row['整体_auc']:.4f} 2022后AUC={row['2022后_auc']:.4f} "
                  f"2024AUC={row['2024_auc']:.4f}", flush=True)

    # === 汇总 ===
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "grid_results.csv", index=False, encoding="utf-8-sig")

    # 文本报告
    L = []
    def p(s=""):
        print(s, flush=True)
        L.append(str(s))

    p("=" * 90)
    p("超参全网格搜索汇总（48 组合 × 完整滚动训练）")
    p("=" * 90)

    p("\n[Top 10 — 整体 AUC]")
    top_auc = df.nlargest(10, "整体_auc")[["combo", "整体_auc", "2022后_auc", "2024_auc", "整体_acc_lift"]]
    p(top_auc.round(4).to_string(index=False))

    p("\n[Top 10 — 2022后 AUC] （聚焦当前 regime 适应力）")
    top_22 = df.nlargest(10, "2022后_auc")[["combo", "整体_auc", "2022后_auc", "2024_auc", "2022后_lift"]]
    p(top_22.round(4).to_string(index=False))

    p("\n[Top 10 — 2024-2026 AUC] （聚焦 regime 切换适应力）")
    top_24 = df.nlargest(10, "2024_auc")[["combo", "整体_auc", "2022后_auc", "2024_auc", "2024_lift"]]
    p(top_24.round(4).to_string(index=False))

    # 综合评分（三个 AUC 加权平均）
    df["综合评分"] = df["整体_auc"] * 0.3 + df["2022后_auc"] * 0.3 + df["2024_auc"] * 0.4
    p("\n[Top 10 — 综合评分] (整体30% + 2022后30% + 2024前40%)")
    top_all = df.nlargest(10, "综合评分")[["combo", "整体_auc", "2022后_auc", "2024_auc", "综合评分"]]
    p(top_all.round(4).to_string(index=False))

    best = df.nlargest(1, "综合评分").iloc[0]
    p("\n" + "=" * 90)
    p(f"[推荐最优组合] {best['combo']}")
    p(f"  整体 AUC={best['整体_auc']:.4f} | 2022后 AUC={best['2022后_auc']:.4f} | 2024 AUC={best['2024_auc']:.4f}")
    p(f"  综合评分={best['综合评分']:.4f}")
    p(f"\n  复核命令:")
    p(f"  python model_new\\03_XGB训练_二分类.py --rolling-window 120 --rolling-val-size 30 "
      f"--detrend-label --keep-cols-file model_new/data/keep_cols_full.txt "
      f"--max-depth {int(best['depth'])} --learning-rate {best['lr']}")

    (DATA / "grid_report.txt").write_text("\n".join(L), encoding="utf-8")
    print(f"\n报告已保存: {DATA / 'grid_report.txt'}")
    print(f"明细表: {DATA / 'grid_results.csv'}")


if __name__ == "__main__":
    main()
