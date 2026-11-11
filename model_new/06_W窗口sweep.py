# -*- coding: utf-8 -*-
"""
W 滚动窗口 sweep + 双轨评估

针对诊断结论（W=504 太长导致 2025 regime 切换滞后），sweep 以下配置：
  W ∈ {90, 120, 150, 180, 210}
  start_date ∈ {2018-01-01, 2022-01-01}  （前者保留全样本，后者聚焦当前 regime）

每个配置跑完整滚动训练，输出：
  - 整体 / 2022前 / 2022后 / 2024-2026 四段指标（Acc, MacroF1, AUC, 下行占比偏差）
  - 汇总表 sweep_results.csv + sweep_report.txt

注意：为控制运行时间，sweep 阶段使用 n_estimators=400（配合 early_stopping），
      最终最优 W 确定后再用完整 03_XGB训练_二分类.py 跑默认 600 复核。
"""

import sys
import warnings
import importlib.util
from pathlib import Path

# 抑制 03 模块 build_features 的 DataFrame fragmentation 警告（仅噪音）
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "model_new" / "data"

# ----------------------------------------------------------------
# 动态导入 03 模块（中文文件名，用 importlib 加载）
# ----------------------------------------------------------------
_mod_path = BASE / "model_new" / "03_XGB训练_二分类.py"
_spec = importlib.util.spec_from_file_location("xgb_bin", _mod_path)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

# 复用 03 模块的核心函数：build_features / load_genetic_features / create_xgb_model / gen_label
build_features = m.build_features
load_genetic_features = m.load_genetic_features
create_xgb_model = m.create_xgb_model
gen_label = m.gen_label


# =====================================================================
# 轻量滚动训练：只返回预测明细，不保存模型/特征（sweep 用，提速）
# =====================================================================

def rolling_train_light(features, y_all, dates, args):
    """
    轻量滚动训练，返回 DataFrame[date, y_true, y_pred, prob_1]。
    与 03 模块 rolling_window_train 逻辑一致，去掉持久化部分。
    """
    W, S, V = args.rolling_window, args.rolling_step, args.rolling_val_size
    n = len(features)
    rows = []
    cols = ["date", "y_true", "y_pred", "prob_1"]

    for start in range(W, n - S + 1, S):
        end = min(start + S, n)
        x_tr = features.iloc[start - W:start - V]
        y_tr = y_all.iloc[start - W:start - V]
        x_val = features.iloc[start - V:start]
        y_val = y_all.iloc[start - V:start]
        x_te = features.iloc[start:end]
        y_te = y_all.iloc[start:end]
        if len(x_tr) < 30:
            continue

        model = create_xgb_model(args)
        model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)

        yp = model.predict(x_te)
        proba = model.predict_proba(x_te)
        for i in range(len(y_te)):
            rows.append({
                "date": dates[start + i],
                "y_true": int(y_te.iloc[i]),
                "y_pred": int(yp[i]),
                "prob_1": float(proba[i, 1]),
            })
    return pd.DataFrame(rows, columns=cols)


# =====================================================================
# 分时段评估
# =====================================================================

def eval_block(df, name):
    """计算单时段指标，返回 dict；样本不足返回 None"""
    if len(df) < 10:
        return None
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    prob1 = df["prob_1"].values
    try:
        auc = roc_auc_score(y_true, prob1)
    except ValueError:
        auc = float("nan")
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    maj = int(pd.Series(y_true).mode().iloc[0])
    maj_acc = float(np.mean(y_true == maj))
    return {
        "name": name, "n": len(df),
        "acc": acc, "baseline": maj_acc, "acc_lift": acc - maj_acc,
        "macro_f1": mf1, "auc": auc,
        "true_down": y_true.mean(), "pred_down": y_pred.mean(),
        "down_bias": y_pred.mean() - y_true.mean(),
    }


def evaluate_periods(pred_df):
    """对一组预测做四段评估"""
    dt = pd.to_datetime(pred_df["date"])
    blocks = [
        ("整体", pred_df),
        ("2022前", pred_df[dt < pd.Timestamp("2022-01-01")]),
        ("2022后", pred_df[dt >= pd.Timestamp("2022-01-01")]),
        ("2024-2026", pred_df[dt >= pd.Timestamp("2024-01-01")]),
    ]
    return [eval_block(d, n) for n, d in blocks]


# =====================================================================
# 构造一次 args（用 SimpleNamespace 替代 argparse）
# =====================================================================

from types import SimpleNamespace

def make_args(W, start_date, n_estimators=400):
    # 验证集按窗口比例取（约 1/4），保证不同 W 下训练/验证划分可比
    # W=90→V=22, W=120→V=30, W=150→V=37, W=180→V=45, W=210→V=52
    V = max(21, W // 4)
    return SimpleNamespace(
        start_date=start_date,
        rolling_window=W,
        rolling_step=63,
        rolling_val_size=V,
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=8.0,
        reg_alpha=1.0,
        genetic_features_file="model_new/data/x_genetic_bin.csv",
        no_genetic=False,
        keep_cols_file=None,
    )


# =====================================================================
# 数据准备（只做一次，所有 sweep 复用）
# =====================================================================

def prepare_data():
    print("[准备] 加载数据并构建特征（仅一次）...", flush=True)
    df_data = pd.read_csv(DATA / "data.csv")
    df_rate = pd.read_csv(DATA / "rate.csv")
    df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"])
    df_rate["time"] = pd.to_datetime(df_rate["time"])
    merged = pd.merge(df_data, df_rate, left_on="Unnamed: 0", right_on="time", how="inner").sort_values("Unnamed: 0")
    merged["target_date_5d"] = merged["Unnamed: 0"].shift(-5)
    merged["rate_delta_5d"] = merged["rate"].shift(-5) - merged["rate"]
    merged["label"] = merged["rate_delta_5d"].apply(gen_label)
    merged = merged.dropna(subset=["label"]).copy()
    merged["label"] = merged["label"].astype(int)

    features = build_features(merged)
    # 遗传特征（用默认 args 加载）
    gp_args = SimpleNamespace(
        genetic_features_file="model_new/data/x_genetic_bin.csv",
        no_genetic=False,
    )
    gp = load_genetic_features(gp_args)
    if gp is not None:
        features = features.join(gp, how="left")
        gp_cols = [c for c in gp.columns if c in features.columns]
        features[gp_cols] = features[gp_cols].fillna(0.0)

    print(f"[准备] 全量样本: {len(features)} 行, 特征数: {features.shape[1]}", flush=True)
    return merged, features


def slice_by_start(merged, features, start_date):
    """按 start_date 切片（返回 features_full, y_all, dates 用于滚动训练）"""
    cutoff = pd.to_datetime(start_date)
    mask = pd.to_datetime(merged["Unnamed: 0"]) >= cutoff
    feats = features.loc[mask.values].reset_index(drop=True)
    y_all = merged.loc[mask, "label"].reset_index(drop=True)
    dates = merged.loc[mask, "Unnamed: 0"].reset_index(drop=True)
    return feats, y_all, dates


# =====================================================================
# 主 sweep
# =====================================================================

def main():
    merged, features_full = prepare_data()

    W_LIST = [90, 120, 150, 180, 210]
    START_LIST = ["2018-01-01", "2022-01-01"]

    all_blocks = []   # 每个配置×每段
    summary_rows = []  # 每个配置整体一行

    for start_date in START_LIST:
        feats, y_all, dates = slice_by_start(merged, features_full, start_date)
        print(f"\n=== start_date={start_date} | 可用样本 {len(feats)} ===", flush=True)

        for W in W_LIST:
            # W 过大导致训练样本不足时跳过
            if W >= len(feats) - 63:
                print(f"  W={W}: 样本不足，跳过", flush=True)
                continue
            args = make_args(W, start_date)
            print(f"  >>> 训练 W={W} step=63 val={args.rolling_val_size} (train={W-args.rolling_val_size}) ...", flush=True)
            pred_df = rolling_train_light(feats, y_all, dates, args)
            if len(pred_df) == 0:
                print(f"      !!! 无有效 fold（训练样本不足），跳过", flush=True)
                continue
            blocks = evaluate_periods(pred_df)

            tag = f"W{W}_{start_date[:4]}"
            for b in blocks:
                if b is None:
                    continue
                b["config"] = tag
                b["W"] = W
                b["start_date"] = start_date
                all_blocks.append(b)

            # 整体指标做摘要
            overall = blocks[0]
            b22 = blocks[3]  # 2024-2026
            summary_rows.append({
                "config": tag, "W": W, "start": start_date,
                "n": overall["n"],
                "整体_acc": overall["acc"], "整体_baseline": overall["baseline"],
                "整体_lift": overall["acc_lift"],
                "整体_macroF1": overall["macro_f1"], "整体_AUC": overall["auc"],
                "2024_AUC": b22["auc"] if b22 else np.nan,
                "2024_acc": b22["acc"] if b22 else np.nan,
                "2024_lift": b22["acc_lift"] if b22 else np.nan,
            })
            ov = overall
            print(f"      整体 Acc={ov['acc']:.4f}(基线{ov['baseline']:.4f},lift{ov['acc_lift']:+.4f}) "
                  f"MacroF1={ov['macro_f1']:.4f} AUC={ov['auc']:.4f} | "
                  f"2024-26 Acc={(b22['acc'] if b22 else np.nan):.4f} AUC={(b22['auc'] if b22 else np.nan):.4f}",
                  flush=True)

    # === 汇总 ===
    blocks_df = pd.DataFrame(all_blocks)
    summary_df = pd.DataFrame(summary_rows).sort_values(["start", "W"])

    blocks_df.to_csv(DATA / "sweep_blocks.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(DATA / "sweep_results.csv", index=False, encoding="utf-8-sig")

    # 文本报告
    L = []
    def p(s=""):
        print(s, flush=True)
        L.append(str(s))

    p("=" * 80)
    p("W 窗口 sweep 汇总报告")
    p(f"step=63, val=63, n_estimators=400(sweep提速用), start_date ∈ {{2018, 2022}}")
    p("=" * 80)

    p("\n[整体指标对比] 按 acc_lift (相对基线提升) 排序")
    p(summary_df.sort_values("整体_lift", ascending=False).round(4).to_string(index=False))

    p("\n[2024-2026 段表现对比] （聚焦 regime 切换适应力）")
    p(summary_df.sort_values("2024_AUC", ascending=False).round(4).to_string(index=False))

    p("\n[各配置 × 各时段明细]")
    for cfg in blocks_df["config"].unique():
        p(f"\n--- {cfg} ---")
        sub = blocks_df[blocks_df["config"] == cfg][["name", "n", "acc", "baseline", "acc_lift",
                                                      "macro_f1", "auc", "true_down", "pred_down", "down_bias"]]
        p(sub.round(4).to_string(index=False))

    # 推荐结论
    p("\n" + "=" * 80)
    p("[推荐]")
    best_overall = summary_df.loc[summary_df["整体_lift"].idxmax()]
    best_2024 = summary_df.loc[summary_df["2024_AUC"].idxmax()]
    p(f"  整体 lift 最高: {best_overall['config']} (lift={best_overall['整体_lift']:+.4f})")
    p(f"  2024-2026 AUC 最高: {best_2024['config']} (AUC={best_2024['2024_AUC']:.4f})")
    p("\n  决策建议：优先看 2024-2026 段 AUC 与 lift（regime 适应力），")
    p("  整体指标作参考。确定最优 W 后用 03_XGB训练_二分类.py 跑 n_estimators=600 复核。")

    (DATA / "sweep_report.txt").write_text("\n".join(L), encoding="utf-8")
    print(f"\n报告已保存: {DATA / 'sweep_report.txt'}")
    print(f"汇总表: {DATA / 'sweep_results.csv'}")


if __name__ == "__main__":
    main()
