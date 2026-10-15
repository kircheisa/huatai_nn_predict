# -*- coding: utf-8 -*-
"""
滚动窗口大小批量对比测试

测试不同窗口大小（90/120/180/270/360/756）对二分类效果的影响。
固定 step=63, val=63，唯一变量是训练窗口大小 W。

复用 43_3 脚本中的特征工程与模型构建逻辑。
输出到 _window_compare.txt
"""
import sys, os, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

os.chdir(Path(__file__).resolve().parent)
BASE = Path(".").resolve()

# 动态导入 43_3 脚本中的函数（避免重复造轮子）
spec = importlib.util.spec_from_file_location(
    "train_bin",
    str(BASE / "43_3_华泰证券债券预测_XGB训练_二分类.py"),
)
train_bin = importlib.util.module_from_spec(spec)
# 屏蔽其 main() 执行
import types
sys.argv = ["dummy"]  # 防止 parse_args 读取命令行
spec.loader.exec_module(train_bin)


def gen_label(v):
    """二分类标签：下行=1, 上行=0"""
    if pd.isna(v):
        return np.nan
    return 1 if v < 0 else 0


def run_one_window(features, y_all, merged, W, S=63, V=63):
    """
    对单个窗口大小 W 执行滚动训练，返回核心指标。
    训练样本数 = W - V（验证集从窗口末尾切出）
    """
    n = len(features)
    all_preds, all_probas, all_true = [], [], []
    fold = 0
    train_size = W - V  # 实际训练样本数

    for start in range(W, n - S + 1, S):
        end = min(start + S, n)
        x_tr = features.iloc[start - W:start - V]
        y_tr = y_all.iloc[start - W:start - V]
        x_val = features.iloc[start - V:start]
        y_val = y_all.iloc[start - V:start]
        x_te = features.iloc[start:end]
        y_te = y_all.iloc[start:end]
        if len(x_tr) < 50:  # 训练样本太少则跳过
            continue

        m = XGBClassifier(
            n_estimators=600, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=8.0,
            reg_alpha=0.5, reg_lambda=2.0, gamma=0.1,
            random_state=42, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist",
            early_stopping_rounds=50,
        )
        m.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)

        yp = m.predict(x_te)
        proba = m.predict_proba(x_te)
        all_preds.extend(yp.tolist())
        all_probas.extend(proba.tolist())
        all_true.extend(y_te.values.tolist())
        fold += 1

    if not all_true:
        return None

    y_pred = np.array(all_preds)
    y_true = np.array(all_true)
    y_proba = np.array(all_probas)

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    except ValueError:
        auc = float("nan")
    maj = int(pd.Series(y_true).mode().iloc[0])
    maj_acc = float(np.mean(y_true == maj))

    return {
        "W": W,
        "train_size": train_size,
        "folds": fold,
        "n_test": len(y_true),
        "accuracy": acc,
        "macro_f1": mf1,
        "auc": auc,
        "baseline": maj_acc,
        "acc_vs_baseline": acc - maj_acc,
    }


def main():
    # ============ 1. 加载数据 ============
    print("[1/3] 加载数据并构建特征...", flush=True)
    df_data = pd.read_csv("data/43_data.csv")
    df_rate = pd.read_csv("data/43_rate.csv")
    df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"])
    df_rate["time"] = pd.to_datetime(df_rate["time"])
    merged = pd.merge(df_data, df_rate, left_on="Unnamed: 0", right_on="time", how="inner").sort_values("Unnamed: 0")
    merged["target_date_5d"] = merged["Unnamed: 0"].shift(-5)
    merged["rate_delta_5d"] = merged["rate"].shift(-5) - merged["rate"]
    merged["label"] = merged["rate_delta_5d"].apply(gen_label)
    merged = merged.dropna(subset=["label"]).copy()
    merged["label"] = merged["label"].astype(int)
    y_all = merged["label"].copy()

    # 构建特征（含技术面动量）
    features = train_bin.build_features(merged)

    # 合并遗传特征
    gp_file = BASE / "data/43_x.csv"
    if gp_file.exists():
        gp = pd.read_csv(gp_file, index_col=0)
        gp.index = pd.to_datetime(gp.index)
        gp = gp.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        keep_cols = ["gp_feature_1", "gp_feature_2", "gp_feature_3", "gp_feature_4", "gp_feature_8"]
        gp = gp[[c for c in keep_cols if c in gp.columns]]
        features = features.join(gp, how="left")
        gp_cols = [c for c in gp.columns if c in features.columns]
        features[gp_cols] = features[gp_cols].fillna(0.0)

    # 过滤起始日期
    cutoff = pd.to_datetime("2018-01-01")
    mask = pd.to_datetime(merged["Unnamed: 0"]) >= cutoff
    features = features.loc[mask.values].reset_index(drop=True)
    y_all = y_all.loc[mask.values].reset_index(drop=True)
    print(f"  样本: {len(features)}, 特征数: {features.shape[1]}\n", flush=True)

    # ============ 2. 批量测试不同窗口 ============
    print("[2/3] 批量测试窗口大小...", flush=True)
    windows = [90, 120, 180, 270, 360, 756]
    results = []
    for W in windows:
        print(f"  测试 W={W} ...", end="", flush=True)
        r = run_one_window(features, y_all, merged, W)
        if r is None:
            print(" 样本不足，跳过", flush=True)
            continue
        results.append(r)
        print(f"  Acc={r['accuracy']:.4f}  F1={r['macro_f1']:.4f}  AUC={r['auc']:.4f}  (folds={r['folds']})", flush=True)

    # ============ 3. 输出对比表 ============
    print("[3/3] 生成对比报告...", flush=True)
    lines = []
    lines.append("=" * 90)
    lines.append("滚动窗口大小对比测试（二分类，下行=1/上行=0）")
    lines.append(f"固定: step=63, val=63, 特征数={features.shape[1]}, 数据起始=2018-01-01")
    lines.append("=" * 90)
    header = f"{'窗口W':>6} {'训练样本':>8} {'folds':>6} {'测试样本':>8} {'Accuracy':>10} {'MacroF1':>10} {'AUC':>8} {'基线Acc':>10} {'vs基线':>10}"
    lines.append(header)
    lines.append("-" * 90)
    for r in results:
        row = (f"{r['W']:>6} {r['train_size']:>8} {r['folds']:>6} {r['n_test']:>8} "
               f"{r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['auc']:>8.4f} "
               f"{r['baseline']:>10.4f} {r['acc_vs_baseline']:>+10.4f}")
        lines.append(row)
    lines.append("-" * 90)

    # 找最优
    best = max(results, key=lambda x: x["auc"])
    lines.append(f"\n最高 AUC:    W={best['W']}  AUC={best['auc']:.4f}  Acc={best['accuracy']:.4f}  F1={best['macro_f1']:.4f}")
    best_f1 = max(results, key=lambda x: x["macro_f1"])
    lines.append(f"最高 MacroF1: W={best_f1['W']}  F1={best_f1['macro_f1']:.4f}  Acc={best_f1['accuracy']:.4f}  AUC={best_f1['auc']:.4f}")

    report = "\n".join(lines)
    (BASE / "_window_compare.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("\nDONE -> _window_compare.txt")


if __name__ == "__main__":
    main()
