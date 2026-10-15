# -*- coding: utf-8 -*-
"""
批量调参脚本 —— 按优先级分组测试，固定 W=180

分组测试顺序（每组只改一个维度，其余用当前最优值）：
  阶段1：滚动步长 step [21, 42, 63]
  阶段2：max_depth [2, 3, 4]
  阶段3：learning_rate [0.03, 0.05, 0.1]
  阶段4：min_child_weight [3, 5, 8]
  阶段5：reg_alpha [0.1, 0.5, 1.0]

特征只构建一次（225 列含 Hurst/FFT 计算较慢），后续所有参数组合复用。
输出到 _tuning_results.txt
"""
import sys, os, importlib.util, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.chdir(Path(__file__).resolve().parent)
BASE = Path(".").resolve()

# 动态导入 43_3 中的特征工程函数
sys.argv = ["dummy"]
spec = importlib.util.spec_from_file_location(
    "train_bin", str(BASE / "43_3_华泰证券债券预测_XGB训练_二分类.py"))
train_bin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_bin)


def gen_label(v):
    if pd.isna(v): return np.nan
    return 1 if v < 0 else 0


def run_one_config(features, y_all, merged, W=180, S=63, V=63,
                   max_depth=3, n_estimators=600, learning_rate=0.05,
                   min_child_weight=8.0, reg_alpha=0.5, reg_lambda=2.0,
                   gamma=0.1, subsample=0.8, colsample_bytree=0.8):
    """
    单次滚动训练，返回核心指标 dict。
    """
    n = len(features)
    all_preds, all_probas, all_true = [], [], []
    fold = 0

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

        m = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, gamma=gamma,
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
    return {"folds": fold, "n_test": len(y_true), "accuracy": acc,
            "macro_f1": mf1, "auc": auc, "baseline": maj_acc}


def print_row(r, label):
    """格式化输出一行结果"""
    print(f"  {label:<45} folds={r['folds']:>3}  "
          f"Acc={r['accuracy']:.4f}  F1={r['macro_f1']:.4f}  "
          f"AUC={r['auc']:.4f}  (base={r['baseline']:.4f})", flush=True)


def main():
    t0 = time.time()
    # 清空旧日志
    (BASE / "_tuning_log.txt").write_text("", encoding="utf-8")

    # ============ 1. 加载数据 + 构建特征（只做一次）============
    print("=" * 70, flush=True)
    print("[准备] 加载数据并构建特征（225列，含Hurst/FFT）...", flush=True)
    print("=" * 70, flush=True)
    df_data = pd.read_csv("data/43_data.csv")
    df_rate = pd.read_csv("data/43_rate.csv")
    df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"])
    df_rate["time"] = pd.to_datetime(df_rate["time"])
    merged = pd.merge(df_data, df_rate, left_on="Unnamed: 0", right_on="time", how="inner").sort_values("Unnamed: 0")
    merged["rate_delta_5d"] = merged["rate"].shift(-5) - merged["rate"]
    merged["label"] = merged["rate_delta_5d"].apply(gen_label)
    merged = merged.dropna(subset=["label"]).copy()
    merged["label"] = merged["label"].astype(int)
    y_all = merged["label"].copy()

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

    cutoff = pd.to_datetime("2018-01-01")
    mask = pd.to_datetime(merged["Unnamed: 0"]) >= cutoff
    features = features.loc[mask.values].reset_index(drop=True)
    y_all = y_all.loc[mask.values].reset_index(drop=True)
    merged = merged.loc[mask].reset_index(drop=True)
    print(f"  样本: {len(features)}, 特征数: {features.shape[1]}\n", flush=True)

    # 当前最优参数（逐步更新）
    best = {"W": 180, "S": 63, "V": 63, "max_depth": 3, "n_estimators": 600,
            "learning_rate": 0.05, "min_child_weight": 8.0,
            "reg_alpha": 0.5, "reg_lambda": 2.0, "gamma": 0.1}
    all_results = []  # 收集所有结果用于最终排名

    def run_and_record(label, **overrides):
        """用 best 参数为基础，应用 overrides 后运行"""
        params = {**best, **overrides}
        r = run_one_config(features, y_all, merged, **params)
        if r:
            r["label"] = label
            r["params"] = str(params)
            all_results.append(r)
            print_row(r, label)
            # 同时追加写入日志文件，防止 shell 重定向丢输出
            with open(BASE / "_tuning_log.txt", "a", encoding="utf-8") as f:
                f.write(f"{label:<45} folds={r['folds']:>3}  "
                        f"Acc={r['accuracy']:.4f}  F1={r['macro_f1']:.4f}  "
                        f"AUC={r['auc']:.4f}  (base={r['baseline']:.4f})\n")
        return r

    # ============ 阶段1：滚动步长 step ============
    print("=" * 70, flush=True)
    print("[阶段1] 测试滚动步长 step (W=180, V=63)", flush=True)
    print("=" * 70, flush=True)
    for s in [42, 63]:
        run_and_record(f"step={s}", S=s)
    # 选最优
    best_step = max(
        [r for r in all_results if "step=" in r["label"]],
        key=lambda x: x["auc"]
    )
    best["S"] = 21 if "21" in best_step["label"] else (42 if "42" in best_step["label"] else 63)
    print(f"\n  >>> 阶段1最优: step={best['S']} (AUC={best_step['auc']:.4f})\n", flush=True)

    # ============ 阶段2：max_depth ============
    print("=" * 70, flush=True)
    print(f"[阶段2] 测试 max_depth (W=180, step={best['S']})", flush=True)
    print("=" * 70, flush=True)
    for d in [2, 3, 4]:
        run_and_record(f"max_depth={d}", max_depth=d)
    best_d = max(
        [r for r in all_results if r["label"].startswith("max_depth=")],
        key=lambda x: x["auc"]
    )
    best["max_depth"] = int(best_d["label"].split("=")[1])
    print(f"\n  >>> 阶段2最优: max_depth={best['max_depth']} (AUC={best_d['auc']:.4f})\n", flush=True)

    # ============ 阶段3：learning_rate ============
    print("=" * 70, flush=True)
    print(f"[阶段3] 测试 learning_rate (depth={best['max_depth']}, step={best['S']})", flush=True)
    print("=" * 70, flush=True)
    for lr in [0.03, 0.05, 0.1]:
        run_and_record(f"lr={lr}", learning_rate=lr)
    best_lr = max(
        [r for r in all_results if r["label"].startswith("lr=")],
        key=lambda x: x["auc"]
    )
    best["learning_rate"] = float(best_lr["label"].split("=")[1])
    print(f"\n  >>> 阶段3最优: lr={best['learning_rate']} (AUC={best_lr['auc']:.4f})\n", flush=True)

    # ============ 阶段4：min_child_weight ============
    print("=" * 70, flush=True)
    print(f"[阶段4] 测试 min_child_weight (depth={best['max_depth']}, lr={best['learning_rate']})", flush=True)
    print("=" * 70, flush=True)
    for mcw in [3, 5, 8]:
        run_and_record(f"min_child_weight={mcw}", min_child_weight=float(mcw))
    best_mcw = max(
        [r for r in all_results if r["label"].startswith("min_child_weight=")],
        key=lambda x: x["auc"]
    )
    best["min_child_weight"] = float(best_mcw["label"].split("=")[1])
    print(f"\n  >>> 阶段4最优: min_child_weight={best['min_child_weight']} (AUC={best_mcw['auc']:.4f})\n", flush=True)

    # ============ 阶段5：reg_alpha ============
    print("=" * 70, flush=True)
    print(f"[阶段5] 测试 reg_alpha (mcw={best['min_child_weight']})", flush=True)
    print("=" * 70, flush=True)
    for ra in [0.1, 0.5, 1.0]:
        run_and_record(f"reg_alpha={ra}", reg_alpha=ra)
    best_ra = max(
        [r for r in all_results if r["label"].startswith("reg_alpha=")],
        key=lambda x: x["auc"]
    )
    best["reg_alpha"] = float(best_ra["label"].split("=")[1])
    print(f"\n  >>> 阶段5最优: reg_alpha={best['reg_alpha']} (AUC={best_ra['auc']:.4f})\n", flush=True)

    # ============ 汇总报告 ============
    elapsed = time.time() - t0
    print("\n" + "=" * 70, flush=True)
    print("最终调参结果汇总", flush=True)
    print("=" * 70, flush=True)

    lines = []
    lines.append("=" * 80)
    lines.append("批量调参结果汇总（W=180, 特征225, 二分类）")
    lines.append(f"总耗时: {elapsed/60:.1f} 分钟")
    lines.append("=" * 80)

    for stage_name, prefix in [("阶段1: step", "step="), ("阶段2: max_depth", "max_depth="),
                                ("阶段3: learning_rate", "lr="), ("阶段4: min_child_weight", "min_child_weight="),
                                ("阶段5: reg_alpha", "reg_alpha=")]:
        stage_results = [r for r in all_results if r["label"].startswith(prefix)]
        lines.append(f"\n--- {stage_name} ---")
        lines.append(f"{'参数':<20} {'folds':>6} {'Accuracy':>10} {'MacroF1':>10} {'AUC':>10} {'基线':>10}")
        for r in sorted(stage_results, key=lambda x: -x["auc"]):
            lines.append(f"{r['label']:<20} {r['folds']:>6} {r['accuracy']:>10.4f} "
                        f"{r['macro_f1']:>10.4f} {r['auc']:>10.4f} {r['baseline']:>10.4f}")

    # 最终最优组合
    lines.append("\n" + "=" * 80)
    lines.append("逐步最优组合（贪心搜索）:")
    lines.append(f"  step={best['S']}, max_depth={best['max_depth']}, "
                f"lr={best['learning_rate']}, min_child_weight={best['min_child_weight']}, "
                f"reg_alpha={best['reg_alpha']}")

    # 全局 Top 5
    lines.append("\n全局 Top 5 (按 AUC):")
    for i, r in enumerate(sorted(all_results, key=lambda x: -x["auc"])[:5]):
        lines.append(f"  {i+1}. {r['label']:<20} AUC={r['auc']:.4f}  F1={r['macro_f1']:.4f}  Acc={r['accuracy']:.4f}")

    report = "\n".join(lines)
    (BASE / "_tuning_results.txt").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)
    print(f"\nDONE -> _tuning_results.txt ({elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
