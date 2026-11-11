# -*- coding: utf-8 -*-
"""
双周模型快速训练 A/B 对比 —— 复用预存特征，绕过 STL 内存瓶颈
标签用简单方向（rate_delta < 0 → 下行=1），不依赖 STL
"""
import gc
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parent.parent / "model_new" / "data"

def load_data():
    """
    加载数据 —— 正确合并 data.csv + rate.csv，过滤到 2018+
    标签用 horizon=10 的简单方向（rate_delta_10d < 0 → 下行=1）
    """
    features = pd.read_csv(DATA / "xgb_train_features_bin.csv", index_col=0)
    keep_path = DATA / "keep_cols_full.txt"
    keep_cols = [c.strip() for c in keep_path.read_text(encoding="utf-8").splitlines() if c.strip()]
    features = features[[c for c in keep_cols if c in features.columns]]

    # 正确合并 data + rate（与主脚本相同逻辑）
    df_data = pd.read_csv(DATA / "data.csv")
    df_rate = pd.read_csv(DATA / "rate.csv")
    df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"])
    df_rate["time"] = pd.to_datetime(df_rate["time"])
    merged = pd.merge(df_data, df_rate, left_on="Unnamed: 0", right_on="time", how="inner").sort_values("Unnamed: 0")

    H = 10
    merged[f"rate_delta_{H}d"] = merged["rate"].shift(-H) - merged["rate"]
    y_all = (merged[f"rate_delta_{H}d"] < 0).astype(float)

    # 过滤到 2018+（与主脚本 start_date=2018-01-01 一致）
    cutoff = pd.to_datetime("2018-01-01")
    mask = merged["Unnamed: 0"] >= cutoff
    merged = merged.loc[mask].reset_index(drop=True)
    y_all = y_all.loc[mask].reset_index(drop=True)

    # dropna（先截断到相同长度，避免索引错位）
    n = min(len(features), len(y_all))
    features = features.iloc[:n]
    y_all = y_all.iloc[:n]
    merged = merged.iloc[:n]
    valid = y_all.notna()
    features = features.loc[valid].reset_index(drop=True)
    y_all = y_all.loc[valid].reset_index(drop=True)
    dates = merged.loc[valid, "Unnamed: 0"].values

    # 死区降权
    delta_bp = merged[f"rate_delta_{H}d"].abs() * 100
    delta_bp = delta_bp.loc[valid].reset_index(drop=True)
    sw_all = np.where(delta_bp < 4.0, 0.3, 1.0)

    gc.collect()
    return features, y_all, sw_all, dates


def train_and_eval(features, y_all, sw_all, dates, use_feedback=False, label=""):
    W, S, V = 120, 63, 42
    H = 10
    n = len(features)
    all_preds, all_true, all_proba, all_dates = [], [], [], []
    fold_metrics = []
    fold = 0
    recent_fold_results = deque(maxlen=3)
    correction_indices = set()

    for start in range(W, n - S + 1, S):
        end = min(start + S, n)
        x_tr = features.iloc[start - W:start - V]
        y_tr = y_all.iloc[start - W:start - V]
        x_val = features.iloc[start - V:start]
        y_val = y_all.iloc[start - V:start]
        x_te = features.iloc[start:end]
        y_te = y_all.iloc[start:end]

        sw_tr = sw_all[start - W:start - V].copy()
        decay = np.exp(-2.0 * np.arange(len(sw_tr))[::-1] / max(len(sw_tr), 1))
        sw_tr = sw_tr * (decay / decay.max())

        # 方案5：错样本加权
        if use_feedback and correction_indices:
            tr_start = start - W
            for gi in correction_indices:
                local_i = gi - tr_start
                if 0 <= local_i < len(sw_tr):
                    sw_tr[local_i] *= 3.0

        model = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=8.0, reg_alpha=1.7, reg_lambda=2.0, gamma=0.1,
            random_state=42, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", early_stopping_rounds=50,
        )
        model.fit(x_tr, y_tr, sample_weight=sw_tr, eval_set=[(x_val, y_val)], verbose=False)
        proba = model.predict_proba(x_te)

        # 方案1+6：偏差检测与修正
        recent_acc = float("nan")
        recent_bias = 0.0
        bias_alert = False
        correction_applied = 0.0
        if use_feedback and len(recent_fold_results) >= 2:
            all_rp = np.concatenate([r["pred"] for r in recent_fold_results])
            all_rt = np.concatenate([r["true"] for r in recent_fold_results])
            recent_acc = float(np.mean(all_rp == all_rt))
            recent_bias = float(np.mean(all_rp == 1)) - float(np.mean(all_rt == 1))
            # 温和修正：只在偏差>15% 且 准确率低时才触发，修正幅度收紧到±0.05
            if abs(recent_bias) > 0.15:
                if recent_acc < 0.45:
                    coef = 0.25  # 准确率很低 → 稍激进
                elif recent_acc < 0.52:
                    coef = 0.15  # 准确率勉强 → 轻微修正
                else:
                    coef = 0.0   # 准确率正常 → 不修正
                correction = float(np.clip(-recent_bias * coef, -0.05, 0.05))
                if abs(correction) > 0.001:
                    proba[:, 1] = np.clip(proba[:, 1] + correction, 0.01, 0.99)
                    proba[:, 0] = 1.0 - proba[:, 1]
                    correction_applied = correction
                    bias_alert = True

        yp = (proba[:, 1] > 0.5).astype(int)
        try:
            fold_auc = roc_auc_score(y_te.values, proba[:, 1])
        except ValueError:
            fold_auc = float("nan")
        fold_metrics.append({
            "fold": fold + 1, "acc": accuracy_score(y_te.values, yp), "auc": fold_auc,
            "recent_acc": recent_acc, "recent_bias": round(recent_bias, 4),
            "bias_alert": int(bias_alert), "correction": round(correction_applied, 4),
        })

        all_preds.extend(yp.tolist())
        all_true.extend(y_te.values.tolist())
        all_proba.extend(proba[:, 1].tolist())
        all_dates.extend(dates[start:end])

        if use_feedback:
            recent_fold_results.append({"pred": yp.copy(), "true": y_te.values.copy()})
            wrong_mask = yp != y_te.values
            correction_indices = set([start + i for i in range(len(yp)) if wrong_mask[i]][-H:])

        fold += 1
        if fold % 5 == 0:
            print(f"  [{label}] fold {fold}...", flush=True)

    y_true = np.array(all_true)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_proba)
    years = pd.to_datetime(all_dates).year
    yearly = {}
    for yr in sorted(set(years)):
        m = years == yr
        if m.sum() > 5:
            yearly[yr] = {
                "acc": accuracy_score(y_true[m], y_pred[m]),
                "auc": roc_auc_score(y_true[m], y_proba[m]) if len(set(y_true[m])) > 1 else float("nan"),
                "n": int(m.sum()),
            }
    return {
        "acc": accuracy_score(y_true, y_pred), "mf1": f1_score(y_true, y_pred, average="macro"),
        "auc": roc_auc_score(y_true, y_proba), "fold_metrics": fold_metrics, "yearly": yearly,
        "pred_down": float(y_pred.mean()), "true_down": float(y_true.mean()),
        "n_alerts": sum(1 for fm in fold_metrics if fm.get("bias_alert", 0)),
    }


if __name__ == "__main__":
    features, y_all, sw_all, dates = load_data()
    print(f"  样本: {len(y_all)}, 特征: {features.shape[1]}, 下行占比: {y_all.mean():.3f}", flush=True)

    # 上次 baseline 结果（硬编码，避免跑两次 OOM）
    base = {
        "acc": 0.5260, "mf1": 0.5025, "auc": 0.4883,
        "pred_down": 0.655, "true_down": 0.562,
        "yearly": {
            2018: {"acc": 0.6423, "auc": 0.2618, "n": 123},
            2019: {"acc": 0.5492, "auc": 0.6059, "n": 244},
            2020: {"acc": 0.5226, "auc": 0.4032, "n": 243},
            2021: {"acc": 0.4362, "auc": 0.4976, "n": 243},
            2022: {"acc": 0.6198, "auc": 0.6537, "n": 242},
            2023: {"acc": 0.3636, "auc": 0.3504, "n": 242},
            2024: {"acc": 0.5992, "auc": 0.3933, "n": 242},
            2025: {"acc": 0.5432, "auc": 0.5229, "n": 243},
        },
    }

    print("\n=== Feedback (方案1+5+6, 温和修正) ===", flush=True)
    fb = train_and_eval(features, y_all, sw_all, dates, use_feedback=True, label="fb")
    print(f"  Acc={fb['acc']:.4f} AUC={fb['auc']:.4f}", flush=True)

    # 对比报告
    lines = [
        "=" * 60,
        "A/B 对比：反馈式优化（方案1+5+6）",
        "=" * 60, "",
        f"{'指标':<20} {'Baseline':>12} {'Feedback':>12} {'提升':>10}",
        f"{'-'*56}",
        f"{'Accuracy':<20} {base['acc']:>12.4f} {fb['acc']:>12.4f} {fb['acc']-base['acc']:>+10.4f}",
        f"{'Macro F1':<20} {base['mf1']:>12.4f} {fb['mf1']:>12.4f} {fb['mf1']-base['mf1']:>+10.4f}",
        f"{'AUC':<20} {base['auc']:>12.4f} {fb['auc']:>12.4f} {fb['auc']-base['auc']:>+10.4f}",
        f"{'pred_down':<20} {base['pred_down']:>12.3f} {fb['pred_down']:>12.3f}",
        f"{'true_down':<20} {base['true_down']:>12.3f} {fb['true_down']:>12.3f}",
        "",
        "=== 分年对比 ===",
        f"{'年份':<8} {'n':>6} {'B.ACC':>8} {'F.ACC':>8} {'B.AUC':>8} {'F.AUC':>8}",
        f"{'-'*48}",
    ]
    for yr in sorted(set(list(base["yearly"].keys()) + list(fb["yearly"].keys()))):
        b = base["yearly"].get(yr, {"acc": float("nan"), "auc": float("nan"), "n": 0})
        f = fb["yearly"].get(yr, {"acc": float("nan"), "auc": float("nan"), "n": 0})
        lines.append(f"{yr:<8} {b['n']:>6} {b['acc']:>8.4f} {f['acc']:>8.4f} {b['auc']:>8.4f} {f['auc']:>8.4f}")
    lines.append("")
    lines.append(f"方案6偏差修正触发: {fb['n_alerts']}/{len(fb['fold_metrics'])} folds")
    alerts = [fm for fm in fb["fold_metrics"] if fm.get("bias_alert")]
    if alerts:
        lines.append(f"{'Fold':<8}{'ACC':>8}{'r_ACC':>8}{'r_BIAS':>8}{'CORR':>8}")
        for fm in alerts:
            lines.append(f"{fm['fold']:<8}{fm['acc']:>8.3f}{fm['recent_acc']:>8.3f}{fm['recent_bias']:>8.3f}{fm['correction']:>8.3f}")

    report = "\n".join(lines)
    print(report, flush=True)
    (DATA / "ab_compare_feedback.txt").write_text(report, encoding="utf-8")
