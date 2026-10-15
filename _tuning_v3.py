# -*- coding: utf-8 -*-
"""
简化调参 —— 固定 W=180, step=63, 只调 XGBoost 超参
4个阶段：depth → lr → min_child_weight → reg_alpha
"""
import os, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.chdir(Path(__file__).resolve().parent)
BASE = Path(".").resolve()
LOG = BASE / "_tuning_log.txt"
LOG.write_text("", encoding="utf-8")

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def gen_label(v):
    if pd.isna(v): return np.nan
    return 1 if v < 0 else 0

# 导入特征工程（直接 from 43_3）
import importlib.util
spec = importlib.util.spec_from_file_location("m", str(BASE / "43_3_华泰证券债券预测_XGB训练_二分类.py"))
m = importlib.util.module_from_spec(spec)
import sys; sys.argv = ["x"]
spec.loader.exec_module(m)

# 加载数据
log("[准备] 加载数据...")
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

log("[准备] 构建特征...")
features = m.build_features(merged)
gp_file = BASE / "data/43_x.csv"
if gp_file.exists():
    gp = pd.read_csv(gp_file, index_col=0)
    gp.index = pd.to_datetime(gp.index)
    gp = gp.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    keep = ["gp_feature_1", "gp_feature_2", "gp_feature_3", "gp_feature_4", "gp_feature_8"]
    gp = gp[[c for c in keep if c in gp.columns]]
    features = features.join(gp, how="left")
    for c in gp.columns:
        if c in features.columns: features[c] = features[c].fillna(0.0)
cutoff = pd.to_datetime("2018-01-01")
mask = pd.to_datetime(merged["Unnamed: 0"]) >= cutoff
features = features.loc[mask.values].reset_index(drop=True)
y_all = y_all.loc[mask.values].reset_index(drop=True)
merged = merged.loc[mask].reset_index(drop=True)
log(f"  样本: {len(features)}, 特征: {features.shape[1]}")

W, S, V = 180, 63, 63
n = len(features)

def run(depth=3, lr=0.05, mcw=8.0, ra=0.5):
    preds, probas, trues = [], [], []
    fold = 0
    for start in range(W, n - S + 1, S):
        end = min(start + S, n)
        x_tr = features.iloc[start-W:start-V]
        y_tr = y_all.iloc[start-W:start-V]
        x_val = features.iloc[start-V:start]
        y_val = y_all.iloc[start-V:start]
        x_te = features.iloc[start:end]
        y_te = y_all.iloc[start:end]
        if len(x_tr) < 30: continue
        clf = XGBClassifier(
            n_estimators=600, max_depth=depth, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=mcw,
            reg_alpha=ra, reg_lambda=2.0, gamma=0.1, random_state=42,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", early_stopping_rounds=50,
        )
        clf.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)
        preds.extend(clf.predict(x_te).tolist())
        probas.extend(clf.predict_proba(x_te).tolist())
        trues.extend(y_te.values.tolist())
        fold += 1
    y_pred, y_true = np.array(preds), np.array(trues)
    y_proba = np.array(probas)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_proba[:, 1])
    base = float(np.mean(y_true == int(pd.Series(y_true).mode().iloc[0])))
    return acc, f1, auc, base, fold

t0 = time.time()
best = {"depth": 3, "lr": 0.05, "mcw": 8.0, "ra": 0.5}
all_r = []

def test(label, **kw):
    p = {**best, **kw}
    log(f"  测试 {label} ...")
    acc, f1, auc, base, fold = run(**p)
    all_r.append({"label": label, "acc": acc, "f1": f1, "auc": auc, "base": base, "fold": fold})
    log(f"  {label:<25} Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  folds={fold}")
    return auc

# 阶段1: depth
log("\n[阶段1] max_depth [2,3,4]")
for d in [2, 3, 4]:
    test(f"depth={d}", depth=d)
best_d = max([r for r in all_r if "depth=" in r["label"]], key=lambda x: x["auc"])
best["depth"] = int(best_d["label"].split("=")[1])
log(f"  >>> 最优 depth={best['depth']} (AUC={best_d['auc']:.4f})")

# 阶段2: lr
log(f"\n[阶段2] lr [0.03,0.05,0.1] (depth={best['depth']})")
for lr in [0.03, 0.05, 0.1]:
    test(f"lr={lr}", lr=lr)
best_lr = max([r for r in all_r if r["label"].startswith("lr=")], key=lambda x: x["auc"])
best["lr"] = float(best_lr["label"].split("=")[1])
log(f"  >>> 最优 lr={best['lr']} (AUC={best_lr['auc']:.4f})")

# 阶段3: min_child_weight
log(f"\n[阶段3] mcw [3,5,8] (lr={best['lr']})")
for mcw in [3, 5, 8]:
    test(f"mcw={mcw}", mcw=float(mcw))
best_mcw = max([r for r in all_r if "mcw=" in r["label"]], key=lambda x: x["auc"])
best["mcw"] = float(best_mcw["label"].split("=")[1])
log(f"  >>> 最优 mcw={best['mcw']} (AUC={best_mcw['auc']:.4f})")

# 阶段4: reg_alpha
log(f"\n[阶段4] reg_alpha [0.1,0.5,1.0] (mcw={best['mcw']})")
for ra in [0.1, 0.5, 1.0]:
    test(f"ra={ra}", ra=ra)
best_ra = max([r for r in all_r if "ra=" in r["label"]], key=lambda x: x["auc"])
best["ra"] = float(best_ra["label"].split("=")[1])
log(f"  >>> 最优 ra={best['ra']} (AUC={best_ra['auc']:.4f})")

# 汇总
log("\n" + "=" * 60)
log("最终最优组合:")
log(f"  depth={best['depth']}, lr={best['lr']}, mcw={best['mcw']}, ra={best['ra']}")
log(f"\n全局 Top 5:")
for i, r in enumerate(sorted(all_r, key=lambda x: -x["auc"])[:5]):
    log(f"  {i+1}. {r['label']:<20} AUC={r['auc']:.4f}  F1={r['f1']:.4f}  Acc={r['acc']:.4f}")
log(f"\n耗时: {(time.time()-t0)/60:.1f} min")
log("DONE")
