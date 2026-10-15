# -*- coding: utf-8 -*-
"""
轻量调参 V5 —— 加载已保存的特征矩阵，纯调 XGBoost 超参
W=180, step=63, val=63 固定
"""
import os, sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.chdir(Path(__file__).resolve().parent)
BASE = Path(".").resolve()

# 直接加载已保存的特征和标签
print("[加载] 读取已保存的特征矩阵...", flush=True)
features = pd.read_csv("data/43_xgb_train_features_bin.csv", index_col=0)
y_all = pd.read_csv("data/43_xgb_labels_bin.csv")["label"]
print(f"  样本: {len(features)}, 特征: {features.shape[1]}", flush=True)

W, S, V = 180, 63, 63
n = len(features)
LOG = open("_tuning_results.txt", "w", encoding="utf-8")

def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()

def run(depth, lr, mcw, ra):
    preds, probas, trues = [], [], []
    fold = 0
    for start in range(W, n - S + 1, S):
        end = min(start + S, n)
        x_tr = features.iloc[start-W:start-V]
        y_tr = y_all.iloc[start-W:start-V]
        x_v = features.iloc[start-V:start]
        y_v = y_all.iloc[start-V:start]
        x_t = features.iloc[start:end]
        y_t = y_all.iloc[start:end]
        if len(x_tr) < 30: continue
        clf = XGBClassifier(n_estimators=600, max_depth=depth, learning_rate=lr,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=mcw,
                            reg_alpha=ra, reg_lambda=2.0, gamma=0.1, random_state=42,
                            objective="binary:logistic", eval_metric="logloss",
                            tree_method="hist", early_stopping_rounds=50)
        clf.fit(x_tr, y_tr, eval_set=[(x_v, y_v)], verbose=False)
        preds.extend(clf.predict(x_t).tolist())
        probas.extend(clf.predict_proba(x_t).tolist())
        trues.extend(y_t.values.tolist())
        fold += 1
    yp, yt = np.array(preds), np.array(trues)
    ypr = np.array(probas)
    return (accuracy_score(yt, yp), f1_score(yt, yp, average="macro"),
            roc_auc_score(yt, ypr[:, 1]), fold)

t0 = time.time()
best = {"depth": 3, "lr": 0.05, "mcw": 8.0, "ra": 0.5}
all_r = []

def test(label, **kw):
    p = {**best, **kw}
    acc, f1, auc, fold = run(**p)
    all_r.append({"label": label, "acc": acc, "f1": f1, "auc": auc})
    log(f"  {label:<25} Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  folds={fold}")
    return auc

# 阶段1: depth
log("\n=== [阶段1] max_depth [2,3,4] ===")
for d in [2, 3, 4]:
    test(f"depth={d}", depth=d)
bd = max([r for r in all_r if "depth=" in r["label"]], key=lambda x: x["auc"])
best["depth"] = int(bd["label"].split("=")[1])
log(f"  >>> depth={best['depth']} AUC={bd['auc']:.4f}")

# 阶段2: lr
log(f"\n=== [阶段2] lr [0.03,0.05,0.1] (depth={best['depth']}) ===")
for lr in [0.03, 0.05, 0.1]:
    test(f"lr={lr}", lr=lr)
bl = max([r for r in all_r if r["label"].startswith("lr=")], key=lambda x: x["auc"])
best["lr"] = float(bl["label"].split("=")[1])
log(f"  >>> lr={best['lr']} AUC={bl['auc']:.4f}")

# 阶段3: mcw
log(f"\n=== [阶段3] mcw [3,5,8] (lr={best['lr']}) ===")
for mcw in [3, 5, 8]:
    test(f"mcw={mcw}", mcw=float(mcw))
bm = max([r for r in all_r if "mcw=" in r["label"]], key=lambda x: x["auc"])
best["mcw"] = float(bm["label"].split("=")[1])
log(f"  >>> mcw={best['mcw']} AUC={bm['auc']:.4f}")

# 阶段4: ra
log(f"\n=== [阶段4] reg_alpha [0.1,0.5,1.0] (mcw={best['mcw']}) ===")
for ra in [0.1, 0.5, 1.0]:
    test(f"ra={ra}", ra=ra)
br = max([r for r in all_r if "ra=" in r["label"]], key=lambda x: x["auc"])
best["ra"] = float(br["label"].split("=")[1])
log(f"  >>> ra={best['ra']} AUC={br['auc']:.4f}")

# 汇总
log(f"\n{'='*60}")
log(f"最优组合: depth={best['depth']}, lr={best['lr']}, mcw={best['mcw']}, ra={best['ra']}")
log("\nTop 5:")
for i, r in enumerate(sorted(all_r, key=lambda x: -x["auc"])[:5]):
    log(f"  {i+1}. {r['label']:<20} AUC={r['auc']:.4f}  F1={r['f1']:.4f}  Acc={r['acc']:.4f}")
log(f"\n耗时: {(time.time()-t0)/60:.1f} min")
log("DONE")
LOG.close()
