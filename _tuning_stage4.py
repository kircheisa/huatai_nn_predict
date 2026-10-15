# -*- coding: utf-8 -*-
"""只跑阶段4: reg_alpha (基于已确定 lr=0.1, mcw=8, depth=2/3)"""
import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')
warnings.filterwarnings("ignore")
os.chdir(Path(__file__).resolve().parent)

features = pd.read_csv("data/43_xgb_train_features_bin.csv", index_col=0)
y_all = pd.read_csv("data/43_xgb_labels_bin.csv")["label"]
W, S, V = 180, 63, 63
n = len(features)

def run(ra):
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
        clf = XGBClassifier(n_estimators=600, max_depth=3, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=8.0,
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
    return accuracy_score(yt, yp), f1_score(yt, yp, average="macro"), roc_auc_score(yt, ypr[:, 1])

with open("_tuning_stage4.txt", "w", encoding="utf-8") as f:
    f.write("阶段4: reg_alpha (depth=3, lr=0.1, mcw=8)\n")
    for ra in [0.1, 0.5, 1.0]:
        acc, f1, auc = run(ra)
        f.write(f"ra={ra:<10} Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}\n")
    f.write("DONE\n")
