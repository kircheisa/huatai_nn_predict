import os, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

os.chdir(Path(__file__).resolve().parent)
F = open("_one_test.txt", "w", encoding="utf-8")
try:
    features = pd.read_csv("data/43_xgb_train_features_bin.csv", index_col=0)
    y_all = pd.read_csv("data/43_xgb_labels_bin.csv")["label"]
    F.write(f"loaded {len(features)} rows\n"); F.flush()

    W, S, V = 180, 63, 63
    n = len(features)
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
        if len(x_tr) < 30:
            continue
        clf = XGBClassifier(n_estimators=600, max_depth=3, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=8.0,
                            reg_alpha=0.5, reg_lambda=2.0, gamma=0.1, random_state=42,
                            objective="binary:logistic", eval_metric="logloss",
                            tree_method="hist", early_stopping_rounds=50)
        clf.fit(x_tr, y_tr, eval_set=[(x_v, y_v)], verbose=False)
        preds.extend(clf.predict(x_t).tolist())
        probas.extend(clf.predict_proba(x_t).tolist())
        trues.extend(y_t.values.tolist())
        fold += 1
        F.write(f"fold {fold} done\n"); F.flush()

    yp, yt = np.array(preds), np.array(trues)
    ypr = np.array(probas)
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, average="macro")
    auc = roc_auc_score(yt, ypr[:, 1])
    F.write(f"\nRESULT: Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  folds={fold}\n")
    F.write("DONE\n")
except Exception as e:
    F.write(f"ERROR: {e}\n")
    F.write(traceback.format_exc())
finally:
    F.close()
