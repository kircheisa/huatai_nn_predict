# 快速网格：用最后120天训练最终模型，打印各正则化参数下 importance 分布
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

feat = pd.read_csv(r"model_new\data\xgb_train_features_bin.csv", index_col=0)
y_all = pd.read_csv(r"model_new\data\xgb_labels_bin.csv")["label"]
n = len(feat)
W = 120; V = 30
x_tr, y_tr = feat.iloc[n-W:n-V], y_all.iloc[n-W:n-V]
x_val, y_val = feat.iloc[n-V:], y_all.iloc[n-V:]
print(f"train={len(x_tr)} val={len(x_val)}")
print(f"{'ra':>5} {'mcw':>4} {'best_iter':>10} {'nonzero':>8} {'imp_max':>8}")
print("-" * 45)

for ra in [1.0, 1.5, 2.0]:
    for mcw in [8, 12]:
        m = XGBClassifier(
            n_estimators=600, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=mcw, reg_alpha=ra, reg_lambda=2.0, gamma=0.1,
            random_state=42, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", early_stopping_rounds=50,
        )
        m.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)
        imp = m.feature_importances_
        nz = int((imp > 0).sum())
        bi = getattr(m, "best_iteration", -1)
        top3 = ", ".join(f"{feat.columns[i]}({imp[i]:.3f})" for i in np.argsort(-imp)[:3])
        print(f"  {ra:.1f} {mcw:4d} {bi:10d} {nz:8d} {imp.max():8.4f}  top3:{top3}")
