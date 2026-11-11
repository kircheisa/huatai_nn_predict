# -*- coding: utf-8 -*-
"""43v4 新族前向 AUC 衰减泄漏诊断（对照 8.2 节协议）
因果特征对未来 k 日方向的 AUC 应在 0.5 附近平坦或温和衰减；
若长周期(k=63/126) AUC>0.6 则疑似前视。对照: y__mom20(已知因果)。"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = Path(r"F:\43_华泰证券预处理\43_华泰证券预处理\data_new")
f = pd.read_csv(D / "43v4_features.csv", index_col=0, parse_dates=True)
old = set(pd.read_csv(D / "43v3_features.csv", index_col=0, nrows=2).columns)
new_cols = [c for c in f.columns if c not in old]

y = pd.read_csv(D / "43v3_labels_bin.csv", index_col=0, parse_dates=True)["rate_delta_5d"]
ks = [1, 5, 10, 21, 63, 126]
fwd = {k: (y.shift(-k) - y) for k in ks}   # 未来k日下行幅度(负=下行)

rows = []
sel = new_cols + ["y__mom20"]
for c in sel:
    s = f[c]
    r = {"feature": c}
    for k in ks:
        d = fwd[k]
        m = s.notna() & d.notna()
        if m.sum() < 200:
            r[f"k{k}"] = np.nan
            continue
        try:
            # 下行=1: 用 |z| 方向不确定，统一用 -s 也算一遍取与k=1一致的符号约定:
            # 报告 max(auc, 1-auc) 作为"区分度"（方向无关），泄漏判定看长k不衰减
            auc = roc_auc_score((d < 0).astype(int)[m], s[m])
            r[f"k{k}"] = round(max(auc, 1 - auc), 3)
        except ValueError:
            r[f"k{k}"] = np.nan
    rows.append(r)

res = pd.DataFrame(rows)
print(res.to_string(index=False))
bad = res[(res["k63"] > 0.60) | (res["k126"] > 0.60)]
print("\n疑似泄漏(k63或k126>0.60):", bad["feature"].tolist() if len(bad) else "无")
res.to_csv(D / "43v4_leak_check.csv", index=False, encoding="utf-8-sig")
