# -*- coding: utf-8 -*-
"""临时复算：滚动校准真实Acc（修正02脚本的accuracy_score误用bug）"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r"f:\43_华泰证券预处理\43_华泰证券预处理\model_nn")
df = pd.read_csv(BASE / "out" / "H10_S40_FINAL5_nn_test_predictions.csv",
                 parse_dates=["date"]).sort_values("date").reset_index(drop=True)

THS = np.round(np.arange(0.35, 0.6501, 0.01), 2)
LAG = 10
p = df["proba_down"].to_numpy()
y = df["y_true"].to_numpy()
yr = df["date"].dt.year.to_numpy()

correct = ((p[None, :] >= THS[:, None]).astype(int) == y[None, :]).astype(int)
cum = np.zeros((len(THS), len(df) + 1), dtype=int)
cum[:, 1:] = np.cumsum(correct, axis=1)

for W in (120, 250):
    idx, ths, preds = [], [], []
    for i in range(len(df)):
        hi, lo = max(0, i - LAG), max(0, i - LAG - W)
        if hi - lo < 30:
            continue
        accs = (cum[:, hi] - cum[:, lo]) / (hi - lo)
        k = int(np.argmax(accs))
        idx.append(i); ths.append(THS[k]); preds.append(int(p[i] >= THS[k]))
    idx = np.array(idx); ths = np.array(ths); preds = np.array(preds)
    yt, yy = y[idx], yr[idx]
    ok = preds == yt
    print(f"\n===== W={W} 真实滚动校准 =====")
    print(f"全局 Acc={ok.mean():.4f} (n={len(idx)})  pred1={preds.mean():.3f}")
    for y_ in np.unique(yy):
        m = yy == y_
        print(f"  {y_}: acc={ok[m].mean():.4f} (n={m.sum()})")
    m25 = yy == 2025
    # 2025分月
    mo = pd.to_datetime(df['date'].iloc[idx]).dt.month.to_numpy()
    for mo_ in range(1, 13):
        mm = m25 & (mo == mo_)
        if mm.sum():
            print(f"    2025-{mo_:02d}: acc={ok[mm].mean():.3f} (n={mm.sum()}, th均值={ths[mm].mean():.2f})")
