# -*- coding: utf-8 -*-
"""逐fold分析滚动训练结果"""
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/43_xgb_test_predictions.csv")
df["date"] = pd.to_datetime(df["date"])

fold_size = 63
total_folds = 14
fold14 = df.iloc[-fold_size:]

print("=== Fold 14 (最后一个季度) ===")
d0 = fold14["date"].iloc[0].strftime("%Y-%m-%d")
d1 = fold14["date"].iloc[-1].strftime("%Y-%m-%d")
print(f"日期范围: {d0} ~ {d1}")
print(f"样本数: {len(fold14)}")

y_true = fold14["y_true"]
y_pred = fold14["y_pred"]
acc = (y_true == y_pred).mean()
maj = int(y_true.mode().iloc[0])
maj_acc = (y_true == maj).mean()

print(f"Accuracy: {acc:.4f}")
print(f"多数类基线: {maj_acc:.4f} (label={maj})")
print()

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print("=== 混淆矩阵 ===")
print(pd.DataFrame(cm, index=["true_0偏多", "true_1震荡", "true_2偏空"], columns=["pred_0", "pred_1", "pred_2"]))
print()
print("=== 分类报告 ===")
print(classification_report(y_true, y_pred, digits=4, target_names=["0偏多", "1震荡", "2偏空"]))

print()
print("=== 所有14个fold逐fold对比 ===")
for i in range(total_folds):
    start = i * fold_size
    end = start + fold_size
    fold = df.iloc[start:end]
    fa = (fold["y_true"] == fold["y_pred"]).mean()
    fmaj = (fold["y_true"] == fold["y_true"].mode().iloc[0]).mean()
    fd0 = fold["date"].iloc[0].strftime("%Y-%m-%d")
    fd1 = fold["date"].iloc[-1].strftime("%Y-%m-%d")
    flag = "OK" if fa > fmaj else "--"
    print(f"fold {i+1:2d}: {fd0} ~ {fd1}  acc={fa:.4f}  基线={fmaj:.4f}  {flag}")
