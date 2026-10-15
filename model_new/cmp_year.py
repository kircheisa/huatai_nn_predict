import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

df = pd.read_csv(r"model_new\data\xgb_test_predictions_bin.csv")
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year

print("step=21 (当前文件)")
for y, g in df.groupby("year"):
    if y >= 2024:
        acc = accuracy_score(g["y_true"], g["y_pred"])
        auc = roc_auc_score(g["y_true"], g["prob_1_下行"])
        print(f"  {y}: n={len(g)}, Acc={acc:.4f}, AUC={auc:.4f}")
