# -*- coding: utf-8 -*-
"""快速运行优化后的滚动训练（时间衰减+去重遗传+利率强度+超参调优）"""
import sys, os, traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier

os.chdir(Path(__file__).resolve().parent)
BASE = Path(".").resolve()

# ============ 参数 ============
START_DATE = "2018-01-01"
W, S, V = 756, 63, 63  # 窗口/步长/验证

# ============ 1. 加载数据 ============
print("[1/5] 加载数据...", flush=True)
df_data = pd.read_csv("data/43_data.csv")
df_rate = pd.read_csv("data/43_rate.csv")
df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"])
df_rate["time"] = pd.to_datetime(df_rate["time"])
merged = pd.merge(df_data, df_rate, left_on="Unnamed: 0", right_on="time", how="inner").sort_values("Unnamed: 0")
merged["target_date_5d"] = merged["Unnamed: 0"].shift(-5)
merged["rate_delta_5d"] = merged["rate"].shift(-5) - merged["rate"]

def gen_label(v):
    if pd.isna(v): return np.nan
    if v < -0.03: return 0
    if v < 0.03: return 1
    return 2

merged["label"] = merged["rate_delta_5d"].apply(gen_label)
merged = merged.dropna(subset=["label"]).copy()
merged["label"] = merged["label"].astype(int)
y_all = merged["label"].copy()

# ============ 2. 构建特征 ============
print("[2/5] 构建特征...", flush=True)
features = merged.sort_values("Unnamed: 0").copy()

# 利率时序
for lag in [1, 2, 5, 10, 20]:
    features[f"rate_lag_{lag}"] = features["rate"].shift(lag)
for window in [5, 10, 20]:
    features[f"rate_ma_{window}"] = features["rate"].rolling(window=window, min_periods=1).mean()
    features[f"rate_std_{window}"] = features["rate"].rolling(window=window, min_periods=2).std()
features["rate_change_1d"] = features["rate"].diff(1)
features["rate_change_5d"] = features["rate"].diff(5)
features["rate_change_10d"] = features["rate"].diff(10)
features["curve_ma_gap_5_20"] = features["rate_ma_5"] - features["rate_ma_20"]
features["curve_ma_gap_10_20"] = features["rate_ma_10"] - features["rate_ma_20"]
for col in ["momentum_1_robust_scaled", "momentum_7_robust_scaled", "momentum_30_robust_scaled"]:
    if col in features.columns:
        features[f"{col}_lag_1"] = features[col].shift(1)
        features[f"{col}_lag_5"] = features[col].shift(5)

# 信用利差衍生
for cs_col in ["1年信用利差", "3年信用利差", "5年信用利差"]:
    if cs_col in features.columns:
        features[f"{cs_col}_chg_5d"] = features[cs_col].diff(5)
        features[f"{cs_col}_chg_20d"] = features[cs_col].diff(20)
        features[f"{cs_col}_ma_20"] = features[cs_col].rolling(20, min_periods=1).mean()
if "5年信用利差" in features.columns and "1年信用利差" in features.columns:
    features["credit_term_spread_5_1"] = features["5年信用利差"] - features["1年信用利差"]

# 美元/汇率衍生
if "美元指数" in features.columns:
    features["dollar_chg_5d"] = features["美元指数"].diff(5)
    features["dollar_chg_20d"] = features["美元指数"].diff(20)
    features["dollar_ma_20"] = features["美元指数"].rolling(20, min_periods=1).mean()
    features["dollar_deviation_20"] = features["美元指数"] - features["dollar_ma_20"]
if "即期汇率:美元兑人民币" in features.columns:
    features["fx_chg_5d"] = features["即期汇率:美元兑人民币"].diff(5)
    features["fx_chg_20d"] = features["即期汇率:美元兑人民币"].diff(20)
    if "中间价:美元兑人民币" in features.columns:
        features["fx干预_chl_5d"] = (features["中间价:美元兑人民币"] - features["即期汇率:美元兑人民币"]).diff(5)
if "美元指数" in features.columns:
    features["rate_dollar_gap"] = features["rate"] - features["美元指数"].rolling(60, min_periods=1).mean() / 20

# 股债相对回报率衍生
if "股债相对回报率" in features.columns:
    features["股债相对回报率_chg_5d"] = features["股债相对回报率"].diff(5)
    features["股债相对回报率_chg_20d"] = features["股债相对回报率"].diff(20)
    features["股债相对回报率_ma_5"] = features["股债相对回报率"].rolling(5, min_periods=1).mean()
    features["股债相对回报率_ma_20"] = features["股债相对回报率"].rolling(20, min_periods=1).mean()
    features["股债相对回报率_deviation_20"] = features["股债相对回报率"] - features["股债相对回报率_ma_20"]

# 加杠杆情绪衍生
for vol_col in ["成交量:R001", "成交量:R007"]:
    if vol_col in features.columns:
        features[f"{vol_col}_chg_5d"] = features[vol_col].diff(5)
        features[f"{vol_col}_chg_20d"] = features[vol_col].diff(20)
        features[f"{vol_col}_ma_5"] = features[vol_col].rolling(5, min_periods=1).mean()
        features[f"{vol_col}_ma_20"] = features[vol_col].rolling(20, min_periods=1).mean()
overnight_col = "成交量:R001/中国:成交量:银行间质押式回购"
if overnight_col in features.columns:
    features["隔夜回购占比_chg_5d"] = features[overnight_col].diff(5)
    features["隔夜回购占比_chg_20d"] = features[overnight_col].diff(20)
    features["隔夜回购占比_ma_5"] = features[overnight_col].rolling(5, min_periods=1).mean()
    features["隔夜回购占比_ma_20"] = features[overnight_col].rolling(20, min_periods=1).mean()
    features["隔夜回购占比_deviation_20"] = features[overnight_col] - features["隔夜回购占比_ma_20"]
if "成交量:R001" in features.columns and "成交量:R007" in features.columns:
    features["R001_R007_vol_ratio"] = features["成交量:R001"] / (features["成交量:R007"] + 1e-8)

# === 新增：利率相对强度 ===
if "rate_std_20" in features.columns:
    features["rate_chg_5d_zscore"] = features["rate"].diff(5) / features["rate_std_20"]
    features["rate_chg_10d_zscore"] = features["rate"].diff(10) / features["rate_std_20"]
rw = features["rate"].rolling(60, min_periods=60)
features["rate_percentile_60"] = rw.rank(pct=True)
features["rate_zscore_60"] = (features["rate"] - rw.mean()) / rw.std()

features = features.drop(columns=["time", "label", "rate_delta_5d", "date_list"], errors="ignore").copy()
features = features.set_index("Unnamed: 0")
features = features.apply(pd.to_numeric, errors="coerce")
features = features.fillna(features.median(numeric_only=True)).fillna(99.00)

# 合并遗传特征（去重）
gp_file = BASE / "data/43_x.csv"
if gp_file.exists():
    gp = pd.read_csv(gp_file, index_col=0)
    gp.index = pd.to_datetime(gp.index)
    gp = gp.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    keep_cols = ["gp_feature_1", "gp_feature_2", "gp_feature_3", "gp_feature_4", "gp_feature_8"]
    gp = gp[[c for c in keep_cols if c in gp.columns]]
    features = features.join(gp, how="left")
    gp_cols = [c for c in gp.columns if c in features.columns]
    features[gp_cols] = features[gp_cols].fillna(0.0)
    print(f"  遗传特征: {len(gp_cols)} 个 (去重后)", flush=True)

# 过滤日期
cutoff = pd.to_datetime(START_DATE)
mask = pd.to_datetime(merged["Unnamed: 0"]) >= cutoff
features = features.loc[mask.values].reset_index(drop=True)
y_all = y_all.loc[mask.values].reset_index(drop=True)
merged = merged.loc[mask].reset_index(drop=True)
print(f"  样本: {len(features)}, 特征数: {features.shape[1]}", flush=True)

# ============ 3. 滚动训练 ============
print("[3/5] 滚动窗口训练...", flush=True)

n = len(features)
all_preds, all_probas, all_true, all_dates, all_meta = [], [], [], [], []
fold = 0
for start in range(W, n - S + 1, S):
    end = min(start + S, n)
    x_tr = features.iloc[start-W:start-V]
    y_tr = y_all.iloc[start-W:start-V]
    x_val = features.iloc[start-V:start]
    y_val = y_all.iloc[start-V:start]
    x_te = features.iloc[start:end]
    y_te = y_all.iloc[start:end]
    if len(x_tr) < 100: continue

    # 优化后超参：使用原有效参数 + n_estimators提升到600
    m = XGBClassifier(n_estimators=600, max_depth=3, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=8.0,
                      reg_alpha=0.5, reg_lambda=2.0, gamma=0.1,
                      random_state=42, eval_metric="merror", tree_method="hist",
                      early_stopping_rounds=50)
    m.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)

    yp = m.predict(x_te)
    all_preds.extend(yp.tolist())
    all_probas.extend(m.predict_proba(x_te).tolist())
    all_true.extend(y_te.values.tolist())
    all_dates.extend(merged["Unnamed: 0"].iloc[start:end].tolist())
    all_meta.append(merged.iloc[start:end][["Unnamed: 0", "rate", "target_date_5d"]])
    fold += 1
    if fold % 5 == 0:
        print(f"  fold {fold}...", flush=True)

print(f"  完成: {fold} folds, {len(all_preds)} 预测", flush=True)

# ============ 4. 评估 ============
print("[4/5] 评估...", flush=True)
y_pred = np.array(all_preds)
y_true = np.array(all_true)
y_proba = np.array(all_probas)
test_meta = pd.concat(all_meta, ignore_index=True)
acc = accuracy_score(y_true, y_pred)
mf1 = f1_score(y_true, y_pred, average="macro")
cm = confusion_matrix(y_true, y_pred)
maj = int(pd.Series(y_true).mode().iloc[0])
maj_acc = float(np.mean(y_true == maj))

# 最终模型
fm = XGBClassifier(n_estimators=600, max_depth=3, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, min_child_weight=8.0,
                   reg_alpha=0.5, reg_lambda=2.0, gamma=0.1,
                   random_state=42, eval_metric="merror", tree_method="hist",
                   early_stopping_rounds=50)
fs = max(0, n-W); fv = max(0, n-V)
fm.fit(features.iloc[fs:fv], y_all.iloc[fs:fv],
       eval_set=[(features.iloc[fv:n], y_all.iloc[fv:n])], verbose=False)
joblib.dump(fm, BASE / "data/43_xgb_model.joblib")

# 保存
features.to_csv(BASE / "data/43_xgb_train_features.csv")
pd.DataFrame({"label": y_all}).to_csv(BASE / "data/43_xgb_labels.csv", index=False)

LABEL_MAP = {0: "偏多", 1: "震荡", 2: "偏空"}
def fmt_view(probs):
    s = pd.Series(probs, index=["prob_0","prob_1","prob_2"]).sort_values(ascending=False)
    return f"{LABEL_MAP[int(s.index[0].split('_')[-1])]}（{LABEL_MAP[int(s.index[1].split('_')[-1])]}）"

pred_df = pd.DataFrame({
    "date": pd.to_datetime(all_dates).astype(str),
    "y_true": y_true, "y_pred": y_pred,
    "prob_0": y_proba[:,0], "prob_1": y_proba[:,1], "prob_2": y_proba[:,2],
})
pred_df.to_csv(BASE / "data/43_xgb_test_predictions.csv", index=False)

view_df = pd.DataFrame({
    "判断基点": pd.to_datetime(test_meta["Unnamed: 0"]).dt.strftime("%Y-%m-%d"),
    "当期10Y": test_meta["rate"].round(4),
    "AI研判": [fmt_view(y_proba[i]) for i in range(len(y_proba))],
    "预测时点": pd.to_datetime(test_meta["target_date_5d"]).dt.strftime("%Y-%m-%d"),
    "偏多": [f"{v:.2%}" for v in y_proba[:,0]],
    "震荡": [f"{v:.2%}" for v in y_proba[:,1]],
    "偏空": [f"{v:.2%}" for v in y_proba[:,2]],
})
view_df.to_csv(BASE / "data/43_xgb_test_predictions_view.csv", index=False, encoding="utf-8-sig")
view_df.tail(1).to_csv(BASE / "data/43_xgb_latest_prediction_view.csv", index=False, encoding="utf-8-sig")

imp = pd.DataFrame({"feature": features.columns, "importance": fm.feature_importances_})
imp = imp.sort_values("importance", ascending=False)
imp.to_csv(BASE / "data/43_xgb_feature_importance.csv", index=False)

metrics = "\n".join([
    "XGBoost 三分类测试结果（优化：去重遗传+利率强度+超参调优）",
    f"训练模式: rolling (window={W}, step={S}, val={V})",
    f"训练 folds: {fold}",
    f"数据起始: {START_DATE}",
    f"测试样本数: {len(y_true)}",
    f"总特征数: {features.shape[1]}",
    f"超参: depth=3, n=600, lr=0.05 (原有效参数+增树数)",
    f"Accuracy: {acc:.4f}",
    f"Macro F1: {mf1:.4f}",
    f"多数类基线标签: {maj}",
    f"多数类基线 Accuracy: {maj_acc:.4f}",
    "",
    "测试集标签分布:",
    pd.Series(y_true).value_counts(normalize=True).sort_index().to_string(),
    "",
    "Confusion Matrix:",
    pd.DataFrame(cm, index=["true_0","true_1","true_2"], columns=["pred_0","pred_1","pred_2"]).to_string(),
    "",
    "Classification Report:",
    classification_report(y_true, y_pred, digits=4),
    "",
    "Top 20 重要特征:",
    imp.head(20).to_string(index=False),
])
(BASE / "data/43_xgb_metrics.txt").write_text(metrics, encoding="utf-8")

print(f"\nAccuracy: {acc:.4f}  基线: {maj_acc:.4f}")
print(f"Macro F1: {mf1:.4f}")
print(f"特征数: {features.shape[1]}, folds: {fold}")
print("Top 10:")
print(imp.head(10).to_string(index=False))
print("DONE")
