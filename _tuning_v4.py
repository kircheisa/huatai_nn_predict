# -*- coding: utf-8 -*-
"""
调参 V4 —— 完全独立，内联特征，无 importlib
去掉慢速 Hurst/FFT，保留核心特征
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
LOG = open(BASE / "_tuning_log.txt", "w", encoding="utf-8")

def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()

def gen_label(v):
    if pd.isna(v): return np.nan
    return 1 if v < 0 else 0

# ====== 加载数据 ======
log("[1] 加载数据...")
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

# ====== 构建特征 ======
log("[2] 构建特征...")
f = merged.sort_values("Unnamed: 0").copy()
rate = f["rate"]

for lag in [1, 2, 5, 10, 20]:
    f[f"rate_lag_{lag}"] = rate.shift(lag)
for w in [5, 10, 20]:
    f[f"rate_ma_{w}"] = rate.rolling(w, min_periods=1).mean()
    f[f"rate_std_{w}"] = rate.rolling(w, min_periods=2).std()
f["rate_change_1d"] = rate.diff(1)
f["rate_change_5d"] = rate.diff(5)
f["rate_change_10d"] = rate.diff(10)
f["curve_ma_gap_5_20"] = f["rate_ma_5"] - f["rate_ma_20"]
f["curve_ma_gap_10_20"] = f["rate_ma_10"] - f["rate_ma_20"]
for col in ["momentum_1_robust_scaled", "momentum_7_robust_scaled", "momentum_30_robust_scaled"]:
    if col in f.columns:
        f[f"{col}_lag_1"] = f[col].shift(1)
        f[f"{col}_lag_5"] = f[col].shift(5)

for cs_col in ["1年信用利差", "3年信用利差", "5年信用利差"]:
    if cs_col in f.columns:
        f[f"{cs_col}_chg_5d"] = f[cs_col].diff(5)
        f[f"{cs_col}_chg_20d"] = f[cs_col].diff(20)
        f[f"{cs_col}_ma_20"] = f[cs_col].rolling(20, min_periods=1).mean()
if "5年信用利差" in f.columns and "1年信用利差" in f.columns:
    f["credit_term_spread_5_1"] = f["5年信用利差"] - f["1年信用利差"]

if "美元指数" in f.columns:
    f["dollar_chg_5d"] = f["美元指数"].diff(5)
    f["dollar_chg_20d"] = f["美元指数"].diff(20)
    f["dollar_ma_20"] = f["美元指数"].rolling(20, min_periods=1).mean()
    f["dollar_deviation_20"] = f["美元指数"] - f["dollar_ma_20"]
if "即期汇率:美元兑人民币" in f.columns:
    f["fx_chg_5d"] = f["即期汇率:美元兑人民币"].diff(5)
    f["fx_chg_20d"] = f["即期汇率:美元兑人民币"].diff(20)
    if "中间价:美元兑人民币" in f.columns:
        f["fx干预_chl_5d"] = (f["中间价:美元兑人民币"] - f["即期汇率:美元兑人民币"]).diff(5)

if "股债相对回报率" in f.columns:
    f["股债相对回报率_chg_5d"] = f["股债相对回报率"].diff(5)
    f["股债相对回报率_ma_5"] = f["股债相对回报率"].rolling(5, min_periods=1).mean()
    f["股债相对回报率_ma_20"] = f["股债相对回报率"].rolling(20, min_periods=1).mean()

for vc in ["成交量:R001", "成交量:R007"]:
    if vc in f.columns:
        f[f"{vc}_chg_5d"] = f[vc].diff(5)
        f[f"{vc}_ma_20"] = f[vc].rolling(20, min_periods=1).mean()
oc = "成交量:R001/中国:成交量:银行间质押式回购"
if oc in f.columns:
    f["隔夜回购占比_chg_5d"] = f[oc].diff(5)
    f["隔夜回购占比_ma_20"] = f[oc].rolling(20, min_periods=1).mean()

if "rate_std_20" in f.columns:
    f["rate_chg_5d_zscore"] = rate.diff(5) / f["rate_std_20"]
rw = rate.rolling(60, min_periods=60)
f["rate_percentile_60"] = rw.rank(pct=True)
f["rate_zscore_60"] = (rate - rw.mean()) / rw.std()

# 技术面（精简版，无 Hurst/FFT）
delta = rate.diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
for p in [6, 14, 24]:
    ag = gain.rolling(p, min_periods=p).mean()
    al = loss.rolling(p, min_periods=p).mean()
    f[f"rsi_{p}"] = 100 - 100 / (1 + ag / (al + 1e-10))
bb_m = rate.rolling(20, min_periods=20).mean()
bb_s = rate.rolling(20, min_periods=20).std()
f["bb_pctb_20"] = (rate - (bb_m - 2*bb_s)) / (4*bb_s + 1e-10)
for p in [5, 10, 20]:
    sh = rate.shift(p)
    f[f"roc_{p}"] = (rate - sh) / (sh.abs() + 1e-10)
for p in [10, 20, 60]:
    f[f"ar1_{p}"] = rate.rolling(p).corr(rate.shift(1))
ema12 = rate.ewm(span=12, adjust=False).mean()
ema26 = rate.ewm(span=26, adjust=False).mean()
f["macd_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

# 跨资产背离
rate_mom = rate.pct_change(20)
rate_mom_s = rate_mom.rolling(60, min_periods=20).std() + 1e-10
for rc, sn in [("期货结算价(活跃合约):螺纹钢:前一个观测值", "rebar"),
                ("中国:水泥价格指数:前一个观测值", "cement")]:
    if rc in f.columns:
        cm = f[rc].pct_change(20)
        cs = cm.rolling(60, min_periods=20).std() + 1e-10
        f[f"divergence_{sn}"] = (cm / cs) - (rate_mom / rate_mom_s)
        f[f"corr_rate_{sn}_20"] = rate.rolling(20).corr(f[rc])

# 波动率
dc = rate.diff().abs()
for sw in [5, 10, 20]:
    rv = dc.rolling(sw, min_periods=sw).std()
    f[f"vol_pct_{sw}"] = rv.rolling(120, min_periods=60).rank(pct=True)

# 清理
f = f.drop(columns=["time", "label", "rate_delta_5d", "date_list"], errors="ignore").copy()
f = f.set_index("Unnamed: 0")
f = f.apply(pd.to_numeric, errors="coerce")
f = f.replace([np.inf, -np.inf], np.nan)
f = f.fillna(f.median(numeric_only=True)).fillna(99.0)

# 遗传特征
gp_file = BASE / "data/43_x.csv"
if gp_file.exists():
    gp = pd.read_csv(gp_file, index_col=0)
    gp.index = pd.to_datetime(gp.index)
    gp = gp.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    keep = ["gp_feature_1", "gp_feature_2", "gp_feature_3", "gp_feature_4", "gp_feature_8"]
    gp = gp[[c for c in keep if c in gp.columns]]
    f = f.join(gp, how="left")
    for c in gp.columns:
        if c in f.columns: f[c] = f[c].fillna(0.0)

cutoff = pd.to_datetime("2018-01-01")
mask = pd.to_datetime(merged["Unnamed: 0"]) >= cutoff
f = f.loc[mask.values].reset_index(drop=True)
y_all = y_all.loc[mask.values].reset_index(drop=True)
merged = merged.loc[mask].reset_index(drop=True)
log(f"  样本: {len(f)}, 特征: {f.shape[1]}")

# ====== 调参 ======
W, S, V = 180, 63, 63
n = len(f)

def run(depth, lr, mcw, ra):
    preds, probas, trues = [], [], []
    fold = 0
    for start in range(W, n - S + 1, S):
        end = min(start + S, n)
        x_tr = f.iloc[start-W:start-V]
        y_tr = y_all.iloc[start-W:start-V]
        x_v = f.iloc[start-V:start]
        y_v = y_all.iloc[start-V:start]
        x_t = f.iloc[start:end]
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
        if fold % 5 == 0:
            log(f"    fold {fold}...")
    yp, yt = np.array(preds), np.array(trues)
    ypr = np.array(probas)
    return (accuracy_score(yt, yp), f1_score(yt, yp, average="macro"),
            roc_auc_score(yt, ypr[:, 1]),
            float(np.mean(yt == int(pd.Series(yt).mode().iloc[0]))), fold)

t0 = time.time()
best = {"depth": 3, "lr": 0.05, "mcw": 8.0, "ra": 0.5}
all_r = []

def test(label, **kw):
    p = {**best, **kw}
    log(f"  run {label}...")
    acc, f1, auc, base, fold = run(**p)
    all_r.append({"label": label, "acc": acc, "f1": f1, "auc": auc})
    log(f"  {label:<25} Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return auc

# 阶段1: depth
log("\n=== [阶段1] max_depth ===")
for d in [2, 3, 4]:
    test(f"depth={d}", depth=d)
bd = max([r for r in all_r if "depth=" in r["label"]], key=lambda x: x["auc"])
best["depth"] = int(bd["label"].split("=")[1])
log(f"  >>> depth={best['depth']} AUC={bd['auc']:.4f}")

# 阶段2: lr
log(f"\n=== [阶段2] lr (depth={best['depth']}) ===")
for lr in [0.03, 0.05, 0.1]:
    test(f"lr={lr}", lr=lr)
bl = max([r for r in all_r if r["label"].startswith("lr=")], key=lambda x: x["auc"])
best["lr"] = float(bl["label"].split("=")[1])
log(f"  >>> lr={best['lr']} AUC={bl['auc']:.4f}")

# 阶段3: mcw
log(f"\n=== [阶段3] mcw (lr={best['lr']}) ===")
for mcw in [3, 5, 8]:
    test(f"mcw={mcw}", mcw=float(mcw))
bm = max([r for r in all_r if "mcw=" in r["label"]], key=lambda x: x["auc"])
best["mcw"] = float(bm["label"].split("=")[1])
log(f"  >>> mcw={best['mcw']} AUC={bm['auc']:.4f}")

# 阶段4: ra
log(f"\n=== [阶段4] reg_alpha (mcw={best['mcw']}) ===")
for ra in [0.1, 0.5, 1.0]:
    test(f"ra={ra}", ra=ra)
br = max([r for r in all_r if "ra=" in r["label"]], key=lambda x: x["auc"])
best["ra"] = float(br["label"].split("=")[1])
log(f"  >>> ra={best['ra']} AUC={br['auc']:.4f}")

log(f"\n{'='*60}")
log(f"最优: depth={best['depth']}, lr={best['lr']}, mcw={best['mcw']}, ra={best['ra']}")
log("Top 5:")
for i, r in enumerate(sorted(all_r, key=lambda x: -x["auc"])[:5]):
    log(f"  {i+1}. {r['label']:<20} AUC={r['auc']:.4f}  F1={r['f1']:.4f}")
log(f"耗时: {(time.time()-t0)/60:.1f} min")
log("DONE")
LOG.close()
