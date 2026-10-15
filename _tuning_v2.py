# -*- coding: utf-8 -*-
"""
批量调参脚本 V2 —— 独立实现，不依赖 importlib
固定 W=180，按阶段测试不同参数
"""
import sys, os, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.chdir(Path(__file__).resolve().parent)
BASE = Path(".").resolve()


def gen_label(v):
    if pd.isna(v): return np.nan
    return 1 if v < 0 else 0


# =====================================================================
# 特征工程（直接内联，从 43_3 复制核心逻辑）
# =====================================================================
def build_all_features(merged):
    """构建全部特征（利率时序+利差+美元+股债+杠杆+技术面+跨资产+波动率+季节性）"""
    features = merged.sort_values("Unnamed: 0").copy()
    rate = features["rate"]

    # 利率时序
    for lag in [1, 2, 5, 10, 20]:
        features[f"rate_lag_{lag}"] = rate.shift(lag)
    for window in [5, 10, 20]:
        features[f"rate_ma_{window}"] = rate.rolling(window=window, min_periods=1).mean()
        features[f"rate_std_{window}"] = rate.rolling(window=window, min_periods=2).std()
    features["rate_change_1d"] = rate.diff(1)
    features["rate_change_5d"] = rate.diff(5)
    features["rate_change_10d"] = rate.diff(10)
    features["curve_ma_gap_5_20"] = features["rate_ma_5"] - features["rate_ma_20"]
    features["curve_ma_gap_10_20"] = features["rate_ma_10"] - features["rate_ma_20"]
    for col in ["momentum_1_robust_scaled", "momentum_7_robust_scaled", "momentum_30_robust_scaled"]:
        if col in features.columns:
            features[f"{col}_lag_1"] = features[col].shift(1)
            features[f"{col}_lag_5"] = features[col].shift(5)

    # 信用利差
    for cs_col in ["1年信用利差", "3年信用利差", "5年信用利差"]:
        if cs_col in features.columns:
            features[f"{cs_col}_chg_5d"] = features[cs_col].diff(5)
            features[f"{cs_col}_chg_20d"] = features[cs_col].diff(20)
            features[f"{cs_col}_ma_20"] = features[cs_col].rolling(20, min_periods=1).mean()
    if "5年信用利差" in features.columns and "1年信用利差" in features.columns:
        features["credit_term_spread_5_1"] = features["5年信用利差"] - features["1年信用利差"]

    # 美元/汇率
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
        features["rate_dollar_gap"] = rate - features["美元指数"].rolling(60, min_periods=1).mean() / 20

    # 股债相对回报率
    if "股债相对回报率" in features.columns:
        features["股债相对回报率_chg_5d"] = features["股债相对回报率"].diff(5)
        features["股债相对回报率_chg_20d"] = features["股债相对回报率"].diff(20)
        features["股债相对回报率_ma_5"] = features["股债相对回报率"].rolling(5, min_periods=1).mean()
        features["股债相对回报率_ma_20"] = features["股债相对回报率"].rolling(20, min_periods=1).mean()
        features["股债相对回报率_deviation_20"] = features["股债相对回报率"] - features["股债相对回报率_ma_20"]

    # 加杠杆情绪
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

    # 利率相对强度
    if "rate_std_20" in features.columns:
        features["rate_chg_5d_zscore"] = rate.diff(5) / features["rate_std_20"]
        features["rate_chg_10d_zscore"] = rate.diff(10) / features["rate_std_20"]
    rw = rate.rolling(60, min_periods=60)
    features["rate_percentile_60"] = rw.rank(pct=True)
    features["rate_zscore_60"] = (rate - rw.mean()) / rw.std()

    # === 技术面动量 ===
    delta = rate.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    for period in [6, 14, 24]:
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        features[f"rsi_{period}"] = 100 - 100 / (1 + rs)
    bb_mean = rate.rolling(20, min_periods=20).mean()
    bb_std = rate.rolling(20, min_periods=20).std()
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    features["bb_pctb_20"] = (rate - bb_lower) / (bb_upper - bb_lower + 1e-10)
    features["bb_width_20"] = (bb_upper - bb_lower) / (bb_mean + 1e-10)
    for period in [5, 10, 20]:
        shifted = rate.shift(period)
        features[f"roc_{period}"] = (rate - shifted) / (shifted.abs() + 1e-10)
    # AR(1)
    for period in [10, 20, 60]:
        features[f"ar1_{period}"] = rate.rolling(period).corr(rate.shift(1))
    # MACD
    ema12 = rate.ewm(span=12, adjust=False).mean()
    ema26 = rate.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    features["macd_hist"] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

    # === 跨资产背离 ===
    commodity_cols = {
        "期货结算价(活跃合约):螺纹钢:前一个观测值": "rebar",
        "中国:水泥价格指数:前一个观测值": "cement",
    }
    rate_mom = rate.pct_change(20)
    rate_mom_std = rate_mom.rolling(60, min_periods=20).std() + 1e-10
    for raw_col, short_name in commodity_cols.items():
        if raw_col in features.columns:
            comm_mom = features[raw_col].pct_change(20)
            comm_mom_std = comm_mom.rolling(60, min_periods=20).std() + 1e-10
            features[f"divergence_{short_name}"] = (comm_mom / comm_mom_std) - (rate_mom / rate_mom_std)
            features[f"corr_rate_{short_name}_20"] = rate.rolling(20).corr(features[raw_col])
    for fx_col in ["即期汇率:美元兑人民币", "美元指数"]:
        if fx_col not in features.columns: continue
        short = "usdcny" if "人民币" in fx_col else "dollar"
        corr_20 = rate.rolling(20).corr(features[fx_col])
        corr_60 = rate.rolling(60).corr(features[fx_col])
        features[f"corr_break_{short}"] = corr_20 - corr_60

    # === 波动率结构 ===
    daily_change = rate.diff().abs()
    for sw in [5, 10, 20]:
        real_vol = daily_change.rolling(sw, min_periods=sw).std()
        features[f"realized_vol_{sw}"] = real_vol
        features[f"vol_percentile_{sw}"] = real_vol.rolling(120, min_periods=60).rank(pct=True)
    vol_20 = daily_change.rolling(20, min_periods=10).std()
    features["vol_of_vol_60"] = vol_20.rolling(60, min_periods=30).std()

    # === 季节性 ===
    dates = pd.to_datetime(features["Unnamed: 0"])
    month = dates.dt.month
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    features["is_quarter_end_month"] = dates.dt.month.isin([3, 6, 9, 12]).astype(int)

    # === 清理 ===
    features = features.drop(columns=["time", "label", "rate_delta_5d", "date_list"], errors="ignore").copy()
    features = features.set_index("Unnamed: 0")
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(99.00)
    return features


def run_one(features, y_all, W=180, S=63, V=63,
            max_depth=3, n_estimators=600, learning_rate=0.05,
            min_child_weight=8.0, reg_alpha=0.5):
    """单次滚动训练，返回指标 dict"""
    n = len(features)
    all_preds, all_probas, all_true = [], [], []
    fold = 0
    for start in range(W, n - S + 1, S):
        end = min(start + S, n)
        x_tr = features.iloc[start - W:start - V]
        y_tr = y_all.iloc[start - W:start - V]
        x_val = features.iloc[start - V:start]
        y_val = y_all.iloc[start - V:start]
        x_te = features.iloc[start:end]
        y_te = y_all.iloc[start:end]
        if len(x_tr) < 30:
            continue
        m = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=min_child_weight, reg_alpha=reg_alpha,
            reg_lambda=2.0, gamma=0.1, random_state=42,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", early_stopping_rounds=50,
        )
        m.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)
        all_preds.extend(m.predict(x_te).tolist())
        all_probas.extend(m.predict_proba(x_te).tolist())
        all_true.extend(y_te.values.tolist())
        fold += 1
    if not all_true:
        return None
    y_pred, y_true = np.array(all_preds), np.array(all_true)
    y_proba = np.array(all_probas)
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    except ValueError:
        auc = float("nan")
    maj = int(pd.Series(y_true).mode().iloc[0])
    maj_acc = float(np.mean(y_true == maj))
    return {"folds": fold, "n_test": len(y_true), "acc": acc, "f1": mf1,
            "auc": auc, "base": maj_acc}


def main():
    log_file = BASE / "_tuning_log.txt"
    log_file.write_text("", encoding="utf-8")
    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    t0 = time.time()
    log("=" * 70)
    log("[准备] 加载数据并构建特征...")

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

    features = build_all_features(merged)
    # 合并遗传特征
    gp_file = BASE / "data/43_x.csv"
    if gp_file.exists():
        gp = pd.read_csv(gp_file, index_col=0)
        gp.index = pd.to_datetime(gp.index)
        gp = gp.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        keep_cols = ["gp_feature_1", "gp_feature_2", "gp_feature_3", "gp_feature_4", "gp_feature_8"]
        gp = gp[[c for c in keep_cols if c in gp.columns]]
        features = features.join(gp, how="left")
        for c in gp.columns:
            if c in features.columns:
                features[c] = features[c].fillna(0.0)

    cutoff = pd.to_datetime("2018-01-01")
    mask = pd.to_datetime(merged["Unnamed: 0"]) >= cutoff
    features = features.loc[mask.values].reset_index(drop=True)
    y_all = y_all.loc[mask.values].reset_index(drop=True)
    merged = merged.loc[mask].reset_index(drop=True)
    log(f"  样本: {len(features)}, 特征数: {features.shape[1]}")

    # 当前最优参数
    best = {"S": 63, "depth": 3, "lr": 0.05, "mcw": 8.0, "ra": 0.5}
    results = []

    def test(label, **kw):
        params = {"W": 180, "S": best["S"], "V": 63,
                  "max_depth": best["depth"], "learning_rate": best["lr"],
                  "min_child_weight": best["mcw"], "reg_alpha": best["ra"],
                  **kw}
        try:
            log(f"  开始测试 {label} ...")
            r = run_one(features, y_all, **params)
        except Exception as e:
            log(f"  [错误] {label}: {e}")
            import traceback
            log(traceback.format_exc())
            return None
        if r:
            r["label"] = label
            results.append(r)
            log(f"  {label:<30} folds={r['folds']:>3}  Acc={r['acc']:.4f}  F1={r['f1']:.4f}  AUC={r['auc']:.4f}")
        else:
            log(f"  {label:<30} 无结果（样本不足）")
        return r

    # 阶段1: step
    log("\n[阶段1] step [42, 63]")
    for s in [42, 63]:
        test(f"step={s}", S=s)
    best_s = max([r for r in results if "step=" in r["label"]], key=lambda x: x["auc"])
    best["S"] = int(best_s["label"].split("=")[1])
    log(f"  >>> 最优 step={best['S']} (AUC={best_s['auc']:.4f})")

    # 阶段2: max_depth
    log(f"\n[阶段2] max_depth [2, 3, 4] (step={best['S']})")
    for d in [2, 3, 4]:
        test(f"depth={d}", max_depth=d)
    best_d = max([r for r in results if "depth=" in r["label"]], key=lambda x: x["auc"])
    best["depth"] = int(best_d["label"].split("=")[1])
    log(f"  >>> 最优 depth={best['depth']} (AUC={best_d['auc']:.4f})")

    # 阶段3: learning_rate
    log(f"\n[阶段3] lr [0.03, 0.05, 0.1] (depth={best['depth']})")
    for lr in [0.03, 0.05, 0.1]:
        test(f"lr={lr}", learning_rate=lr)
    best_lr = max([r for r in results if r["label"].startswith("lr=")], key=lambda x: x["auc"])
    best["lr"] = float(best_lr["label"].split("=")[1])
    log(f"  >>> 最优 lr={best['lr']} (AUC={best_lr['auc']:.4f})")

    # 阶段4: min_child_weight
    log(f"\n[阶段4] min_child_weight [3, 5, 8] (lr={best['lr']})")
    for mcw in [3, 5, 8]:
        test(f"mcw={mcw}", min_child_weight=float(mcw))
    best_mcw = max([r for r in results if "mcw=" in r["label"]], key=lambda x: x["auc"])
    best["mcw"] = float(best_mcw["label"].split("=")[1])
    log(f"  >>> 最优 mcw={best['mcw']} (AUC={best_mcw['auc']:.4f})")

    # 阶段5: reg_alpha
    log(f"\n[阶段5] reg_alpha [0.1, 0.5, 1.0] (mcw={best['mcw']})")
    for ra in [0.1, 0.5, 1.0]:
        test(f"ra={ra}", reg_alpha=ra)
    best_ra = max([r for r in results if "ra=" in r["label"]], key=lambda x: x["auc"])
    best["ra"] = float(best_ra["label"].split("=")[1])
    log(f"  >>> 最优 ra={best['ra']} (AUC={best_ra['auc']:.4f})")

    # 汇总
    elapsed = time.time() - t0
    log("\n" + "=" * 70)
    log("调参完成！最终最优组合：")
    log(f"  step={best['S']}, depth={best['depth']}, lr={best['lr']}, "
        f"mcw={best['mcw']}, ra={best['ra']}")
    log(f"\n全局 Top 5 (按 AUC):")
    for i, r in enumerate(sorted(results, key=lambda x: -x["auc"])[:5]):
        log(f"  {i+1}. {r['label']:<20} AUC={r['auc']:.4f}  F1={r['f1']:.4f}  Acc={r['acc']:.4f}")
    log(f"\n总耗时: {elapsed/60:.1f} 分钟")
    log("DONE")


if __name__ == "__main__":
    main()
