# -*- coding: utf-8 -*-
"""
10年期国债收益率未来5天方向预测 —— 二分类 XGBoost 滚动训练（2026新版）

基于 43_3_华泰证券债券预测_XGB训练_二分类.py 改造。

与原版的核心区别：
  1. 列名适配新数据格式（即期汇率、中间价、股债相对等列名简化）
  2. 新增海外资产特征（美债利差、中美利差、美股动量）
  3. 新增机构净买入特征（券商/保险/基金/理财/货币基金）
  4. 新增国债期货特征（活跃券成交量、期货持仓变化）
  5. 删除新数据中不存在的列引用（成交量:R007等）

标签：下行=1 / 上行=0
模型：XGBClassifier objective="binary:logistic"
训练：滚动窗口 W=180, S=63, V=63

输出文件（model_new/data/ 目录）：
  xgb_metrics_bin.txt              -> 指标汇总
  xgb_test_predictions_bin.csv     -> 测试集预测明细
  xgb_model_bin.joblib             -> 最终模型
  xgb_feature_importance_bin.csv   -> 特征重要性
  xgb_fold_metrics_bin.csv         -> 逐fold指标
  xgb_feature_stability_bin.csv    -> 特征稳定性统计
"""

import sys
import os
import argparse
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import warnings
# 抑制 pandas fragmentation 警告和 sklearn penalty 弃用警告（仅噪音）
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from statsmodels.tsa.seasonal import STL

os.chdir(Path(__file__).resolve().parent.parent)  # 切到项目根目录
BASE = Path(__file__).resolve().parent.parent

# 全局 args 缓存：供 build_features 等无参函数读取 CLI 配置（STL 周期等）
_ARGS_CACHE = type("ArgsCache", (), {})()


# =====================================================================
# CLI 参数（调参最优值已设为默认）
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="二分类 XGBoost 滚动训练（2026新版）")
    parser.add_argument("--start-date", default="2018-01-01")
    # === 预测期限（horizon）===
    parser.add_argument("--horizon", type=int, default=5,
                        help="预测期限（交易日）：5=周预测，10=双周预测")
    parser.add_argument("--suffix", default="",
                        help="输出文件后缀（如 _10d），空=默认_bin")
    parser.add_argument("--rolling-window", type=int, default=180)
    parser.add_argument("--rolling-step", type=int, default=63)
    parser.add_argument("--rolling-val-size", type=int, default=63)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--min-child-weight", type=float, default=8.0,
                        help="叶子节点最小样本权重和，12+会在单边市压死所有特征")
    parser.add_argument("--reg-alpha", type=float, default=1.5,
                        help="L1 正则化强度，1.5 在强趋势市中平衡信号与过拟合")
    parser.add_argument("--reg-lambda", type=float, default=2.0,
                        help="L2 正则化强度")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="节点分裂最小增益阈值，降低则更多弱特征参与分裂")
    parser.add_argument("--genetic-features-file", default="model_new/data/x_genetic_bin.csv")
    parser.add_argument("--no-genetic", action="store_true")
    parser.add_argument("--keep-cols-file", default=None,
                        help="特征保留列表文件（每行一个列名），由 04_特征筛选.py 生成。"
                             "指定后只保留列表中的列，用于对比实验")
    # === STL 分解相关（优化1/3）===
    parser.add_argument("--no-stl", action="store_true",
                        help="禁用 STL 分解特征（默认启用）")
    parser.add_argument("--stl-period", type=int, default=63,
                        help="STL 季节性周期，63=季度, 126=半年, 252=年度")
    # === 超趋势标签（优化2）===
    parser.add_argument("--detrend-label", action="store_true",
                        help="启用'超趋势方向'标签：实际变动 vs 趋势预期变动，剔除趋势贡献")
    # === 概率校准器（优化4）===
    parser.add_argument("--no-calibrator", action="store_true",
                        help="禁用 XGBoost 后接 LR 概率校准器")
    # === 置信度门控（P2 优化）===
    parser.add_argument("--no-gating", action="store_true",
                        help="禁用置信度门控评估（默认启用）")
    parser.add_argument("--gating-prob-margin", type=float, default=0.15,
                        help="概率模糊带半宽：|prob-0.5|<此值 且 高波动时 → 弃权")
    parser.add_argument("--gating-vol-quantile", type=float, default=0.5,
                        help="波动率门控分位：vol_regime 高于此分位视为高波动体制")
    # === 2025 震荡市专项优化 ===
    parser.add_argument("--adaptive-window", action="store_true", default=True,
                        help="自适应窗口：震荡市(vol_regime骤降)时自动缩短W，加速旧数据滑出")
    parser.add_argument("--no-adaptive-window", dest="adaptive_window", action="store_false")
    parser.add_argument("--label-dead-zone", type=float, default=2.0,
                        help="标签死区(bp)：|rate_delta_5d|<此值的样本降权，0=禁用(默认2.0)")
    parser.add_argument("--no-detrend-label", dest="detrend_label", action="store_false",
                        help="禁用超趋势标签")
    parser.set_defaults(detrend_label=True)
    # === 优化E：CUSUM 变点检测 ===
    parser.add_argument("--cusum-threshold", type=float, default=2.0,
                        help="CUSUM 变点检测灵敏度阈值（越大越保守，0=禁用）")
    # === 复用预存特征（绕过 STL 内存瓶颈）===
    parser.add_argument("--load-features", default=None,
                        help="从 CSV 加载预构建特征（如 xgb_train_features_bin.csv），跳过 build_features")
    return parser.parse_args()


# =====================================================================
# 二分类标签
# =====================================================================

def gen_label(v):
    """下行=1, 上行=0，只看方向不看幅度"""
    if pd.isna(v):
        return np.nan
    return 1 if v < 0 else 0


# =====================================================================
# STL 时间序列分解特征（优化1/3）
# 思路参考：STL 分解 → rate = trend + seasonal + residual
#           XGBoost 不直接学原始 rate，而是获得分解后的结构化信号
# =====================================================================

def add_stl_decomposition_features(features, rate_col="rate", stl_period=63):
    """
    基于 STL 分解构建趋势/季节/偏离度特征，原地修改 features。

    核心特征：
      - stl_trend         : 趋势成分（结构性下行/上行通道）
      - stl_seasonal      : 季节性成分（年内资金紧张等周期性）
      - stl_residual      : 残差成分（短期冲击）
      - dev_from_trend    : 收益率偏离趋势的幅度（均值回归压力）
      - dev_zscore_120    : 偏离度的历史分位（极端偏离 = 反转信号）
      - trend_slope_20    : 趋势斜率（加速/减速）
      - season_position   : 季节性位置（年内相对位置）

    参数：
      stl_period : STL 季节性周期（日频下 63=季, 126=半年, 252=年）
    """
    rate = features[rate_col]

    # === STL 分解（robust=True 抗异常值）===
    # 注意：STL 要求至少 2 个完整周期，前段会有 NaN
    try:
        stl = STL(rate.fillna(rate.mean()), period=stl_period, robust=True)
        res = stl.fit()
        features["stl_trend"] = res.trend
        features["stl_seasonal"] = res.seasonal
        features["stl_residual"] = res.resid
    except Exception as e:
        print(f"  [警告] STL 分解失败: {e}，跳过 STL 特征", flush=True)
        return

    # === 偏离度特征（关键：捕捉均值回归压力）===
    # 收益率相对趋势的偏离，正值=在趋势上方，负值=在趋势下方
    dev = rate - features["stl_trend"]
    features["dev_from_trend"] = dev

    # 偏离度的滚动 z-score（120 天历史分位）
    dev_std = dev.rolling(120, min_periods=60).std()
    features["dev_zscore_120"] = dev / (dev_std + 1e-10)

    # 偏离度的 20 日变动（偏离在加速还是收敛）
    features["dev_chg_20d"] = dev.diff(20)

    # === 优化2：均值回归增强（非线性反转信号）===
    # 当 |dev_zscore| 超过阈值时，收益率偏离趋势过远，均值回归概率增大
    # 使用立方函数非线性放大极端偏离，让模型在反转点附近更敏感
    z = features["dev_zscore_120"]
    features["reversal_signal"] = np.sign(z) * (z.abs() ** 3) / 10  # 立方放大，除10缩放
    # 反转压力 Flag：|z|>1.5 时标记为"高反转压力"
    features["reversal_flag"] = (z.abs() > 1.5).astype(float)

    # === 趋势斜率特征（趋势加速度）===
    # 趋势本身的 20 日差分 = 趋势斜率；斜率的变动 = 加速度
    features["trend_slope_20"] = features["stl_trend"].diff(20)
    features["trend_accel"] = features["trend_slope_20"].diff(20)

    # === 季节性位置特征 ===
    # 当前季节性成分相对其历史最大最小值的位置（0~1）
    season = features["stl_seasonal"]
    rolling_max = season.rolling(252, min_periods=63).max()
    rolling_min = season.rolling(252, min_periods=63).min()
    features["season_position"] = (season - rolling_min) / (rolling_max - rolling_min + 1e-10)

    # === 残差自回归特征 ===
    # 残差具有较强的自相关性，捕捉其短期延续
    features["stl_resid_lag_1"] = features["stl_residual"].shift(1)
    features["stl_resid_lag_5"] = features["stl_residual"].shift(5)
    features["stl_resid_ma_5"] = features["stl_residual"].rolling(5, min_periods=1).mean()


# =====================================================================
# 多层次趋势偏离度（优化3）
# 不依赖 STL，用滚动回归提取多周期趋势，与 STL 互补
# =====================================================================

def add_multilevel_trend_features(features, rate_col="rate"):
    """
    构建多周期（60/126/252 天）趋势偏离度特征。

    与 STL 区别：
      - STL 用单一周期分解，这里用多个滚动窗口的线性回归斜率
      - 可捕捉多周期共振（短中长期趋势同时指向某方向 = 高置信度）
    """
    rate = features[rate_col].values
    n = len(rate)

    for window in [60, 126, 252]:
        if n < window:
            continue
        t_vals = np.arange(window, dtype=np.float64)
        t_mean = t_vals.mean()
        t_var = ((t_vals - t_mean) ** 2).sum()

        slopes, devs = [], []
        for i in range(n):
            if i < window - 1:
                slopes.append(np.nan)
                devs.append(np.nan)
                continue
            w = rate[i - window + 1: i + 1]
            y_mean = w.mean()
            slope = ((t_vals - t_mean) * (w - y_mean)).sum() / t_var
            # 用线性趋势预测当前时点的"趋势值"
            trend_val = slope * (window - 1 - t_mean) + y_mean
            slopes.append(slope)
            devs.append(rate[i] - trend_val)

        slopes_s = pd.Series(slopes, index=features.index)
        devs_s = pd.Series(devs, index=features.index)
        features[f"lr_trend_slope_{window}"] = slopes_s
        features[f"lr_dev_from_trend_{window}"] = devs_s
        # 偏离度的 z-score（极端偏离捕捉）
        dev_std = devs_s.rolling(252, min_periods=60).std()
        features[f"lr_dev_zscore_{window}"] = devs_s / (dev_std + 1e-10)

    # 多周期趋势共振：3 个周期斜率符号是否一致
    cols = [f"lr_trend_slope_{w}" for w in [60, 126, 252] if f"lr_trend_slope_{w}" in features.columns]
    if len(cols) >= 2:
        signs = np.sign(features[cols])
        # 共振度 = 一致方向的数量 / 周期数（1=全同向，0=完全冲突）
        features["trend_resonance"] = signs.sum(axis=1).abs() / len(cols)


# =====================================================================
# 辅助函数
# =====================================================================

def _hurst_rs(ts):
    """R/S 法估计 Hurst 指数"""
    n = len(ts)
    if n < 20:
        return np.nan
    lags = []
    for size in [n // 2, n // 4, n // 8, n // 16]:
        if size >= 5 and n // size >= 2:
            lags.append(size)
    if len(lags) < 2:
        return np.nan
    rs_values = []
    for size in lags:
        num_chunks = n // size
        if num_chunks < 1:
            continue
        rs_list = []
        for c in range(num_chunks):
            chunk = ts[c * size: (c + 1) * size]
            mean = chunk.mean()
            cumdev = np.cumsum(chunk - mean)
            r = cumdev.max() - cumdev.min()
            s = chunk.std()
            if s > 1e-10:
                rs_list.append(r / s)
        if rs_list:
            rs_values.append((np.log(size), np.log(np.mean(rs_list))))
    if len(rs_values) < 2:
        return np.nan
    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])
    h, _ = np.polyfit(x, y, 1)
    return float(np.clip(h, 0.0, 1.0))


# =====================================================================
# 技术面动量特征（RSI / 布林带 / MACD / ROC / Williams%R / AR(1) / Hurst / FFT）
# =====================================================================

def add_technical_momentum_features(features, price_col="rate"):
    """
    基于价格序列构建技术面动量特征，原地修改 features。
    所有计算严格使用 shift/rolling，无未来信息泄露。
    """
    price = features[price_col]
    delta = price.diff()

    # === RSI ===
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    for period in [6, 14, 24]:
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        features[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    # === 布林带 ===
    bb_mean = price.rolling(20, min_periods=20).mean()
    bb_std = price.rolling(20, min_periods=20).std()
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    features["bb_pctb_20"] = (price - bb_lower) / (bb_upper - bb_lower + 1e-10)
    features["bb_width_20"] = (bb_upper - bb_lower) / (bb_mean + 1e-10)

    # === MACD ===
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    features["macd"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd_line - macd_signal

    # === ROC ===
    for period in [5, 10, 20]:
        shifted = price.shift(period)
        features[f"roc_{period}"] = (price - shifted) / (shifted.abs() + 1e-10)

    # === Williams %R ===
    for period in [14, 28]:
        roll_high = price.rolling(period, min_periods=period).max()
        roll_low = price.rolling(period, min_periods=period).min()
        features[f"williams_r_{period}"] = (roll_high - price) / (roll_high - roll_low + 1e-10) * (-100)

    # === AR(1) 自回归系数 ===
    shifted1 = price.shift(1)
    for period in [10, 20, 60]:
        features[f"ar1_{period}"] = price.rolling(period).corr(shifted1)

    # === 趋势斜率 + R² ===
    for period in [20, 60]:
        t_vals = np.arange(period, dtype=np.float64)
        t_mean = t_vals.mean()
        t_var = ((t_vals - t_mean) ** 2).sum()
        slopes, r2s = [], []
        values = price.values
        for i in range(len(price)):
            if i < period - 1:
                slopes.append(np.nan); r2s.append(np.nan); continue
            window = values[i - period + 1: i + 1]
            y_mean = window.mean()
            slope = ((t_vals - t_mean) * (window - y_mean)).sum() / t_var
            y_pred = slope * (t_vals - t_mean) + y_mean
            ss_res = ((window - y_pred) ** 2).sum()
            ss_tot = ((window - y_mean) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            slopes.append(slope); r2s.append(r2)
        features[f"trend_slope_{period}"] = slopes
        features[f"trend_r2_{period}"] = r2s

    # === 反转计数 ===
    sign = np.sign(delta.fillna(0).values)
    for period in [10, 20, 60]:
        reversals, count, prev = [], 0, 0
        for s in sign:
            if s != 0 and prev != 0 and s != prev:
                count += 1
            if s != 0:
                prev = s
            reversals.append(count)
        rev_series = pd.Series(reversals)
        features[f"reversal_count_{period}"] = rev_series.diff(period).fillna(rev_series)

    # === Hurst 指数 ===
    for period in [60, 120]:
        hurst_vals = []
        values = price.values
        for i in range(len(price)):
            if i < period - 1:
                hurst_vals.append(np.nan); continue
            hurst_vals.append(_hurst_rs(values[i - period + 1: i + 1]))
        features[f"hurst_{period}"] = hurst_vals

    # === FFT 主频能量比 ===
    for period in [60]:
        fft_ratios = []
        values = price.values
        for i in range(len(price)):
            if i < period - 1:
                fft_ratios.append(np.nan); continue
            window = values[i - period + 1: i + 1]
            spectrum = np.abs(np.fft.rfft(window - window.mean()))
            total_energy = (spectrum ** 2).sum() + 1e-10
            fft_ratios.append((spectrum ** 2).max() / total_energy)
        features[f"fft_max_ratio_{period}"] = fft_ratios


# =====================================================================
# 跨资产背离特征
# =====================================================================

def add_cross_asset_divergence_features(features):
    """
    跨资产相对动量 / 背离特征。
    适配新列名：螺纹钢、水泥、阴极铜、南华等。
    新增：中美利差、美股动量。
    """
    rate = features["rate"]

    # === 1. 利率-商品背离 ===
    # 新列名映射（去掉"前一个观测值"后缀）
    commodity_cols = {
        "期货结算价(活跃合约):螺纹钢": "rebar",
        "水泥价格指数": "cement",
        "阴极铜价格": "copper",
        "南华工业品指数": "nanhua",
    }
    for raw_col, short in commodity_cols.items():
        if raw_col not in features.columns:
            continue
        comm = features[raw_col]
        rate_mom = rate.pct_change(20)
        comm_mom = comm.pct_change(20)
        rate_std = rate_mom.rolling(60, min_periods=20).std() + 1e-10
        comm_std = comm_mom.rolling(60, min_periods=20).std() + 1e-10
        features[f"divergence_{short}"] = (comm_mom / comm_std) - (rate_mom / rate_std)
        features[f"corr_rate_{short}_20"] = rate.rolling(20).corr(comm)
        features[f"corr_rate_{short}_60"] = rate.rolling(60).corr(comm)

    # === 2. 利率-汇率相关性突变 ===
    # 新列名：即期汇率、中间价
    if "即期汇率" in features.columns:
        fx = features["即期汇率"]
        short = "usdcny"
        corr_20 = rate.rolling(20).corr(fx)
        corr_60 = rate.rolling(60).corr(fx)
        features[f"corr_rate_{short}_20"] = corr_20
        features[f"corr_rate_{short}_60"] = corr_60
        features[f"corr_break_{short}"] = corr_20 - corr_60
    if "美元指数" in features.columns:
        dollar = features["美元指数"]
        corr_20 = rate.rolling(20).corr(dollar)
        corr_60 = rate.rolling(60).corr(dollar)
        features["corr_rate_dollar_20"] = corr_20
        features["corr_rate_dollar_60"] = corr_60
        features["corr_break_dollar"] = corr_20 - corr_60

    # === 3. 中美利差（新增） ===
    # 利用新数据中的"10年美债"列
    if "10年美债" in features.columns and "rate" in features.columns:
        us_10y = features["10年美债"]
        features["cn_us_spread_10y"] = rate - us_10y  # 中美10年期利差
        features["cn_us_spread_10d_chg"] = features["cn_us_spread_10y"].diff(10)
        features["cn_us_spread_20d_chg"] = features["cn_us_spread_10y"].diff(20)
    if "2年美债" in features.columns and "10年美债" in features.columns:
        # 美债期限利差（2Y-10Y）
        features["us_term_spread"] = features["10年美债"] - features["2年美债"]

    # === 4. 股债跷跷板 ===
    # 新列名：股债相对
    if "股债相对" in features.columns:
        stock_bond = features["股债相对"]
        features["stock_bond_swing_5"] = stock_bond.pct_change(5)
        features["stock_bond_swing_20"] = stock_bond.pct_change(20)
        features["corr_stock_bond_rate_20"] = stock_bond.rolling(20).corr(rate)
        features["corr_stock_bond_rate_60"] = stock_bond.rolling(60).corr(rate)

    # === 5. 海外股市动量（新增） ===
    # 利用新数据中的美股、A股指数
    for idx_col, short in [("标普500", "sp500"), ("沪深300", "csi300"), ("恒生科技", "hsi_tech")]:
        if idx_col not in features.columns:
            continue
        idx = features[idx_col]
        features[f"{short}_mom_5"] = idx.pct_change(5)
        features[f"{short}_mom_20"] = idx.pct_change(20)
        # 股市与利率的滚动相关性
        features[f"corr_rate_{short}_20"] = rate.rolling(20).corr(idx)

    # === 6. 机构净买入特征（新增） ===
    # 利用新数据中的净买入列
    net_buy_cols = {
        "净买入：基金": "fund",
        "净买入：券商自营": "broker",
        "净买入：保险": "insurance",
        "净买入：理财": "wealth_mgmt",
    }
    for col, short in net_buy_cols.items():
        if col not in features.columns:
            continue
        nb = features[col]
        features[f"netbuy_{short}_5d"] = nb.rolling(5).mean()
        features[f"netbuy_{short}_20d"] = nb.rolling(20).mean()
        features[f"netbuy_{short}_chg"] = nb.diff(5)

    # === 7. 波动率结构 ===
    add_volatility_structure_features(features, rate)


def add_volatility_structure_features(features, rate):
    """波动率结构特征（已实现波动率、波动率锥、vol of vol）"""
    daily_change = rate.diff().abs()

    # 已实现波动率 + 历史分位
    for sw in [5, 10, 20]:
        real_vol = daily_change.rolling(sw, min_periods=sw).std()
        features[f"realized_vol_{sw}"] = real_vol
        features[f"vol_percentile_{sw}"] = real_vol.rolling(120, min_periods=60).rank(pct=True)

    # vol of vol
    vol_20 = daily_change.rolling(20, min_periods=10).std()
    features["vol_of_vol_60"] = vol_20.rolling(60, min_periods=30).std()
    features["vol_of_vol_120"] = vol_20.rolling(120, min_periods=60).std()

    # 波动率偏离 z-score
    for vw in [20, 60]:
        vol = daily_change.rolling(vw, min_periods=vw).std()
        vol_mean = vol.rolling(120, min_periods=60).mean()
        vol_std = vol.rolling(120, min_periods=60).std()
        features[f"vol_deviation_zscore_{vw}"] = (vol - vol_mean) / (vol_std + 1e-10)
        features[f"vol_ratio_{vw}"] = vol / (vol_mean + 1e-10)

    # 波动率体制
    vol_10 = daily_change.rolling(10, min_periods=5).std()
    vol_5 = daily_change.rolling(5, min_periods=3).std()
    features["vol_regime"] = vol_5 / (vol_10 + 1e-10)


# =====================================================================
# 季节性特征
# =====================================================================

def add_seasonality_features(features):
    """季节性 / 时间特征（sin/cos 周期编码）"""
    dates = pd.to_datetime(features["Unnamed: 0"])

    month = dates.dt.month
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)

    quarter = dates.dt.quarter
    features["quarter_sin"] = np.sin(2 * np.pi * quarter / 4)
    features["quarter_cos"] = np.cos(2 * np.pi * quarter / 4)

    # 距月末天数（dates 已是 Series，直接 apply）
    def _days_to_month_end(d):
        # 下个月1号 - 1天 = 当月最后一天
        if d.month == 12:
            next_first = pd.Timestamp(year=d.year + 1, month=1, day=1)
        else:
            next_first = pd.Timestamp(year=d.year, month=d.month + 1, day=1)
        return (next_first - pd.Timedelta(days=1) - d).days
    features["days_to_month_end"] = dates.apply(_days_to_month_end)

    # 距季末天数
    def _days_to_quarter_end(d):
        q_end_month = ((d.month - 1) // 3 + 1) * 3  # 当季最后月
        if q_end_month == 12:
            next_first = pd.Timestamp(year=d.year + 1, month=1, day=1)
        else:
            next_first = pd.Timestamp(year=d.year, month=q_end_month + 1, day=1)
        return (next_first - pd.Timedelta(days=1) - d).days
    features["days_to_quarter_end"] = dates.apply(_days_to_quarter_end)

    day_of_year = dates.dt.dayofyear
    features["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    features["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365)


# =====================================================================
# 主特征构建函数
# =====================================================================

def build_features(merged):
    """
    构建完整特征集，适配新数据列名。

    与原版差异：
    - 即期汇率:美元兑人民币 → 即期汇率
    - 中间价:美元兑人民币 → 中间价
    - 股债相对回报率 → 股债相对
    - 成交量:R001/...银行间质押式回购 → R001成交量占比
    - 成交量:R007 不存在（仅保留 R001 成交量特征）
    - 新增中美利差、美股动量、机构净买入等特征
    """
    features = merged.sort_values("Unnamed: 0").copy()

    # --- 利率时序特征 ---
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

    # --- 信用利差衍生 ---
    for cs_col in ["1年信用利差", "3年信用利差", "5年信用利差"]:
        if cs_col in features.columns:
            features[f"{cs_col}_chg_5d"] = features[cs_col].diff(5)
            features[f"{cs_col}_chg_20d"] = features[cs_col].diff(20)
            features[f"{cs_col}_ma_20"] = features[cs_col].rolling(20, min_periods=1).mean()
    if "5年信用利差" in features.columns and "1年信用利差" in features.columns:
        features["credit_term_spread_5_1"] = features["5年信用利差"] - features["1年信用利差"]

    # --- 美元/汇率衍生（新列名） ---
    if "美元指数" in features.columns:
        features["dollar_chg_5d"] = features["美元指数"].diff(5)
        features["dollar_chg_20d"] = features["美元指数"].diff(20)
        features["dollar_ma_20"] = features["美元指数"].rolling(20, min_periods=1).mean()
        features["dollar_deviation_20"] = features["美元指数"] - features["dollar_ma_20"]
        # 利率-美元偏离（原版最稳定特征）
        features["rate_dollar_gap"] = features["rate"] - features["美元指数"].rolling(60, min_periods=1).mean() / 20

    if "即期汇率" in features.columns:
        features["fx_chg_5d"] = features["即期汇率"].diff(5)
        features["fx_chg_20d"] = features["即期汇率"].diff(20)
        if "中间价" in features.columns:
            # 央行汇率干预信号：中间价 - 即期汇率
            features["fx_intervention_5d"] = (features["中间价"] - features["即期汇率"]).diff(5)

    # --- 股债相对回报率衍生（新列名：股债相对） ---
    if "股债相对" in features.columns:
        features["股债相对_chg_5d"] = features["股债相对"].diff(5)
        features["股债相对_chg_20d"] = features["股债相对"].diff(20)
        features["股债相对_ma_5"] = features["股债相对"].rolling(5, min_periods=1).mean()
        features["股债相对_ma_20"] = features["股债相对"].rolling(20, min_periods=1).mean()
        features["股债相对_deviation_20"] = features["股债相对"] - features["股债相对_ma_20"]

    # --- 加杠杆情绪衍生（新列名适配） ---
    if "成交量:R001" in features.columns:
        features["vol_r001_chg_5d"] = features["成交量:R001"].diff(5)
        features["vol_r001_chg_20d"] = features["成交量:R001"].diff(20)
        features["vol_r001_ma_5"] = features["成交量:R001"].rolling(5, min_periods=1).mean()
        features["vol_r001_ma_20"] = features["成交量:R001"].rolling(20, min_periods=1).mean()
    if "R001成交量占比" in features.columns:
        features["overnight_ratio_chg_5d"] = features["R001成交量占比"].diff(5)
        features["overnight_ratio_chg_20d"] = features["R001成交量占比"].diff(20)
        features["overnight_ratio_ma_5"] = features["R001成交量占比"].rolling(5, min_periods=1).mean()
        features["overnight_ratio_ma_20"] = features["R001成交量占比"].rolling(20, min_periods=1).mean()
        features["overnight_ratio_dev_20"] = features["R001成交量占比"] - features["overnight_ratio_ma_20"]
    # R001-R007 利差（新数据有 R001 和 R007 利率列）
    if "R001" in features.columns and "R007" in features.columns:
        features["r001_r007_spread"] = features["R007"] - features["R001"]
        features["r001_r007_spread_chg_5d"] = features["r001_r007_spread"].diff(5)
    if "DR001" in features.columns and "DR007" in features.columns:
        features["dr001_dr007_spread"] = features["DR007"] - features["DR001"]

    # --- 国债期货特征（新列名） ---
    if "国债期货持仓量:10年" in features.columns:
        features["tf_holdings_chg_5d"] = features["国债期货持仓量:10年"].diff(5)
        features["tf_holdings_ma_20"] = features["国债期货持仓量:10年"].rolling(20, min_periods=1).mean()
    if "国债期货收盘价(活跃合约):10年" in features.columns:
        tf_price = features["国债期货收盘价(活跃合约):10年"]
        features["tf_price_chg_5d"] = tf_price.diff(5)
        features["tf_price_ma_20"] = tf_price.rolling(20, min_periods=1).mean()

    # --- 利率相对强度 ---
    if "rate_std_20" in features.columns:
        features["rate_chg_5d_zscore"] = features["rate"].diff(5) / features["rate_std_20"]
        features["rate_chg_10d_zscore"] = features["rate"].diff(10) / features["rate_std_20"]
    rw = features["rate"].rolling(60, min_periods=60)
    features["rate_zscore_60"] = (features["rate"] - rw.mean()) / rw.std()

    # --- 跨资产背离特征 ---
    add_cross_asset_divergence_features(features)

    # --- 技术面动量特征 ---
    add_technical_momentum_features(features, "rate")

    # --- STL 时间序列分解特征（优化1：趋势/季节/偏离度）---
    if not getattr(_ARGS_CACHE, "no_stl", False):
        add_stl_decomposition_features(
            features, "rate",
            stl_period=getattr(_ARGS_CACHE, "stl_period", 63)
        )

    # --- 多层次趋势偏离度（优化3：60/126/252 多周期共振）---
    add_multilevel_trend_features(features, "rate")

    # --- 季节性特征 ---
    add_seasonality_features(features)

    # --- 清理 + inf 处理 ---
    features = features.drop(
        columns=["time", "label", "rate_delta_5d", "date_list"], errors="ignore"
    ).copy()
    features = features.set_index("Unnamed: 0")
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(99.00)
    return features


# =====================================================================
# 遗传特征加载
# =====================================================================

def load_genetic_features(args):
    """加载遗传特征（去重后保留 5 个）"""
    if args.no_genetic:
        print("  遗传特征: 已禁用", flush=True)
        return None

    gp_file = BASE / args.genetic_features_file
    if not gp_file.exists():
        print(f"  遗传特征: 文件不存在 {gp_file}", flush=True)
        return None

    gp = pd.read_csv(gp_file, index_col=0)
    gp.index = pd.to_datetime(gp.index)
    gp = gp.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    keep_cols = ["gp_feature_1", "gp_feature_2", "gp_feature_3", "gp_feature_4", "gp_feature_8"]
    gp = gp[[c for c in keep_cols if c in gp.columns]]
    print(f"  遗传特征: {len(gp.columns)} 个 <- {gp_file.name}", flush=True)
    return gp


# =====================================================================
# XGBoost 模型创建
# =====================================================================

def create_xgb_model(args):
    """创建二分类 XGBoost 模型（调参最优值）"""
    return XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        random_state=42,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        early_stopping_rounds=50,
    )


# =====================================================================
# 概率校准器（优化4）
# 思路参考：文章的"Residual Calibrator / Stacking"
#   将 XGBoost 的输出概率 + 关键结构化特征喂给一个轻量 LR，
#   让 LR 学会"在什么情境下应该修正 XGBoost 的判断"
# 为什么用 LR 而非 MLP：日频数据样本少，LR 不易过拟合且可解释
# =====================================================================

# 校准器使用的辅助特征（除 XGBoost 概率外）
# 优化4：加入市场状态特征（反转信号、趋势共振），让校准器学会在震荡市降低置信度
_CALIB_FEATURES = [
    "dev_from_trend", "dev_zscore_120",          # STL 偏离度
    "reversal_signal", "reversal_flag",           # 优化2：反转信号
    "lr_dev_zscore_60", "lr_dev_zscore_126",     # 多周期偏离度
    "trend_slope_20", "trend_resonance",         # 趋势与共振
    "vol_regime", "realized_vol_20",             # 波动率体制
]


def fit_calibrator(xgb_model, x_tr, y_tr, x_val, y_val, features_df):
    """
    在 XGBoost 已训练后，拟合 LR 概率校准器。

    流程：
      1. 用 XGBoost 在训练/验证集上输出概率
      2. 拼接 [xgb_prob] + 辅助结构化特征 → LR 输入
      3. 在验证集上训练 LR（小样本用验证集而非再切分）

    返回：(calibrator, scaler) 或 (None, None) 表示禁用/失败
    """
    # 训练集 LR 特征：xgb 概率 + 辅助特征
    prob_tr = xgb_model.predict_proba(x_tr)[:, 1]
    prob_val = xgb_model.predict_proba(x_val)[:, 1]

    aux_cols = [c for c in _CALIB_FEATURES if c in features_df.columns]
    aux_tr = features_df.loc[x_tr.index, aux_cols].values if aux_cols else np.empty((len(x_tr), 0))
    aux_val = features_df.loc[x_val.index, aux_cols].values if aux_cols else np.empty((len(x_val), 0))

    X_lr_tr = np.column_stack([prob_tr, aux_tr])
    X_lr_val = np.column_stack([prob_val, aux_val])

    # 标准化辅助特征（概率列保持原值，不标准化）
    scaler = StandardScaler()
    if aux_cols:
        scaler.fit(X_lr_val[:, 1:])  # 仅拟合辅助列
        X_lr_tr[:, 1:] = scaler.transform(X_lr_tr[:, 1:])
        X_lr_val[:, 1:] = scaler.transform(X_lr_val[:, 1:])

    # 用验证集训练 LR（强正则避免过拟合）
    calibrator = LogisticRegression(
        C=0.5, penalty="l2", solver="lbfgs", max_iter=200, random_state=42
    )
    try:
        calibrator.fit(X_lr_val, y_val.values)
    except Exception as e:
        print(f"  [警告] LR 校准器训练失败: {e}", flush=True)
        return None, None
    return calibrator, (scaler, aux_cols)


def calibrate_proba(xgb_model, calibrator_bundle, x_te, features_df):
    """
    用校准器输出测试集的概率。
    calibrator_bundle = (scaler, aux_cols) 或 None
    """
    if calibrator_bundle is None or calibrator_bundle[0] is None:
        # 校准器禁用 → 直接返回 XGBoost 原始概率
        return xgb_model.predict_proba(x_te)

    calibrator, (scaler, aux_cols) = calibrator_bundle
    prob_te = xgb_model.predict_proba(x_te)[:, 1]
    aux_te = features_df.loc[x_te.index, aux_cols].values if aux_cols else np.empty((len(x_te), 0))
    X_lr_te = np.column_stack([prob_te, aux_te])
    if aux_cols:
        X_lr_te[:, 1:] = scaler.transform(X_lr_te[:, 1:])
    prob_calibrated = calibrator.predict_proba(X_lr_te)
    return prob_calibrated


# =====================================================================
# CUSUM 变点检测（优化E）
# 双侧 CUSUM：检测利率序列的均值漂移，标记 regime 切换点
# =====================================================================

def _detect_cusum_changepoints(series, threshold=0.15):
    """
    双侧 CUSUM 变点检测。

    原理：累积利率的标准化残差，当累积和超过阈值时判定为变点。
    参数 threshold 控制灵敏度（越小越敏感，但误报也越多）。

    返回：变点索引列表（按时间排序）
    """
    s = np.asarray(series, dtype=float)
    # 前向填充 NaN
    mask = ~np.isnan(s)
    if mask.sum() < 20:
        return []
    # 用滚动均值作为参考（60天），避免局部趋势干扰
    roll_mean = pd.Series(s).rolling(60, min_periods=20).mean().bfill().values
    residual = s - roll_mean
    std = np.std(residual[mask]) + 1e-10

    cusum_pos = np.zeros(len(s))  # 正向累积（上升变点）
    cusum_neg = np.zeros(len(s))  # 负向累积（下降变点）
    changepoints = []

    for i in range(1, len(s)):
        if np.isnan(s[i]) or np.isnan(residual[i]):
            cusum_pos[i] = cusum_pos[i-1]
            cusum_neg[i] = cusum_neg[i-1]
            continue
        z = residual[i] / std
        cusum_pos[i] = max(0, cusum_pos[i-1] + z)
        cusum_neg[i] = min(0, cusum_neg[i-1] + z)

        if cusum_pos[i] > threshold * 10:  # 累积上升超过阈值
            changepoints.append(i)
            cusum_pos[i] = 0  # 重置
            cusum_neg[i] = 0
        elif cusum_neg[i] < -threshold * 10:  # 累积下降超过阈值
            changepoints.append(i)
            cusum_pos[i] = 0
            cusum_neg[i] = 0

    # 合并间距过近的变点（<20天内只保留第一个）
    if len(changepoints) < 2:
        return changepoints
    merged = [changepoints[0]]
    for cp in changepoints[1:]:
        if cp - merged[-1] >= 20:
            merged.append(cp)
    return merged


# =====================================================================
# 滚动窗口训练
# =====================================================================

def rolling_window_train(features, y_all, merged, args, sample_weights_all=None):
    """
    滚动窗口训练，返回所有 fold 的预测结果 + 逐 fold 指标 + 特征重要性。

    每个 fold：
    - 训练集: [start-W : start-V]
    - 验证集: [start-V : start]
    - 测试集: [start : start+step]
    """
    W, S, V = args.rolling_window, args.rolling_step, args.rolling_val_size
    W_min = max(90, W - 30)  # 自适应窗口下限（不低于90天）
    n = len(features)

    # 预计算 vol_regime 的全局分位，用于自适应窗口判断
    vol_regime_global = features["vol_regime"].values if "vol_regime" in features.columns else None

    # === 优化E：CUSUM 变点检测（预计算）===
    cusum_threshold = getattr(args, "cusum_threshold", 0.15)
    if cusum_threshold > 0:
        cusum_points = _detect_cusum_changepoints(merged["rate"].values, threshold=cusum_threshold)
        print(f"  CUSUM 变点检测: 发现 {len(cusum_points)} 个变点 (threshold={cusum_threshold})", flush=True)
    else:
        cusum_points = []

    all_preds, all_probas, all_true, all_dates, all_meta = [], [], [], [], []
    fold_metrics, fold_feature_imps = [], []
    fold = 0
    prev_proba_last = None  # 优化D：上一个 fold 最后一个样本的概率，用于 EMA 平滑

    # === 方案1+5+6：预测误差反馈式优化 ===
    # 方案1：近期准确率/偏差追踪（滚动保留最近3个fold的预测结果）
    recent_fold_results = deque(maxlen=3)
    # 方案5：增量修正 —— 上个 fold 预测错的样本全局索引，传入下个 fold 训练集加权
    correction_indices = set()
    n_correction_total = 0  # 累计加权修正的样本数（诊断用）

    for start in range(W, n - S + 1, S):
        end = min(start + S, n)

        # === 优化1：自适应窗口 ===
        # 当近期 vol_regime 低于训练窗口中位数（震荡市）时，缩短窗口加速旧数据滑出
        if getattr(args, "adaptive_window", False) and vol_regime_global is not None:
            vol_recent = vol_regime_global[max(0, start - 20):start]
            vol_window = vol_regime_global[max(0, start - W):start]
            if len(vol_recent) > 0 and len(vol_window) > 0:
                recent_vol = np.nanmedian(vol_recent)
                window_vol = np.nanmedian(vol_window)
                # 震荡市特征：近期波动率显著低于窗口平均（< 0.8倍）
                if recent_vol < window_vol * 0.8:
                    cur_W = W_min  # 缩短到下限
                else:
                    cur_W = W
            else:
                cur_W = W
        else:
            cur_W = W

        # === 优化E：CUSUM 变点重置 ===
        # 如果训练窗口内存在变点，训练集起点从最近变点开始（丢弃旧 regime 数据）
        # 关键：找到最近一个满足"变点后至少留30天训练数据"的变点
        cusum_reset = False
        if cusum_points and len(cusum_points) > 0:
            window_start = start - cur_W
            val_start = start - V
            # 从最近的变点往前找，直到找到一个留足训练数据的
            best_cp = None
            for cp in reversed(cusum_points):
                if cp <= window_start:
                    break  # 超出窗口范围
                if cp < val_start and (val_start - cp) >= 30:
                    best_cp = cp
                    break  # 找到最近的合格变点
            if best_cp is not None and (start - best_cp) < cur_W:
                # cur_W = start - best_cp 使训练集从 best_cp 开始
                # 训练数据量 = cur_W - V = start - best_cp - V
                new_cur_W = start - best_cp
                if new_cur_W - V >= 30:  # 确保训练数据 >= 30
                    cur_W = new_cur_W
                    cusum_reset = True

        x_tr = features.iloc[start - cur_W:start - V]
        y_tr = y_all.iloc[start - cur_W:start - V]
        x_val = features.iloc[start - V:start]
        y_val = y_all.iloc[start - V:start]
        x_te = features.iloc[start:end]
        y_te = y_all.iloc[start:end]
        if len(x_tr) < 30:
            continue

        # === 优化C：时间衰减样本权重（指数加权）===
        # 窗口内越靠近预测日期的样本权重越高，有效缩短窗口但不完全丢弃历史
        # weight(t) = exp(-λ × (T_now - t))，λ 控制衰减速度
        sw_tr = None
        if sample_weights_all is not None:
            base_sw = sample_weights_all[start - cur_W:start - V].copy()
            # 时间衰减：最新的样本权重为1.0，最老的样本衰减到~0.3
            decay = np.exp(-2.0 * np.arange(len(base_sw))[::-1] / max(len(base_sw), 1))
            decay = decay / decay.max()  # 归一化到 [0, 1]
            sw_tr = base_sw * decay

        # === 方案5：增量修正 ===
        # 上个 fold 预测错的样本如果落在当前训练窗口内，给予 3x 权重
        # 逻辑：让模型重点学习"最近犯过的错误"，等效在线增量学习
        # 时间因果性正确：上个 fold 的测试区间 ⊂ 当前 fold 的训练窗口
        if correction_indices and sw_tr is not None:
            tr_start = start - cur_W
            for gi in correction_indices:
                local_i = gi - tr_start
                if 0 <= local_i < len(sw_tr):
                    sw_tr[local_i] *= 3.0
                    n_correction_total += 1

        # === 主模型训练（长窗口 W）===
        model = create_xgb_model(args)
        model.fit(x_tr, y_tr, sample_weight=sw_tr, eval_set=[(x_val, y_val)], verbose=False)

        # === 优化B：短窗口辅助模型（W_short = max(60, cur_W - 30)）===
        W_short = max(60, cur_W - 30)
        proba_s = None
        w_short_final = 0.0
        if W_short < cur_W and (start - W_short) >= 0:
            x_tr_s = features.iloc[start - W_short:start - V]
            y_tr_s = y_all.iloc[start - W_short:start - V]
            # 训练数据必须同时含两类，否则跳过短模型
            if y_tr_s.nunique() < 2 or len(x_tr_s) < 20:
                proba_s = None
            else:
                sw_s = None
                if sw_tr is not None:
                    sw_s = sw_tr[-len(x_tr_s):]  # 取对应的尾部权重
                model_s = create_xgb_model(args)
                model_s.fit(x_tr_s, y_tr_s, sample_weight=sw_s,
                            eval_set=[(x_val, y_val)], verbose=False)

                # 两个模型的概率
                if not args.no_calibrator:
                    cal_s, bundle_s = fit_calibrator(model_s, x_tr_s, y_tr_s, x_val, y_val, features)
                    proba_s = calibrate_proba(model_s, (cal_s, bundle_s), x_te, features)
                else:
                    proba_s = model_s.predict_proba(x_te)

                # === 动态融合权重 ===
                # 震荡市（低 vol + 弱 trend_resonance）加重短窗口；趋势市加重长窗口
                w_short_final = 0.5  # 默认均衡
                if vol_regime_global is not None and "trend_resonance" in features.columns:
                    vol_now = np.nanmedian(vol_regime_global[max(0, start - 20):start])
                    tr_now = features["trend_resonance"].iloc[start - 1] if start > 0 else 0
                    vol_median = np.nanmedian(vol_regime_global[:start])
                    if vol_now < vol_median * 0.8 and abs(tr_now) < 0.5:
                        w_short_final = 0.7  # 震荡市：短窗口主导
                    elif vol_now > vol_median * 1.2 and abs(tr_now) > 0.8:
                        w_short_final = 0.3  # 趋势市：长窗口主导
        else:
            proba_s = None
            w_short_final = 0.0

        # === 校准主模型概率 ===
        if not args.no_calibrator:
            calibrator, calib_bundle = fit_calibrator(
                model, x_tr, y_tr, x_val, y_val, features
            )
            proba = calibrate_proba(model, (calibrator, calib_bundle), x_te, features)
        else:
            proba = model.predict_proba(x_te)

        # === B: 概率融合 ===
        if proba_s is not None:
            proba[:, 1] = (1 - w_short_final) * proba[:, 1] + w_short_final * proba_s[:, 1]
            proba[:, 0] = 1 - proba[:, 1]

        # === 优化D：换手率平滑（EMA）===
        # 相邻 fold 的概率做指数加权平均，抑制极端翻转
        # α 随 horizon 调整：5d→0.7（快速响应），10d→0.6（更长记忆）
        if prev_proba_last is not None and len(proba) > 0:
            alpha_ema = 0.7 if args.horizon <= 5 else 0.6
            # 用上一个 fold 最后一个样本的概率作为本 fold 起始锚点
            proba[:, 1] = alpha_ema * proba[:, 1] + (1 - alpha_ema) * prev_proba_last
            proba[:, 0] = 1 - proba[:, 1]
        if len(proba) > 0:
            prev_proba_last = proba[-1, 1]  # 更新锚点

        # === 方案1+6：实时准确率追踪 + 滑动窗口偏差检测/修正 ===
        # 用最近3个fold的预测-实际对比，检测模型是否系统性偏向某方向
        # 若偏差持续超过阈值 → 对当前fold概率做反向修正（时间因果性正确：只用过去结果修正当前预测）
        recent_acc = float("nan")
        recent_bias = 0.0
        bias_alert = False
        correction_applied = 0.0

        if len(recent_fold_results) >= 2:
            all_rp = np.concatenate([r["pred"] for r in recent_fold_results])
            all_rt = np.concatenate([r["true"] for r in recent_fold_results])
            recent_acc = float(np.mean(all_rp == all_rt))
            recent_pred_down = float(np.mean(all_rp == 1))
            recent_true_down = float(np.mean(all_rt == 1))
            recent_bias = recent_pred_down - recent_true_down

            # 方案6：偏差修正 —— 模型持续偏向某方向时反向偏移概率
            # 触发条件：|偏差| > 10%（预测下行占比与真实下行占比的差异）
            # 温和修正：只在偏差>15% 且 准确率低时才触发，修正幅度收紧到±0.05
            if abs(recent_bias) > 0.15:
                if recent_acc < 0.45:
                    coef = 0.25  # 准确率很低 → 稍激进
                elif recent_acc < 0.52:
                    coef = 0.15  # 准确率勉强 → 轻微修正
                else:
                    coef = 0.0   # 准确率正常 → 不修正
                correction = float(np.clip(-recent_bias * coef, -0.05, 0.05))
                if abs(correction) > 0.001:
                    proba[:, 1] = np.clip(proba[:, 1] + correction, 0.01, 0.99)
                    proba[:, 0] = 1.0 - proba[:, 1]
                    correction_applied = correction
                    bias_alert = True

        # 预测类别基于校准后的概率（>0.5 判下行=1）
        yp = (proba[:, 1] > 0.5).astype(int)

        # 收集 fold 指标
        from sklearn.metrics import accuracy_score as _acc, f1_score as _f1, roc_auc_score as _auc
        try:
            fold_auc = _auc(y_te.values, proba[:, 1])
        except ValueError:
            fold_auc = float("nan")
        fold_metrics.append({
            "fold": fold + 1,
            "date_start": str(merged["Unnamed: 0"].iloc[start].date()),
            "date_end": str(merged["Unnamed: 0"].iloc[end - 1].date()),
            "n_test": len(y_te),
            "accuracy": _acc(y_te.values, yp),
            "macro_f1": _f1(y_te.values, yp, average="macro"),
            "auc": fold_auc,
            "best_iteration": getattr(model, "best_iteration", -1),
            "cur_W": cur_W,
            "cusum_reset": int(cusum_reset),
            # 方案1+6：反馈追踪指标
            "recent_acc": recent_acc,
            "recent_bias": round(recent_bias, 4),
            "bias_alert": int(bias_alert),
            "correction_applied": round(correction_applied, 4),
        })

        # 收集特征重要性
        imp_series = pd.Series(model.feature_importances_, index=features.columns)
        nonzero_imp = imp_series[imp_series > 0].sort_values(ascending=False)
        fold_feature_imps.append(nonzero_imp.to_dict())

        all_preds.extend(yp.tolist())
        all_probas.extend(proba.tolist())
        all_true.extend(y_te.values.tolist())
        all_dates.extend(merged["Unnamed: 0"].iloc[start:end].tolist())
        # 收集 meta（含波动率特征，供置信度门控评估使用）
        meta_slice = merged.iloc[start:end][["Unnamed: 0", "rate", f"target_date_{args.horizon}d"]].copy()
        for vc in ["vol_regime", "realized_vol_20", "dev_zscore_120"]:
            if vc in features.columns:
                meta_slice[vc] = features.iloc[start:end][vc].values
        all_meta.append(meta_slice)

        # === 方案1：更新近期结果缓冲（供下个fold的偏差检测使用）===
        recent_fold_results.append({"pred": yp.copy(), "true": y_te.values.copy()})

        # === 方案5：提取当前fold预测错的样本索引（传给下个fold加权）===
        wrong_mask = yp != y_te.values
        wrong_global_idx = [start + i for i in range(len(yp)) if wrong_mask[i]]
        # 只保留最近 horizon 天的错样本，避免下个fold训练集被过多高权重样本干扰
        correction_indices = set(wrong_global_idx[-args.horizon:])

        fold += 1
        if fold % 5 == 0:
            print(f"  fold {fold}...", flush=True)

    print(f"  完成: {fold} folds, {len(all_preds)} 预测", flush=True)
    if n_correction_total > 0:
        print(f"  方案5增量修正: 累计加权 {n_correction_total} 个错样本", flush=True)
    n_bias_alerts = sum(1 for fm in fold_metrics if fm.get("bias_alert", 0))
    if n_bias_alerts > 0:
        print(f"  方案6偏差修正: {n_bias_alerts}/{fold} folds 触发修正", flush=True)
    return all_preds, all_probas, all_true, all_dates, all_meta, fold, fold_metrics, fold_feature_imps


# =====================================================================
# 主流程
# =====================================================================

def main():
    global _ARGS_CACHE
    args = parse_args()
    _ARGS_CACHE = args  # 让 build_features 能读到 STL/no-stl 等配置
    START_DATE = args.start_date
    W, S, V = args.rolling_window, args.rolling_step, args.rolling_val_size
    OUTPUT_DIR = BASE / "model_new" / "data"

    # === 1. 加载数据 ===
    print("[1/5] 加载数据...", flush=True)
    df_data = pd.read_csv(OUTPUT_DIR / "data.csv")
    df_rate = pd.read_csv(OUTPUT_DIR / "rate.csv")
    df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"])
    df_rate["time"] = pd.to_datetime(df_rate["time"])
    merged = pd.merge(df_data, df_rate, left_on="Unnamed: 0", right_on="time", how="inner").sort_values("Unnamed: 0")
    H = args.horizon  # 预测期限
    merged[f"target_date_{H}d"] = merged["Unnamed: 0"].shift(-H)
    merged[f"rate_delta_{H}d"] = merged["rate"].shift(-H) - merged["rate"]

    # === 标签生成（优化2：超趋势方向标签可选）===
    if args.detrend_label:
        # 超趋势标签：先在 rate 上做 STL 提取趋势，标签 = 实际变动 - 趋势预期变动 < 0
        print(f"  启用超趋势方向标签（detrend-label, horizon={H}d）", flush=True)
        stl_temp = STL(merged["rate"].fillna(merged["rate"].mean()),
                       period=args.stl_period, robust=True).fit()
        trend_now = pd.Series(stl_temp.trend.values, index=merged.index)
        # 趋势预期 H 日变动 = 趋势斜率 × H（用 H 日趋势差分近似）
        trend_expected = trend_now.shift(-H) - trend_now
        excess_move = merged[f"rate_delta_{H}d"] - trend_expected
        merged["label"] = excess_move.apply(gen_label)
    else:
        merged["label"] = merged[f"rate_delta_{H}d"].apply(gen_label)

    merged = merged.dropna(subset=["label"]).copy()
    merged["label"] = merged["label"].astype(int)
    y_all = merged["label"].copy()

    # === 优化3：震荡市标签死区降权 ===
    # |rate_delta| < dead_zone bp 的样本信噪比极低，训练时降权到 0.3
    # 死区默认按 horizon 线性放大：5d→2bp, 10d→4bp
    dead_zone_bp = args.label_dead_zone if args.label_dead_zone != 2.0 else 2.0 * (H / 5)
    if dead_zone_bp > 0:
        delta_bp = (merged[f"rate_delta_{H}d"].abs() * 100)  # 转为 bp
        # 死区内样本权重 0.3，死区外权重 1.0
        sample_weights_all = np.where(delta_bp < dead_zone_bp, 0.3, 1.0)
        n_dead = int((delta_bp < dead_zone_bp).sum())
        print(f"  标签死区(±{dead_zone_bp}bp): {n_dead}/{len(merged)} ({n_dead/len(merged):.1%}) 样本降权到0.3",
              flush=True)
    else:
        sample_weights_all = np.ones(len(merged))

    # === 2. 构建特征 ===
    print("[2/5] 构建特征...", flush=True)
    if args.load_features:
        # 复用预存特征（绕过 build_features 的 STL 内存瓶颈）
        feat_path = OUTPUT_DIR / args.load_features
        features = pd.read_csv(feat_path, index_col=0)
        print(f"  加载预存特征: {features.shape[1]} 列 <- {args.load_features}", flush=True)
    else:
        features = build_features(merged)

    # 合并遗传特征（预存特征已包含遗传特征，跳过避免列名冲突）
    if not args.load_features:
        gp = load_genetic_features(args)
        if gp is not None:
            features = features.join(gp, how="left")
            gp_cols = [c for c in gp.columns if c in features.columns]
            features[gp_cols] = features[gp_cols].fillna(0.0)
    else:
        print(f"  遗传特征: 已包含在预存特征中，跳过合并", flush=True)

    # 特征筛选（配合 04_特征筛选.py 做对比实验）
    if args.keep_cols_file:
        keep_path = BASE / args.keep_cols_file
        keep_cols = [c.strip() for c in keep_path.read_text(encoding="utf-8").splitlines() if c.strip()]
        matched = [c for c in keep_cols if c in features.columns]
        dropped = [c for c in features.columns if c not in set(keep_cols)]
        features = features[matched]
        print(f"  特征筛选: 保留 {len(matched)} 列, 删除 {len(dropped)} 列 <- {keep_path.name}",
              flush=True)
        if len(dropped) > 0 and len(matched) == 0:
            raise ValueError("保留列匹配为0，请检查 keep-cols-file 列名与特征是否一致")

    # 过滤起始日期
    cutoff = pd.to_datetime(START_DATE)
    mask = pd.to_datetime(merged["Unnamed: 0"]) >= cutoff
    y_all = y_all.loc[mask.values].reset_index(drop=True)
    merged = merged.loc[mask].reset_index(drop=True)
    sample_weights_all = sample_weights_all[mask.values]  # 同步过滤
    if args.load_features:
        # 预存特征已在保存时过滤过，只需确保行数对齐
        if len(features) > len(merged):
            features = features.iloc[:len(merged)].reset_index(drop=True)
    else:
        features = features.loc[mask.values].reset_index(drop=True)
    print(f"  样本: {len(features)}, 特征数: {features.shape[1]}", flush=True)

    # === 3. 滚动训练 ===
    print("[3/5] 滚动窗口训练...", flush=True)
    results = rolling_window_train(features, y_all, merged, args, sample_weights_all)
    all_preds, all_probas, all_true, all_dates, all_meta, fold, fold_metrics, fold_imps = results

    # === 4. 评估 ===
    print("[4/5] 评估...", flush=True)
    y_pred = np.array(all_preds)
    y_true = np.array(all_true)
    y_proba = np.array(all_probas)
    test_meta = pd.concat(all_meta, ignore_index=True)

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    f1_pos = f1_score(y_true, y_pred, pos_label=1)
    f1_neg = f1_score(y_true, y_pred, pos_label=0)
    try:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    maj = int(pd.Series(y_true).mode().iloc[0])
    maj_acc = float(np.mean(y_true == maj))

    # === P2 置信度门控评估（中信建投同款手法）===
    # 当预测概率落入模糊带（|prob-0.5|<margin）且波动率体制偏高时，标记为"弃权"
    # 这些样本信噪比极低，模型无判别力，过滤后剩余样本的准确率更具参考价值
    gating_lines = []
    if not args.no_gating and "vol_regime" in test_meta.columns:
        prob_margin = args.gating_prob_margin if args.gating_prob_margin != 0.15 else (0.15 if args.horizon <= 5 else 0.10)
        vol_q = args.gating_vol_quantile
        prob1 = y_proba[:, 1]
        vol_regime = test_meta["vol_regime"].values
        vol_threshold = np.nanquantile(vol_regime, vol_q)

        # 模糊带 = 概率接近 0.5（不确定）且 高波动体制（市场混乱、信号不可靠）
        is_fuzzy = np.abs(prob1 - 0.5) < prob_margin
        is_high_vol = vol_regime > vol_threshold
        is_abstain = is_fuzzy & is_high_vol  # 双条件满足才弃权
        is_confident = ~is_abstain

        n_abstain = int(is_abstain.sum())
        coverage = 1.0 - n_abstain / len(y_true)  # 置信样本覆盖率

        if is_confident.sum() > 0:
            acc_gated = accuracy_score(y_true[is_confident], y_pred[is_confident])
            mf1_gated = f1_score(y_true[is_confident], y_pred[is_confident], average="macro")
            try:
                auc_gated = roc_auc_score(y_true[is_confident], y_proba[is_confident, 1])
            except ValueError:
                auc_gated = float("nan")
        else:
            acc_gated = mf1_gated = auc_gated = float("nan")

        gating_lines = [
            "",
            "=" * 50,
            "=== 置信度门控评估（中信建投同款手法）===",
            f"门控参数: prob_margin={prob_margin}, vol_quantile={vol_q} (阈值={vol_threshold:.3f})",
            f"弃权样本: {n_abstain} / {len(y_true)} ({1-coverage:.1%})",
            f"置信覆盖率: {coverage:.1%}",
            "",
            f"{'指标':<16} {'全量':>10} {'门控后':>10} {'提升':>10}",
            f"{'-'*50}",
            f"{'Accuracy':<16} {acc:>10.4f} {acc_gated:>10.4f} {acc_gated-acc:>+10.4f}",
            f"{'Macro F1':<16} {mf1:>10.4f} {mf1_gated:>10.4f} {mf1_gated-mf1:>+10.4f}",
            f"{'AUC':<16} {auc:>10.4f} {auc_gated:>10.4f} {auc_gated-auc:>+10.4f}",
        ]
        print("\n".join(gating_lines), flush=True)

    # === 5. 训练最终模型 + 保存 ===
    print("[5/5] 保存结果...", flush=True)
    # 输出文件后缀：默认 _bin，horizon!=5 时用 _{H}d
    SUF = args.suffix if args.suffix else ("_bin" if H == 5 else f"_{H}d")
    n = len(features)
    fm = create_xgb_model(args)
    # 最终模型用更大的训练窗口（至少150天），避免 early stopping 立即停
    W_final = max(W, 180)
    V_final = min(V, 30)
    fs, fv = max(0, n - W_final), max(0, n - V_final)
    fm.fit(features.iloc[fs:fv], y_all.iloc[fs:fv],
           sample_weight=sample_weights_all[fs:fv] if sample_weights_all is not None else None,
           eval_set=[(features.iloc[fv:n], y_all.iloc[fv:n])], verbose=False)
    joblib.dump(fm, OUTPUT_DIR / f"xgb_model{SUF}.joblib")

    # 保存特征和标签
    features.to_csv(OUTPUT_DIR / f"xgb_train_features{SUF}.csv")
    pd.DataFrame({"label": y_all}).to_csv(OUTPUT_DIR / f"xgb_labels{SUF}.csv", index=False)

    # 保存预测明细
    LABEL_MAP = {0: "上行(偏空)", 1: "下行(偏多)"}
    pred_df = pd.DataFrame({
        "date": pd.to_datetime(all_dates).astype(str),
        "y_true": y_true, "y_pred": y_pred,
        "prob_0_上行": y_proba[:, 0], "prob_1_下行": y_proba[:, 1],
    })
    pred_df.to_csv(OUTPUT_DIR / f"xgb_test_predictions{SUF}.csv", index=False)

    # 可视化友好的预测表
    view_df = pd.DataFrame({
        "判断基点": pd.to_datetime(test_meta["Unnamed: 0"]).dt.strftime("%Y-%m-%d"),
        "当期10Y": test_meta["rate"].round(4),
        "AI研判": [LABEL_MAP[int(p)] for p in y_pred],
        "预测时点": pd.to_datetime(test_meta[f"target_date_{H}d"]).dt.strftime("%Y-%m-%d"),
        "上行(偏空)概率": [f"{v:.2%}" for v in y_proba[:, 0]],
        "下行(偏多)概率": [f"{v:.2%}" for v in y_proba[:, 1]],
    })
    view_df.to_csv(OUTPUT_DIR / f"xgb_test_predictions_view{SUF}.csv", index=False, encoding="utf-8-sig")
    view_df.tail(1).to_csv(OUTPUT_DIR / f"xgb_latest_prediction_view{SUF}.csv", index=False, encoding="utf-8-sig")

    # 特征重要性
    imp = pd.DataFrame({"feature": features.columns, "importance": fm.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(OUTPUT_DIR / f"xgb_feature_importance{SUF}.csv", index=False)

    # === 逐 fold 分析 + 特征稳定性 ===
    fold_df = pd.DataFrame(fold_metrics)

    # 特征稳定性统计
    feature_stats = {}
    for imp_dict in fold_imps:
        for rank, (feat, val) in enumerate(sorted(imp_dict.items(), key=lambda x: -x[1])):
            if feat not in feature_stats:
                feature_stats[feat] = {"appear_count": 0, "imp_sum": 0.0, "rank_sum": 0}
            feature_stats[feat]["appear_count"] += 1
            feature_stats[feat]["imp_sum"] += val
            feature_stats[feat]["rank_sum"] += (rank + 1)

    if feature_stats:
        stability_df = pd.DataFrame([
            {"feature": feat, "appear_in_top30": s["appear_count"],
             "appear_ratio": s["appear_count"] / max(len(fold_imps), 1),
             "avg_importance": s["imp_sum"] / max(s["appear_count"], 1),
             "avg_rank": s["rank_sum"] / max(s["appear_count"], 1)}
            for feat, s in feature_stats.items()
        ]).sort_values("appear_ratio", ascending=False)
    else:
        # 强正则化下可能出现所有特征 importance=0 的情况
        stability_df = pd.DataFrame(
            columns=["feature", "appear_in_top30", "appear_ratio", "avg_importance", "avg_rank"]
        )

    # 从未出现的特征
    all_features_set = set(features.columns)
    appeared = set(feature_stats.keys())
    never_nonzero = sorted(all_features_set - appeared)

    fold_df.to_csv(OUTPUT_DIR / f"xgb_fold_metrics{SUF}.csv", index=False)
    stability_df.to_csv(OUTPUT_DIR / f"xgb_feature_stability{SUF}.csv", index=False)
    pd.DataFrame({"never_nonzero_feature": never_nonzero}).to_csv(
        OUTPUT_DIR / f"xgb_never_nonzero{SUF}.csv", index=False)

    # === 指标汇总 ===
    metrics_text = "\n".join([
        f"XGBoost 二分类测试结果（horizon={H}d）",
        f"训练模式: rolling (window={W}, step={S}, val={V})",
        f"训练 folds: {fold}",
        f"数据起始: {START_DATE}",
        f"测试样本数: {len(y_true)}",
        f"总特征数: {features.shape[1]}",
        f"超参: depth={args.max_depth}, n={args.n_estimators}, lr={args.learning_rate}, "
        f"mcw={args.min_child_weight}, ra={args.reg_alpha}",
        "",
        "=== 核心指标 ===",
        f"Accuracy:       {acc:.4f}  (基线: {maj_acc:.4f})",
        f"Macro F1:       {mf1:.4f}",
        f"F1 (下行=1):    {f1_pos:.4f}",
        f"F1 (上行=0):    {f1_neg:.4f}",
        f"AUC:            {auc:.4f}",
        "",
        "=== 混淆矩阵 ===",
        pd.DataFrame(cm, index=["true_0_上行", "true_1_下行"],
                     columns=["pred_0_上行", "pred_1_下行"]).to_string(),
        "",
        "=== Classification Report ===",
        classification_report(y_true, y_pred, target_names=["上行(偏空)", "下行(偏多)"], digits=4),
        "",
        "=== Top 20 重要特征 ===",
        imp.head(20).to_string(index=False),
        "",
        "=" * 70,
        "=== 逐 fold 指标 ===",
        fold_df[["fold", "date_start", "date_end", "accuracy", "macro_f1", "auc", "best_iteration"]].to_string(index=False),
        "",
        "=" * 70,
        "=== 特征稳定性 Top 30 ===",
        stability_df.head(30).to_string(index=False),
        "",
        f"--- 从未出现 importance>0 的特征 ({len(never_nonzero)} 个) ---",
        "\n".join(f"  {f}" for f in never_nonzero),
        *gating_lines,  # 置信度门控评估结果（P2）
    ])
    (OUTPUT_DIR / f"xgb_metrics{SUF}.txt").write_text(metrics_text, encoding="utf-8")

    # 控制台输出
    print(f"\n{'='*50}")
    print(f"Accuracy: {acc:.4f}  (基线: {maj_acc:.4f})")
    print(f"Macro F1: {mf1:.4f}  AUC: {auc:.4f}")
    print(f"F1(下行): {f1_pos:.4f}  F1(上行): {f1_neg:.4f}")
    print(f"特征数: {features.shape[1]}, folds: {fold}")
    print(f"{'='*50}")
    print("\nTop 10 重要特征:")
    print(imp.head(10).to_string(index=False))
    print("\nDONE")


if __name__ == "__main__":
    main()
