# -*- coding: utf-8 -*-
"""
10年期国债收益率未来5天方向预测 —— 二分类 XGBoost 滚动训练

与三分类版本的核心区别：
  1. 标签：下行=1 / 上行=0（去掉 ±3BP 阈值，只判断方向）
  2. 模型：XGBClassifier objective="binary:logistic"
  3. 评估：Accuracy / F1 / AUC / 2x2 混淆矩阵

  特征工程与三分类版本完全一致（利率时序、信用利差、美元/汇率、
  股债相对回报率、加杠杆情绪、利率相对强度、遗传特征去重）。

  训练方式：滚动窗口（rolling window），窗口=756天，步长=63天，验证=63天。
  不使用时间衰减权重（前序实验证明负面）。

输出文件（全部加 _bin 后缀）：
  data/43_xgb_metrics_bin.txt              -> 指标汇总
  data/43_xgb_test_predictions_bin.csv     -> 测试集预测明细
  data/43_xgb_test_predictions_view_bin.csv -> 可视化友好的预测表
  data/43_xgb_model_bin.joblib             -> 最终模型
  data/43_xgb_feature_importance_bin.csv   -> 特征重要性
  data/43_xgb_train_features_bin.csv       -> 训练特征矩阵
  data/43_xgb_labels_bin.csv               -> 标签
  data/43_xgb_latest_prediction_view_bin.csv -> 最新一期预测
"""

import sys
import os
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

os.chdir(Path(__file__).resolve().parent)
BASE = Path(".").resolve()


# =====================================================================
# CLI 参数
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="二分类 XGBoost 滚动训练")
    parser.add_argument("--start-date", default="2018-01-01", help="训练数据起始日期")
    parser.add_argument("--rolling-window", type=int, default=180, help="滚动窗口大小（交易日）")
    parser.add_argument("--rolling-step", type=int, default=63, help="滚动步长（交易日）")
    parser.add_argument("--rolling-val-size", type=int, default=63, help="验证集大小（交易日）")
    parser.add_argument("--n-estimators", type=int, default=600, help="XGBoost 树数量")
    parser.add_argument("--max-depth", type=int, default=3, help="最大深度")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="学习率（调参最优值）")
    parser.add_argument("--min-child-weight", type=float, default=8.0, help="叶子节点最小权重")
    parser.add_argument("--reg-alpha", type=float, default=1.0, help="L1 正则化（调参最优值）")
    parser.add_argument("--genetic-features-file", default="data/43_x_bin.csv",
                        help="遗传特征文件（二分类版本）；若不存在则尝试 data/43_x.csv")
    parser.add_argument("--no-genetic", action="store_true", help="不加载遗传特征")
    return parser.parse_args()


# =====================================================================
# 二分类标签
# =====================================================================
def gen_label(v):
    """
    二分类标签：下行=1, 上行=0。
    不设幅度阈值，只看方向，避免单边市场下的标签噪音。
    """
    if pd.isna(v):
        return np.nan
    return 1 if v < 0 else 0


# =====================================================================
# 特征工程（与三分类版本完全一致）
# =====================================================================
def _add_technical_momentum_features(features: pd.DataFrame, price_col: str) -> None:
    """
    基于价格序列构建技术面动量特征，直接原地修改 features。

    包含的特征族：
    1. RSI（相对强弱指数）：衡量超买/超卖，周期 6/14/24
    2. 布林带突破：价格偏离 20 日均值 ±2σ 的位置与突破信号
    3. 连涨/连跌天数：连续方向计数
    4. MACD：12/26 EMD 差离值 + 9 日信号线
    5. ROC（价格动量/变化率）
    6. 威廉指标 %R：衡量收盘价在近期高低区间的位置

    所有计算只使用 shift / rolling，严格无未来信息泄露。

    注意：这里"上涨"指 rate 数值上升（利率上行，债市偏空）。
    """
    price = features[price_col]

    # ========== 1. RSI（相对强弱指数）==========
    # 公式：RS = N 日平均涨幅 / N 日平均跌幅；RSI = 100 - 100/(1+RS)
    # 用 EMA 平滑（Wilder 法）或简单平均均可，这里用简单滚动平均
    delta = price.diff()
    gain = delta.clip(lower=0)   # 上涨部分
    loss = (-delta).clip(lower=0)  # 下跌部分（取正值）
    for period in [6, 14, 24]:
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        # 防止除零
        rs = avg_gain / (avg_loss + 1e-10)
        features[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    # ========== 2. 布林带突破 ==========
    # 中轨 = 20 日均值，上轨 = +2σ，下轨 = -2σ
    bb_window = 20
    bb_mean = price.rolling(window=bb_window, min_periods=bb_window).mean()
    bb_std = price.rolling(window=bb_window, min_periods=bb_window).std()
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    # %B：价格在布林带中的相对位置（0=下轨，1=上轨，>1 突破上轨，<0 突破下轨）
    features["bb_pctb_20"] = (price - bb_lower) / (bb_upper - bb_lower + 1e-10)
    # 带宽：衡量波动率压缩/扩张
    features["bb_width_20"] = (bb_upper - bb_lower) / (bb_mean + 1e-10)
    # 突破信号：1=突破上轨，-1=突破下轨，0=带内
    features["bb_break_upper_20"] = (price > bb_upper).astype(int)
    features["bb_break_lower_20"] = (price < bb_lower).astype(int)

    # ========== 3. 连涨/连跌天数 ==========
    # 用 cumsum trick：方向改变时重置计数器
    direction = np.sign(delta.fillna(0))  # +1 涨, -1 跌, 0 平
    # 连续上涨天数
    up = (direction > 0).astype(int)
    down = (direction < 0).astype(int)
    # 用分组累计：当方向变化时开启新组
    up_streak = up.copy()
    down_streak = down.copy()
    for i in range(1, len(up_streak)):
        if up.iloc[i] == 1 and up.iloc[i - 1] == 1:
            up_streak.iloc[i] = up_streak.iloc[i - 1] + 1
        if down.iloc[i] == 1 and down.iloc[i - 1] == 1:
            down_streak.iloc[i] = down_streak.iloc[i - 1] + 1
    features["consec_up_days"] = up_streak   # 连涨天数
    features["consec_down_days"] = down_streak  # 连跌天数
    # 净连涨（正=连涨，负=连跌）
    features["net_streak"] = up_streak - down_streak

    # ========== 4. MACD ==========
    # MACD = EMA12 - EMA26，信号线 = MACD 的 9 日 EMA
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    features["macd"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd_line - macd_signal  # 柱状图（动能）

    # ========== 5. ROC（变化率 Rate of Change）==========
    # ROC = (当前价 - N 日前价) / N 日前价
    for period in [5, 10, 20]:
        shifted = price.shift(period)
        features[f"roc_{period}"] = (price - shifted) / (shifted.abs() + 1e-10)

    # ========== 6. 威廉指标 %R ==========
    # %R = (最高价 - 收盘价) / (最高价 - 最低价) * (-100)
    # 值域 [-100, 0]，越接近 0 越超买，越接近 -100 越超卖
    for period in [14, 28]:
        roll_high = price.rolling(window=period, min_periods=period).max()
        roll_low = price.rolling(window=period, min_periods=period).min()
        features[f"williams_r_{period}"] = (roll_high - price) / (roll_high - roll_low + 1e-10) * (-100)

    # ====================================================================
    # 第1类：滚动时序形态统计（压缩走势形状为截面值）
    # 目的：让 XGBoost 间接获得"形态识别"能力
    # ====================================================================

    # ========== 7. 滚动 AR(1) 自回归系数 ==========
    # AR(1) = corr(x_t, x_{t-1})，衡量动量持续性
    # 高值（接近1）= 趋势市，低值（接近0或负）= 震荡市/均值回复
    price_diff = price.diff()
    for period in [10, 20, 60]:
        shifted = price.shift(1)
        # 滚动计算 (x, x_{t-1}) 的相关系数
        features[f"ar1_{period}"] = price.rolling(period).corr(shifted)

    # ========== 8. 滚动线性趋势斜率 + R² 拟合度 ==========
    # 用过去 N 天做 OLS 回归 y = a*t + b，提取斜率和 R²
    # 斜率：趋势方向和强度；R²：趋势的稳定性/线性度
    for period in [20, 60]:
        # t = [0, 1, ..., period-1]，对每个窗口做回归
        t_vals = np.arange(period, dtype=np.float64)
        t_mean = t_vals.mean()
        t_var = ((t_vals - t_mean) ** 2).sum()

        slopes = []
        r2s = []
        values = price.values
        for i in range(len(price)):
            if i < period - 1:
                slopes.append(np.nan)
                r2s.append(np.nan)
                continue
            window = values[i - period + 1: i + 1]
            y_mean = window.mean()
            # 斜率 = sum((t-t_mean)(y-y_mean)) / sum((t-t_mean)^2)
            slope = ((t_vals - t_mean) * (window - y_mean)).sum() / t_var
            # R² = 1 - SS_res / SS_tot
            y_pred = slope * (t_vals - t_mean) + y_mean
            ss_res = ((window - y_pred) ** 2).sum()
            ss_tot = ((window - y_mean) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            slopes.append(slope)
            r2s.append(r2)
        features[f"trend_slope_{period}"] = slopes
        features[f"trend_r2_{period}"] = r2s

    # ========== 9. 滚动转折点数 ==========
    # 方向反转次数：低=强趋势，高=无序震荡
    sign = np.sign(price_diff.fillna(0).values)
    for period in [10, 20, 60]:
        reversals = []
        count = 0
        prev_sign = 0
        for i in range(len(sign)):
            s = sign[i]
            # 检测方向变化（忽略0）
            if s != 0 and prev_sign != 0 and s != prev_sign:
                count += 1
            if s != 0:
                prev_sign = s
            reversals.append(count)
        # 转换为滚动窗口内的反转次数：用差分
        rev_series = pd.Series(reversals)
        features[f"reversal_count_{period}"] = rev_series.diff(period).fillna(rev_series)

    # ========== 10. 滚动 Hurst 指数 ==========
    # Hurst > 0.5：趋势延续（长期记忆性）
    # Hurst < 0.5：均值回复
    # Hurst ≈ 0.5：随机游走
    # 用 R/S 法（Rescaled Range）估计
    for period in [60, 120]:
        hurst_vals = []
        values = price.values
        for i in range(len(price)):
            if i < period - 1:
                hurst_vals.append(np.nan)
                continue
            ts = values[i - period + 1: i + 1]
            features_h = _hurst_rs(ts)
            hurst_vals.append(features_h)
        features[f"hurst_{period}"] = hurst_vals

    # ========== 11. 滚动 FFT 主频能量比 ==========
    # 将过去 N 天做 FFT，取最大频率分量的能量占比
    # 高能量比 = 强周期性，低能量比 = 高噪音
    for period in [60]:
        fft_ratios = []
        values = price.values
        for i in range(len(price)):
            if i < period - 1:
                fft_ratios.append(np.nan)
                continue
            window = values[i - period + 1: i + 1]
            # 去均值后做 FFT
            window_centered = window - window.mean()
            spectrum = np.abs(np.fft.rfft(window_centered))
            total_energy = (spectrum ** 2).sum() + 1e-10
            max_energy = (spectrum ** 2).max()
            fft_ratios.append(max_energy / total_energy)
        features[f"fft_max_ratio_{period}"] = fft_ratios


def _hurst_rs(ts: np.ndarray) -> float:
    """
    用 R/S 法（Rescaled Range）估计 Hurst 指数。

    原理：将序列分成若干子区间，计算每个子区间的 R/S 统计量，
    然后 R/S ~ (子区间长度)^H，用对数回归得到 H。

    参数:
        ts: 一维时间序列（长度 >= 20）
    返回:
        Hurst 指数（0~1），NaN 表示无法计算
    """
    n = len(ts)
    if n < 20:
        return np.nan

    # 选择若干个子区间长度（2 的幂次附近的整数）
    lags = []
    sizes = []
    for size in [n // 2, n // 4, n // 8, n // 16]:
        if size >= 5 and n // size >= 2:
            lags.append(size)
            sizes.append(size)
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
            mean_adjusted = chunk - mean
            cumdev = np.cumsum(mean_adjusted)
            r = cumdev.max() - cumdev.min()  # Range
            s = chunk.std()
            if s > 1e-10:
                rs_list.append(r / s)
        if rs_list:
            rs_values.append((np.log(size), np.log(np.mean(rs_list))))

    if len(rs_values) < 2:
        return np.nan

    # 用最小二乘法拟合 log(R/S) = H * log(size) + c
    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])
    h, _ = np.polyfit(x, y, 1)
    # 截断到合理范围
    return float(np.clip(h, 0.0, 1.0))


def _add_cross_asset_divergence_features(features: pd.DataFrame) -> None:
    """
    第2类：跨资产相对动量 / 背离特征。

    核心思路：单资产看不清方向时，跨资产的相对强弱和相关性突变
    往往是拐点的先行信号。

    包含三组：
    1. 利率-商品背离：商品价格上涨但利率不动（或反向），可能预示滞后补涨
    2. 利率-汇率相关性突变：相关性从正转负 = regime 切换
    3. 股债跷跷板强度：股债相关性的滚动变化
    """
    rate = features["rate"]

    # ========== 1. 利率-商品动量背离 ==========
    # 候选商品/周期资产列（存在则计算）
    commodity_cols = {
        "期货结算价(活跃合约):螺纹钢:前一个观测值": "rebar",
        "中国:水泥价格指数:前一个观测值": "cement",
        "期货结算价(活跃合约):阴极铜:前一个观测值": "copper",
        "南华工业品指数": "nanhua",
    }
    for raw_col, short_name in commodity_cols.items():
        if raw_col not in features.columns:
            continue
        commodity = features[raw_col]
        # 两者各自的 20 日动量
        rate_mom = rate.pct_change(20)
        comm_mom = commodity.pct_change(20)
        # 背离 = 商品动量 - 利率动量（归一化后）
        # 正背离 = 商品涨得比利率快 -> 商品强、利率弱（利率有补涨空间）
        rate_mom_std = rate_mom.rolling(60, min_periods=20).std() + 1e-10
        comm_mom_std = comm_mom.rolling(60, min_periods=20).std() + 1e-10
        features[f"divergence_{short_name}"] = (comm_mom / comm_mom_std) - (rate_mom / rate_mom_std)
        # 滚动相关性（20/60 日）
        features[f"corr_rate_{short_name}_20"] = rate.rolling(20).corr(commodity)
        features[f"corr_rate_{short_name}_60"] = rate.rolling(60).corr(commodity)

    # ========== 2. 利率-汇率相关性突变 ==========
    # 美元/汇率与利率的相关性在 regime 切换时会突变
    fx_candidates = ["即期汇率:美元兑人民币", "美元指数"]
    for fx_col in fx_candidates:
        if fx_col not in features.columns:
            continue
        fx = features[fx_col]
        short = "usdcny" if "人民币" in fx_col else "dollar"
        corr_20 = rate.rolling(20).corr(fx)
        corr_60 = rate.rolling(60).corr(fx)
        features[f"corr_rate_{short}_20"] = corr_20
        features[f"corr_rate_{short}_60"] = corr_60
        # 相关性突变 = 短期相关性 - 长期相关性（正负反转时为强信号）
        features[f"corr_break_{short}"] = corr_20 - corr_60

    # ========== 3. 股债跷跷板强度 ==========
    # 股债相对回报率本身就是股息率/国债收益率
    # 这里计算它的滚动变化作为"跷跷板强度"
    if "股债相对回报率" in features.columns:
        stock_bond = features["股债相对回报率"]
        # 股债相对回报率的变化率 = 跷跷板摆动速度
        features["stock_bond_swing_5"] = stock_bond.pct_change(5)
        features["stock_bond_swing_20"] = stock_bond.pct_change(20)
        # 股债相对回报率 vs 利率的滚动相关性
        # 通常为负（股涨债跌），相关性减弱可能预示拐点
        features["corr_stock_bond_rate_20"] = stock_bond.rolling(20).corr(rate)
        features["corr_stock_bond_rate_60"] = stock_bond.rolling(60).corr(rate)

    # ========== 4. 波动率结构特征 ==========
    # 波动率有聚集性和均值回复性，是独立于方向的预测维度
    _add_volatility_structure_features(features, rate)


def _add_volatility_structure_features(features: pd.DataFrame, rate: pd.Series) -> None:
    """
    第3类：波动率结构特征。

    波动率本身有聚集性和均值回复性，是独立于方向的预测维度。

    包含：
    1. 波动率锥：当前波动率在历史分位数的什么位置
    2. vol of vol：波动率的波动率（波动率自身的变化率）
    3. 已实现波动率 vs 滚动均值的偏离
    """
    # 日度变化（绝对变化，国债用绝对值更合理）
    daily_change = rate.diff().abs()

    # ========== 1. 波动率锥（历史分位数）==========
    # 计算过去 120 天的已实现波动率，然后看当前波动率在其中的百分位
    # 已实现波动率 = 过去 N 天日度变化的标准差
    for short_window in [5, 10, 20]:
        real_vol = daily_change.rolling(short_window, min_periods=short_window).std()
        features[f"realized_vol_{short_window}"] = real_vol
        # 在过去 120 天的分位数
        vol_percentile = real_vol.rolling(120, min_periods=60).rank(pct=True)
        features[f"vol_percentile_{short_window}"] = vol_percentile

    # ========== 2. vol of vol（波动率的波动率）==========
    # 衡量波动率本身是否稳定
    vol_20 = daily_change.rolling(20, min_periods=10).std()
    features["vol_of_vol_60"] = vol_20.rolling(60, min_periods=30).std()
    features["vol_of_vol_120"] = vol_20.rolling(120, min_periods=60).std()

    # ========== 3. 已实现波动率 vs 滚动均值偏离 ==========
    # 波动率均值回复：偏离均值越远，反转概率越高
    for vol_window in [20, 60]:
        vol = daily_change.rolling(vol_window, min_periods=vol_window).std()
        vol_mean = vol.rolling(120, min_periods=60).mean()
        vol_std_long = vol.rolling(120, min_periods=60).std()
        # z-score 形式的偏离
        features[f"vol_deviation_zscore_{vol_window}"] = (vol - vol_mean) / (vol_std_long + 1e-10)
        # 简单比率
        features[f"vol_ratio_{vol_window}"] = vol / (vol_mean + 1e-10)

    # ========== 4. 波动率变化方向（波动率扩张 vs 收缩）==========
    # 波动率扩张往往伴随趋势加速，收缩往往预示突破
    vol_10 = daily_change.rolling(10, min_periods=5).std()
    vol_5 = daily_change.rolling(5, min_periods=3).std()
    features["vol_regime"] = vol_5 / (vol_10 + 1e-10)  # >1 扩张, <1 收缩


def _add_seasonality_features(features: pd.DataFrame) -> None:
    """
    第4类：季节性 / 时间特征。

    国债市场有明确的季节性规律：
    - 季末（3/6/9/12 月末）资金面紧张，利率易上行
    - 年初配置盘集中入场，利率易下行
    - 月末/季末效应

    用 sin/cos 周期编码（而非 one-hot），避免稀疏且能捕捉连续周期性。
    """
    # 从索引提取日期（features 的 "Unnamed: 0" 列是日期）
    dates = pd.to_datetime(features["Unnamed: 0"])

    # ========== 1. 月/季周期编码 ==========
    month = dates.dt.month
    # 月份的 sin/cos 编码（周期=12）
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    # 季度的 sin/cos 编码（周期=4）
    quarter = dates.dt.quarter
    features["quarter_sin"] = np.sin(2 * np.pi * quarter / 4)
    features["quarter_cos"] = np.cos(2 * np.pi * quarter / 4)

    # ========== 2. 距季末天数 ==========
    # 季末月份 = 3, 6, 9, 12
    # 计算当前日期距最近季末月最后一天的天数
    quarter_end_months = [3, 6, 9, 12]
    days_to_quarter_end = []
    for d in dates:
        # 找到当前季度末的月份
        current_q = (d.month - 1) // 3 + 1
        q_end_month = current_q * 3
        # 该月最后一天
        if q_end_month == 12:
            q_end = pd.Timestamp(year=d.year + 1, month=1, day=1) - pd.Timedelta(days=1)
        else:
            q_end = pd.Timestamp(year=d.year, month=q_end_month + 1, day=1) - pd.Timedelta(days=1)
        days_to_quarter_end.append((q_end - d).days)
    features["days_to_quarter_end"] = days_to_quarter_end

    # ========== 3. 距月末天数 ==========
    days_to_month_end = []
    for d in dates:
        if d.month == 12:
            month_end = pd.Timestamp(year=d.year + 1, month=1, day=1) - pd.Timedelta(days=1)
        else:
            month_end = pd.Timestamp(year=d.year, month=d.month + 1, day=1) - pd.Timedelta(days=1)
        days_to_month_end.append((month_end - d).days)
    features["days_to_month_end"] = days_to_month_end

    # ========== 4. 是否季末月份（0/1 标志）==========
    features["is_quarter_end_month"] = dates.dt.month.isin([3, 6, 9, 12]).astype(int)
    features["is_year_start"] = dates.dt.month.isin([1, 2]).astype(int)

    # ========== 5. 年内天数编码 ==========
    day_of_year = dates.dt.dayofyear
    features["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    features["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    # ========== 6. 周内星期编码 ==========
    day_of_week = dates.dt.dayofweek  # 0=Monday
    features["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 5)
    features["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 5)


def build_features(merged: pd.DataFrame) -> pd.DataFrame:
    """
    构建完整特征集，包括：
    - 利率时序（lag/ma/std/change/curve gap）
    - 信用利差衍生（chg/ma/term spread）
    - 美元/汇率衍生（chg/ma/deviation/cross）
    - 股债相对回报率衍生（chg/ma/deviation）
    - 加杠杆情绪衍生（R001/R007 成交量、隔夜回购占比）
    - 利率相对强度（zscore/percentile）
    - 技术面动量（RSI/布林带/连涨跌/MACD/ROC/威廉）
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

    # --- 美元/汇率衍生 ---
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

    # --- 股债相对回报率衍生 ---
    if "股债相对回报率" in features.columns:
        features["股债相对回报率_chg_5d"] = features["股债相对回报率"].diff(5)
        features["股债相对回报率_chg_20d"] = features["股债相对回报率"].diff(20)
        features["股债相对回报率_ma_5"] = features["股债相对回报率"].rolling(5, min_periods=1).mean()
        features["股债相对回报率_ma_20"] = features["股债相对回报率"].rolling(20, min_periods=1).mean()
        features["股债相对回报率_deviation_20"] = features["股债相对回报率"] - features["股债相对回报率_ma_20"]

    # --- 加杠杆情绪衍生 ---
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

    # --- 利率相对强度 ---
    if "rate_std_20" in features.columns:
        features["rate_chg_5d_zscore"] = features["rate"].diff(5) / features["rate_std_20"]
        features["rate_chg_10d_zscore"] = features["rate"].diff(10) / features["rate_std_20"]
    rw = features["rate"].rolling(60, min_periods=60)
    features["rate_percentile_60"] = rw.rank(pct=True)
    features["rate_zscore_60"] = (features["rate"] - rw.mean()) / rw.std()

    # --- 第2类：跨资产相对动量 / 背离特征 ---
    # 单资产看不清方向时，跨资产背离往往是拐点信号
    _add_cross_asset_divergence_features(features)

    # --- 技术面动量特征（基于 rate 序列）---
    # 所有计算严格只用历史信息（shift/rolling），不泄露未来
    _add_technical_momentum_features(features, "rate")

    # --- 第4类：季节性 / 时间特征 ---
    # 国债市场有明确的季节性规律（季末资金面、年初配置盘）
    _add_seasonality_features(features)

    # --- 清理非特征列 ---
    features = features.drop(
        columns=["time", "label", "rate_delta_5d", "date_list"], errors="ignore"
    ).copy()
    features = features.set_index("Unnamed: 0")
    features = features.apply(pd.to_numeric, errors="coerce")
    # 先替换 inf（pct_change 除零等产生），再填充缺失值
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(99.00)
    return features


def load_genetic_features(args, base_dir: Path):
    """
    加载遗传特征（去重后保留 5 个）。
    优先加载二分类版本 43_x_bin.csv，不存在则回退到 43_x.csv。
    """
    if args.no_genetic:
        print("  遗传特征: 已禁用 (--no-genetic)", flush=True)
        return None

    gp_file = base_dir / args.genetic_features_file
    if not gp_file.exists():
        # 回退到三分类版本遗传特征
        gp_file = base_dir / "data/43_x.csv"
    if not gp_file.exists():
        print("  遗传特征: 文件不存在，跳过", flush=True)
        return None

    gp = pd.read_csv(gp_file, index_col=0)
    gp.index = pd.to_datetime(gp.index)
    gp = gp.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # 去重：保留信息量最高的 5 个（与前序实验结论一致）
    keep_cols = ["gp_feature_1", "gp_feature_2", "gp_feature_3", "gp_feature_4", "gp_feature_8"]
    gp = gp[[c for c in keep_cols if c in gp.columns]]
    print(f"  遗传特征: {len(gp.columns)} 个 (去重后) <- {gp_file.name}", flush=True)
    return gp


def create_xgb_model(args):
    """
    创建二分类 XGBoost 模型。
    调参最优值：lr=0.1, mcw=8.0, reg_alpha=1.0
    """
    return XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=2.0,
        gamma=0.1,
        random_state=42,
        objective="binary:logistic",  # 二分类专用
        eval_metric="logloss",        # 二分类评估指标
        tree_method="hist",
        early_stopping_rounds=50,
    )


# =====================================================================
# 滚动窗口训练
# =====================================================================
def rolling_window_train(features, y_all, merged, args):
    """
    滚动窗口训练，返回所有 fold 的预测结果 + 逐 fold 指标 + 特征重要性。

    每个 fold：
    - 训练集: [start-W : start-V]
    - 验证集: [start-V : start]
    - 测试集: [start : start+step]

    新增返回：
    - fold_metrics: 每个 fold 的 Accuracy/F1/AUC
    - fold_feature_importances: 每个 fold 的 Top30 特征名 + importance
    """
    W, S, V = args.rolling_window, args.rolling_step, args.rolling_val_size
    n = len(features)

    all_preds, all_probas, all_true, all_dates, all_meta = [], [], [], [], []
    fold_metrics = []       # 每个 fold 的指标
    fold_feature_imps = []  # 每个 fold 的特征重要性 dict
    fold = 0

    for start in range(W, n - S + 1, S):
        end = min(start + S, n)
        x_tr = features.iloc[start - W:start - V]
        y_tr = y_all.iloc[start - W:start - V]
        x_val = features.iloc[start - V:start]
        y_val = y_all.iloc[start - V:start]
        x_te = features.iloc[start:end]
        y_te = y_all.iloc[start:end]
        if len(x_tr) < 30:  # 小窗口时训练样本少，阈值降到 30
            continue

        model = create_xgb_model(args)
        model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)

        yp = model.predict(x_te)
        proba = model.predict_proba(x_te)

        # === 收集该 fold 的指标 ===
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
        })

        # === 收集该 fold 的特征重要性（importance>0 的全部）===
        imp_series = pd.Series(model.feature_importances_, index=features.columns)
        nonzero_imp = imp_series[imp_series > 0].sort_values(ascending=False)
        fold_feature_imps.append(nonzero_imp.to_dict())

        all_preds.extend(yp.tolist())
        all_probas.extend(proba.tolist())
        all_true.extend(y_te.values.tolist())
        all_dates.extend(merged["Unnamed: 0"].iloc[start:end].tolist())
        all_meta.append(merged.iloc[start:end][["Unnamed: 0", "rate", "target_date_5d"]])
        fold += 1
        if fold % 5 == 0:
            print(f"  fold {fold}...", flush=True)

    print(f"  完成: {fold} folds, {len(all_preds)} 预测", flush=True)
    return all_preds, all_probas, all_true, all_dates, all_meta, fold, fold_metrics, fold_feature_imps


# =====================================================================
# 主流程
# =====================================================================
def main():
    args = parse_args()
    START_DATE = args.start_date
    W, S, V = args.rolling_window, args.rolling_step, args.rolling_val_size

    # ============ 1. 加载数据 ============
    print("[1/5] 加载数据...", flush=True)
    df_data = pd.read_csv("data/43_data.csv")
    df_rate = pd.read_csv("data/43_rate.csv")
    df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"])
    df_rate["time"] = pd.to_datetime(df_rate["time"])
    merged = pd.merge(df_data, df_rate, left_on="Unnamed: 0", right_on="time", how="inner").sort_values("Unnamed: 0")
    merged["target_date_5d"] = merged["Unnamed: 0"].shift(-5)
    merged["rate_delta_5d"] = merged["rate"].shift(-5) - merged["rate"]

    # 二分类标签：下行=1, 上行=0
    merged["label"] = merged["rate_delta_5d"].apply(gen_label)
    merged = merged.dropna(subset=["label"]).copy()
    merged["label"] = merged["label"].astype(int)
    y_all = merged["label"].copy()

    # ============ 2. 构建特征 ============
    print("[2/5] 构建特征...", flush=True)
    features = build_features(merged)

    # 合并遗传特征（去重）
    gp = load_genetic_features(args, BASE)
    if gp is not None:
        features = features.join(gp, how="left")
        gp_cols = [c for c in gp.columns if c in features.columns]
        features[gp_cols] = features[gp_cols].fillna(0.0)

    # 过滤起始日期
    cutoff = pd.to_datetime(START_DATE)
    mask = pd.to_datetime(merged["Unnamed: 0"]) >= cutoff
    features = features.loc[mask.values].reset_index(drop=True)
    y_all = y_all.loc[mask.values].reset_index(drop=True)
    merged = merged.loc[mask].reset_index(drop=True)
    print(f"  样本: {len(features)}, 特征数: {features.shape[1]}", flush=True)

    # ============ 3. 滚动训练 ============
    print("[3/5] 滚动窗口训练...", flush=True)
    all_preds, all_probas, all_true, all_dates, all_meta, fold, fold_metrics, fold_imps = rolling_window_train(
        features, y_all, merged, args
    )

    # ============ 4. 评估 ============
    print("[4/5] 评估...", flush=True)
    y_pred = np.array(all_preds)
    y_true = np.array(all_true)
    y_proba = np.array(all_probas)  # shape [N, 2]
    test_meta = pd.concat(all_meta, ignore_index=True)

    # 二分类指标
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    f1_pos = f1_score(y_true, y_pred, pos_label=1)  # 下行类 F1
    f1_neg = f1_score(y_true, y_pred, pos_label=0)  # 上行类 F1
    # AUC（需要概率）
    try:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    maj = int(pd.Series(y_true).mode().iloc[0])
    maj_acc = float(np.mean(y_true == maj))

    # ============ 5. 训练最终模型 + 保存 ============
    print("[5/5] 训练最终模型并保存...", flush=True)
    n = len(features)
    fm = create_xgb_model(args)
    fs, fv = max(0, n - W), max(0, n - V)
    fm.fit(features.iloc[fs:fv], y_all.iloc[fs:fv],
           eval_set=[(features.iloc[fv:n], y_all.iloc[fv:n])], verbose=False)
    joblib.dump(fm, BASE / "data/43_xgb_model_bin.joblib")

    # 保存特征和标签
    features.to_csv(BASE / "data/43_xgb_train_features_bin.csv")
    pd.DataFrame({"label": y_all}).to_csv(BASE / "data/43_xgb_labels_bin.csv", index=False)

    # 保存预测明细
    pred_df = pd.DataFrame({
        "date": pd.to_datetime(all_dates).astype(str),
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_0_上行": y_proba[:, 0],
        "prob_1_下行": y_proba[:, 1],
    })
    pred_df.to_csv(BASE / "data/43_xgb_test_predictions_bin.csv", index=False)

    # 保存可视化友好的预测表
    LABEL_MAP = {0: "上行(偏空)", 1: "下行(偏多)"}
    view_df = pd.DataFrame({
        "判断基点": pd.to_datetime(test_meta["Unnamed: 0"]).dt.strftime("%Y-%m-%d"),
        "当期10Y": test_meta["rate"].round(4),
        "AI研判": [LABEL_MAP[int(p)] for p in y_pred],
        "预测时点": pd.to_datetime(test_meta["target_date_5d"]).dt.strftime("%Y-%m-%d"),
        "上行(偏空)概率": [f"{v:.2%}" for v in y_proba[:, 0]],
        "下行(偏多)概率": [f"{v:.2%}" for v in y_proba[:, 1]],
    })
    view_df.to_csv(BASE / "data/43_xgb_test_predictions_view_bin.csv", index=False, encoding="utf-8-sig")
    view_df.tail(1).to_csv(BASE / "data/43_xgb_latest_prediction_view_bin.csv", index=False, encoding="utf-8-sig")

    # 保存特征重要性
    imp = pd.DataFrame({"feature": features.columns, "importance": fm.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(BASE / "data/43_xgb_feature_importance_bin.csv", index=False)

    # 保存指标汇总
    metrics = "\n".join([
        "XGBoost 二分类测试结果（下行=1 / 上行=0）",
        f"训练模式: rolling (window={W}, step={S}, val={V})",
        f"训练 folds: {fold}",
        f"数据起始: {START_DATE}",
        f"测试样本数: {len(y_true)}",
        f"总特征数: {features.shape[1]}",
        f"超参: depth={args.max_depth}, n={args.n_estimators}, lr={args.learning_rate}",
        "",
        "=== 核心指标 ===",
        f"Accuracy:       {acc:.4f}",
        f"Macro F1:       {mf1:.4f}",
        f"F1 (下行=1):    {f1_pos:.4f}",
        f"F1 (上行=0):    {f1_neg:.4f}",
        f"AUC:            {auc:.4f}",
        f"多数类基线标签: {maj} ({LABEL_MAP[maj]})",
        f"多数类基线 Acc: {maj_acc:.4f}",
        "",
        "=== 测试集标签分布 ===",
        pd.Series(y_true).value_counts(normalize=True).sort_index().to_string(),
        "",
        "=== 混淆矩阵 (行=真实, 列=预测) ===",
        pd.DataFrame(cm, index=["true_0_上行", "true_1_下行"], columns=["pred_0_上行", "pred_1_下行"]).to_string(),
        "",
        "=== Classification Report ===",
        classification_report(y_true, y_pred, target_names=["上行(偏空)", "下行(偏多)"], digits=4),
        "",
        "=== Top 20 重要特征 ===",
        imp.head(20).to_string(index=False),
    ])
    (BASE / "data/43_xgb_metrics_bin.txt").write_text(metrics, encoding="utf-8")

    # ============ 逐 fold 分析 + 特征稳定性统计 ============
    # 1. 逐 fold 指标表
    fold_df = pd.DataFrame(fold_metrics)

    # 2. 特征稳定性：统计每个特征在所有 fold 的 importance>0 中出现次数 + 平均 importance
    feature_stats = {}  # {feature_name: {"appear_count": int, "imp_sum": float, "rank_sum": int}}
    for i, imp_dict in enumerate(fold_imps):
        for rank, (feat, val) in enumerate(sorted(imp_dict.items(), key=lambda x: -x[1])):
            if feat not in feature_stats:
                feature_stats[feat] = {"appear_count": 0, "imp_sum": 0.0, "rank_sum": 0}
            feature_stats[feat]["appear_count"] += 1
            feature_stats[feat]["imp_sum"] += val
            feature_stats[feat]["rank_sum"] += (rank + 1)  # rank 1-based

    stability_df = pd.DataFrame([
        {
            "feature": feat,
            "appear_in_top30": s["appear_count"],
            "appear_ratio": s["appear_count"] / len(fold_imps),  # 在多少比例的 fold 中出现
            "avg_importance": s["imp_sum"] / s["appear_count"],
            "avg_rank": s["rank_sum"] / s["appear_count"],
        }
        for feat, s in feature_stats.items()
    ]).sort_values("appear_ratio", ascending=False)

    # 从未出现 importance>0 的特征（完全无效特征）
    all_features = set(features.columns)
    appeared = set(feature_stats.keys())
    never_nonzero = sorted(all_features - appeared)

    # 保存逐 fold 指标
    fold_df.to_csv(BASE / "data/43_xgb_fold_metrics_bin.csv", index=False)
    # 保存特征稳定性统计
    stability_df.to_csv(BASE / "data/43_xgb_feature_stability_bin.csv", index=False)
    # 保存从未出现 importance>0 的特征列表（完全无效）
    pd.DataFrame({"never_nonzero_feature": never_nonzero}).to_csv(
        BASE / "data/43_xgb_never_nonzero_bin.csv", index=False)

    # 追加到 metrics 文件
    with open(BASE / "data/43_xgb_metrics_bin.txt", "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("=== 逐 fold 指标 ===\n")
        f.write(f"{'fold':>4} {'date_start':>12} {'date_end':>12} {'n_test':>6} "
                f"{'Accuracy':>10} {'MacroF1':>10} {'AUC':>8} {'best_iter':>10}\n")
        f.write("-" * 70 + "\n")
        for _, row in fold_df.iterrows():
            f.write(f"{row['fold']:>4} {row['date_start']:>12} {row['date_end']:>12} "
                    f"{row['n_test']:>6} {row['accuracy']:>10.4f} {row['macro_f1']:>10.4f} "
                    f"{row['auc']:>8.4f} {row['best_iteration']:>10}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'mean':>4} {'':>12} {'':>12} {fold_df['n_test'].sum():>6} "
                f"{fold_df['accuracy'].mean():>10.4f} {fold_df['macro_f1'].mean():>10.4f} "
                f"{fold_df['auc'].mean():>8.4f}\n")

        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("=== 特征稳定性统计（importance>0 出现频率排名）===\n")
        f.write(f"总特征数: {len(all_features)}, 进入过 importance>0 的: {len(appeared)}, "
                f"从未进入: {len(never_top30)}\n\n")
        f.write(f"{'feature':<45} {'appear':>8} {'ratio':>8} {'avg_imp':>10} {'avg_rank':>10}\n")
        f.write("-" * 70 + "\n")
        for _, row in stability_df.head(30).iterrows():
            f.write(f"{row['feature']:<45} {row['appear_in_top30']:>8} "
                    f"{row['appear_ratio']:>7.0%} {row['avg_importance']:>10.4f} "
                    f"{row['avg_rank']:>10.1f}\n")

        f.write(f"\n--- 从未出现 importance>0 的特征 ({len(never_nonzero)} 个) ---\n")
        for feat in never_nonzero:
            f.write(f"  {feat}\n")

    # 控制台输出
    print(f"\n{'='*50}")
    print(f"Accuracy:    {acc:.4f}  (基线: {maj_acc:.4f})")
    print(f"Macro F1:    {mf1:.4f}")
    print(f"AUC:         {auc:.4f}")
    print(f"F1(下行):    {f1_pos:.4f}  F1(上行): {f1_neg:.4f}")
    print(f"特征数: {features.shape[1]}, folds: {fold}")
    print(f"{'='*50}")
    print("\nTop 10 重要特征:")
    print(imp.head(10).to_string(index=False))
    print("\nDONE")


if __name__ == "__main__":
    main()
