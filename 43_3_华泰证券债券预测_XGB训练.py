"""
10年期国债收益率未来5天方向预测 —— XGBoost 三分类基线训练

本脚本复用 `43_2_华泰证券债券预测_训练数据构建.py` 中的训练底表构造逻辑：
1. 读取 `data/43_data.csv` 与 `data/43_rate.csv`
2. 合并后构造 `rate_delta_5d`
3. 按阈值生成三分类标签
4. 使用原始训练特征训练 XGBoost 三分类模型

输出文件：
  data/43_xgb_train_features.csv         -> 训练使用的原始特征
  data/43_xgb_labels.csv                 -> 标签
  data/43_xgb_test_predictions.csv       -> 测试集预测结果与概率
  data/43_xgb_feature_importance.csv     -> 特征重要性
  data/43_xgb_metrics.txt                -> 评估指标
  data/43_xgb_model.joblib              -> 训练后的模型
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

LABEL_NAME_MAP = {
    0: "偏多",
    1: "震荡",
    2: "偏空",
}


def load_data_builder_module(script_path: Path):
    """动态加载训练数据构建脚本，复用其中的 build_training_dataset。"""
    spec = importlib.util.spec_from_file_location("train_data_builder", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 XGBoost 三分类模型")
    parser.add_argument("--data-file", default="data/43_data.csv", help="预处理后的环境特征文件")
    parser.add_argument("--rate-file", default="data/43_rate.csv", help="预处理后的利率特征文件")
    parser.add_argument("--output-dir", default="data", help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集占比，按时间顺序切分")
    parser.add_argument("--validation-ratio", type=float, default=0.15, help="训练集内部验证集占比，用于早停")
    parser.add_argument("--n-estimators", type=int, default=600, help="树数量上限，实际会被早停截断")
    parser.add_argument("--max-depth", type=int, default=3, help="树深度")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="学习率")
    parser.add_argument("--subsample", type=float, default=0.8, help="样本采样比例")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="列采样比例")
    parser.add_argument("--min-child-weight", type=float, default=8.0, help="叶子节点最小样本权重和")
    parser.add_argument("--reg-alpha", type=float, default=0.5, help="L1 正则")
    parser.add_argument("--reg-lambda", type=float, default=2.0, help="L2 正则")
    parser.add_argument("--gamma", type=float, default=0.1, help="节点继续分裂所需的最小增益")
    parser.add_argument("--early-stopping-rounds", type=int, default=50, help="验证集早停轮数")
    parser.add_argument(
        "--class-weight-mode",
        choices=["none", "balanced"],
        default="none",
        help="训练样本权重模式；balanced 更照顾少数类，但可能降低整体准确率",
    )
    parser.add_argument(
        "--genetic-features-file",
        default="data/43_x.csv",
        help="遗传算法生成的特征文件；不存在则跳过",
    )
    parser.add_argument(
        "--train-mode",
        choices=["fixed", "rolling"],
        default="rolling",
        help="训练模式：fixed=固定切分 | rolling=滚动窗口训练（每月重训）",
    )
    parser.add_argument(
        "--start-date",
        default="2018-01-01",
        help="训练数据起始日期；缩短训练集以适应新市场环境",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=756,
        help="滚动窗口大小（交易日），默认 756 ≈ 3年",
    )
    parser.add_argument(
        "--rolling-step",
        type=int,
        default=63,
        help="滚动步长（交易日），默认 63 ≈ 每季度重训一次",
    )
    parser.add_argument(
        "--rolling-val-size",
        type=int,
        default=63,
        help="滚动窗口内验证集大小（交易日），默认 63 ≈ 3个月",
    )
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    return parser.parse_args()


def build_model_features(merged: pd.DataFrame) -> pd.DataFrame:
    """
    从训练底表中构建更适合 XGBoost 的截面特征。

    说明：
    1. 保留当前利率 `rate`，它是可观测信息，不属于目标泄露
    2. 增加利率滞后、均线、波动率与短中期斜率，补足时序结构
    3. 增加利差/汇率衍生特征，捕捉 2022 年后中美货币环境分化
    4. 其余宏观/流动性特征沿用预处理结果
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

    for column in ["momentum_1_robust_scaled", "momentum_7_robust_scaled", "momentum_30_robust_scaled"]:
        if column in features.columns:
            features[f"{column}_lag_1"] = features[column].shift(1)
            features[f"{column}_lag_5"] = features[column].shift(5)

    # --- 利差与汇率衍生特征 ---
    # 原始数据无美债收益率字段，用美元指数 + 汇率 + 信用利差构造代理特征
    # 这些特征在 2022 年后出现结构性变化，能帮助模型适应新环境

    # 1) 信用利差变化（反映信用环境收紧/宽松）
    for cs_col in ["1年信用利差", "3年信用利差", "5年信用利差"]:
        if cs_col in features.columns:
            features[f"{cs_col}_chg_5d"] = features[cs_col].diff(5)
            features[f"{cs_col}_chg_20d"] = features[cs_col].diff(20)
            features[f"{cs_col}_ma_20"] = features[cs_col].rolling(20, min_periods=1).mean()

    # 2) 期限利差（5年-1年信用利差，反映曲线形态）
    if "5年信用利差" in features.columns and "1年信用利差" in features.columns:
        features["credit_term_spread_5_1"] = features["5年信用利差"] - features["1年信用利差"]

    # 3) 美元指数动量（反映外部流动性压力）
    if "美元指数" in features.columns:
        features["dollar_chg_5d"] = features["美元指数"].diff(5)
        features["dollar_chg_20d"] = features["美元指数"].diff(20)
        features["dollar_ma_20"] = features["美元指数"].rolling(20, min_periods=1).mean()
        features["dollar_deviation_20"] = features["美元指数"] - features["dollar_ma_20"]

    # 4) 汇率衍生（反映资本流动方向）
    if "即期汇率:美元兑人民币" in features.columns:
        features["fx_chg_5d"] = features["即期汇率:美元兑人民币"].diff(5)
        features["fx_chg_20d"] = features["即期汇率:美元兑人民币"].diff(20)
        if "中间价:美元兑人民币" in features.columns:
            # 中间价与即期汇率之差变化，反映央行汇率调控力度
            features["fx干预_chl_5d"] = (features["中间价:美元兑人民币"] - features["即期汇率:美元兑人民币"]).diff(5)

    # 5) 利率与美元指数协同（2022年后两者相关性增强）
    if "美元指数" in features.columns:
        features["rate_dollar_gap"] = features["rate"] - features["美元指数"].rolling(60, min_periods=1).mean() / 20

    # 6) 股债相对回报率衍生（沪深300股息率/十年期国债收益率）
    #    反映股债性价比切换，衍生出动量与均线特征
    if "股债相对回报率" in features.columns:
        features["股债相对回报率_chg_5d"] = features["股债相对回报率"].diff(5)
        features["股债相对回报率_chg_20d"] = features["股债相对回报率"].diff(20)
        features["股债相对回报率_ma_5"] = features["股债相对回报率"].rolling(5, min_periods=1).mean()
        features["股债相对回报率_ma_20"] = features["股债相对回报率"].rolling(20, min_periods=1).mean()
        features["股债相对回报率_deviation_20"] = features["股债相对回报率"] - features["股债相对回报率_ma_20"]

    # 7) 加杠杆情绪衍生（R001/R007成交量、隔夜回购占比）
    #    成交量变化反映机构加杠杆意愿，隔夜占比变化反映资金期限偏好
    for vol_col in ["成交量:R001", "成交量:R007"]:
        if vol_col in features.columns:
            features[f"{vol_col}_chg_5d"] = features[vol_col].diff(5)
            features[f"{vol_col}_chg_20d"] = features[vol_col].diff(20)
            features[f"{vol_col}_ma_5"] = features[vol_col].rolling(5, min_periods=1).mean()
            features[f"{vol_col}_ma_20"] = features[vol_col].rolling(20, min_periods=1).mean()

    # 隔夜回购占比（R001/银行间质押式回购成交量）衍生
    overnight_col = "成交量:R001/中国:成交量:银行间质押式回购"
    if overnight_col in features.columns:
        features["隔夜回购占比_chg_5d"] = features[overnight_col].diff(5)
        features["隔夜回购占比_chg_20d"] = features[overnight_col].diff(20)
        features["隔夜回购占比_ma_5"] = features[overnight_col].rolling(5, min_periods=1).mean()
        features["隔夜回购占比_ma_20"] = features[overnight_col].rolling(20, min_periods=1).mean()
        features["隔夜回购占比_deviation_20"] = features[overnight_col] - features["隔夜回购占比_ma_20"]

    # R001与R007成交量比（短期资金需求结构）
    if "成交量:R001" in features.columns and "成交量:R007" in features.columns:
        features["R001_R007_vol_ratio"] = features["成交量:R001"] / (features["成交量:R007"] + 1e-8)

    # 8) 利率相对强度特征（optimization P1）
    #    - 标准化变化量：将绝对变化除以近期波动，衡量"相对于近期有多大偏离"
    #    - 滚动百分位：当前利率在近期分布中的位置，0=近期最低，1=近期最高
    #    这些特征比纯粹的绝对变化量更有统计意义
    if "rate_std_20" in features.columns:
        features["rate_chg_5d_zscore"] = features["rate"].diff(5) / features["rate_std_20"]
        features["rate_chg_10d_zscore"] = features["rate"].diff(10) / features["rate_std_20"]
    m_window = 60
    rwindow = features["rate"].rolling(m_window, min_periods=m_window)
    features["rate_percentile_60"] = rwindow.rank(pct=True)    # 滚动百分位
    features["rate_zscore_60"] = (features["rate"] - rwindow.mean()) / rwindow.std()  # 标准化偏离

    # 只剔除未来信息和非模型输入字段，保留当前 rate 及其衍生特征
    features = features.drop(
        columns=["time", "label", "rate_delta_5d", "date_list"],
        errors="ignore",
    ).copy()
    features = features.set_index("Unnamed: 0")
    features = features.apply(pd.to_numeric, errors="coerce")

    # 用中位数填充新增统计特征，再对极少数全空列回退到 99.00
    median_values = features.median(numeric_only=True)
    features = features.fillna(median_values).fillna(99.00)
    return features


def filter_by_start_date(features: pd.DataFrame, labels: pd.Series, merged: pd.DataFrame, start_date: str):
    """按起始日期过滤数据，缩短训练集以适应新市场环境。"""
    cutoff = pd.to_datetime(start_date)
    mask = pd.to_datetime(merged["Unnamed: 0"]) >= cutoff
    # 同步过滤 features、labels、merged
    filtered_features = features.loc[mask.values].copy()
    filtered_labels = labels.loc[mask.values].copy()
    filtered_merged = merged.loc[mask].reset_index(drop=True).copy()
    dropped = len(features) - len(filtered_features)
    if dropped > 0:
        print(f"已过滤 {dropped} 条 {start_date} 之前的样本，剩余 {len(filtered_features)} 条")
    return filtered_features, filtered_labels, filtered_merged


def create_xgb_model(args) -> XGBClassifier:
    """创建 XGBoost 模型实例，统一参数管理。"""
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        random_state=args.random_state,
        eval_metric="merror",
        tree_method="hist",
        early_stopping_rounds=args.early_stopping_rounds,
    )


def get_time_decay_weight(n_samples: int, decay_rate: float = 0.999) -> np.ndarray:
    """
    时间衰减样本权重：越近的样本权重越大。

    使用指数衰减，最近的样本权重接近1，最远的权重接近 exp(-decay_rate*n)。
    权重归一化后总和等于样本数，不影响梯度计算。

    Args:
        n_samples: 训练样本数
        decay_rate: 衰减率，默认 0.999（3年窗口末端权重≈0.47，比0.998的0.22更温和）
    """
    ages = np.arange(n_samples)[::-1]  # 距离最近的年龄：0, 1, 2, ...
    weights = np.exp(-decay_rate * ages)
    weights = weights / weights.sum() * n_samples
    return weights


def rolling_window_train(
    features: pd.DataFrame,
    labels: pd.Series,
    merged: pd.DataFrame,
    args,
) -> tuple:
    """
    滚动窗口训练：每月用过去 N 天数据重训模型，预测下一个月。

    优势：
    - 始终使用最近的市场数据，适应 regime change
    - 避免远期历史数据（如 2010-2015）干扰当前预测

    参数：
    - rolling_window: 训练窗口大小（交易日）
    - rolling_step: 每次滚动的步长（交易日）
    - rolling_val_size: 窗口末尾的验证集大小（用于早停）
    """
    n = len(features)
    window = args.rolling_window
    step = args.rolling_step
    val_size = args.rolling_val_size

    all_preds = []
    all_probas = []
    all_true = []
    all_dates = []
    all_meta = []
    fold_count = 0

    for start in range(window, n - step + 1, step):
        end = min(start + step, n)

        # 训练窗口：[start-window, start-val_size)
        # 验证窗口：[start-val_size, start)
        # 预测窗口：[start, end)
        train_start = start - window
        val_start = start - val_size

        x_tr = features.iloc[train_start:val_start]
        y_tr = labels.iloc[train_start:val_start]
        x_val = features.iloc[val_start:start]
        y_val = labels.iloc[val_start:start]
        x_te = features.iloc[start:end]
        y_te = labels.iloc[start:end]

        if len(x_tr) < 100 or len(x_te) == 0:
            continue

        sample_weight = get_training_sample_weight(y_tr, args.class_weight_mode)
        # 叠加时间衰减权重：将时间衰减作为基础权重，再乘以类平衡权重（如有）
        time_weights = get_time_decay_weight(len(x_tr))
        if sample_weight is not None:
            sample_weight = sample_weight * time_weights
        else:
            sample_weight = time_weights
        model = create_xgb_model(args)
        model.fit(
            x_tr, y_tr,
            sample_weight=sample_weight,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(x_te)
        y_proba = model.predict_proba(x_te)

        all_preds.extend(y_pred.tolist())
        all_probas.extend(y_proba.tolist())
        all_true.extend(y_te.values.tolist())
        test_dates_slice = merged["Unnamed: 0"].iloc[start:end].tolist()
        all_dates.extend(test_dates_slice)
        all_meta.append(merged.iloc[start:end][["Unnamed: 0", "rate", "target_date_5d"]])
        fold_count += 1

    if not all_preds:
        raise ValueError("滚动训练未产生任何预测，请检查 rolling_window/step 参数。")

    y_pred_all = pd.Series(all_preds)
    y_true_all = pd.Series(all_true)
    y_proba_all = np.array(all_probas)
    test_dates_all = pd.Series(all_dates)
    test_meta_all = pd.concat(all_meta, ignore_index=True)

    print(f"滚动训练完成: {fold_count} 个 fold, 预测 {len(y_pred_all)} 个样本")
    return y_pred_all, y_true_all, y_proba_all, test_dates_all, test_meta_all


def split_train_validation_test(
    features: pd.DataFrame,
    labels: pd.Series,
    train_ratio: float,
    validation_ratio: float,
):
    """按时间顺序切分 train/validation/test，避免未来信息进入训练。"""
    split_idx = int(len(features) * train_ratio)
    if split_idx <= 0 or split_idx >= len(features):
        raise ValueError("train_ratio 切分后训练集或测试集为空，请调整参数。")

    validation_size = int(split_idx * validation_ratio)
    if validation_size <= 0 or validation_size >= split_idx:
        raise ValueError("validation_ratio 导致验证集为空或覆盖全部训练集，请调整参数。")

    train_end = split_idx - validation_size
    x_train_split = features.iloc[:train_end].copy()
    y_train_split = labels.iloc[:train_end].copy()
    x_val_split = features.iloc[train_end:split_idx].copy()
    y_val_split = labels.iloc[train_end:split_idx].copy()
    x_test_split = features.iloc[split_idx:].copy()
    y_test_split = labels.iloc[split_idx:].copy()
    return x_train_split, y_train_split, x_val_split, y_val_split, x_test_split, y_test_split, split_idx


def get_training_sample_weight(labels: pd.Series, class_weight_mode: str):
    """按配置生成训练样本权重。"""
    if class_weight_mode == "balanced":
        return compute_sample_weight(class_weight="balanced", y=labels)
    return None


def load_genetic_features(features_file: Path, model_features: pd.DataFrame) -> pd.DataFrame:
    """
    加载遗传算法生成的特征（gp_feature_1~10），按日期合并到训练特征中。

    遗传特征来源：
    - 由 43_2 脚本中的 SymbolicTransformer 训练生成
    - 函数集参考 model_question/gl_feature.py：ts_corr/ts_cov/ts_mean/
      ts_stddev/ts_kurt/ts_skew/ts_zscore/decay_linear/ts_ascsortcut/
      ts_decsortcut + 逻辑函数，窗口 [5,20,60,120]
    - 通过 pearson 相关性筛选出与标签最相关的 10 个非线性组合特征

    去重逻辑：
    根据 43_x_formula.txt，gp_feature_2/5/6/7 为同一公式重复生成，
    gp_feature_8/9/10 为同一公式重复生成。
    仅保留首现者：gp_feature_1,2,3,4,8（共5个有效特征）。
    """
    if not features_file.exists():
        print(f"警告: 遗传特征文件不存在 ({features_file})，跳过合并。")
        return model_features

    gp_features = pd.read_csv(features_file, index_col=0)
    gp_features.index = pd.to_datetime(gp_features.index, errors="coerce")
    gp_features = gp_features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # 去重：gp_feature_5/6/7 与 gp_feature_2 重复，gp_feature_9/10 与 gp_feature_8 重复
    keep_cols = ["gp_feature_1", "gp_feature_2", "gp_feature_3", "gp_feature_4", "gp_feature_8"]
    gp_features = gp_features[[c for c in keep_cols if c in gp_features.columns]]
    dropped = 10 - len(gp_features.columns)
    print(f"遗传特征去重: 保留 {len(gp_features.columns)} 个, 移除 {dropped} 个重复")

    merged = model_features.join(gp_features, how="left")
    gp_cols = [c for c in gp_features.columns if c in merged.columns]
    merged[gp_cols] = merged[gp_cols].fillna(0.0)
    print(f"已合并遗传特征: {len(gp_cols)} 个 -> {gp_cols}")
    return merged


def format_ai_view(probabilities: pd.Series) -> str:
    """
    将三分类概率转换为展示用观点。

    展示规则：
    - 主观点取最大概率类别
    - 括号内展示次大概率类别，便于体现次优判断
    """
    ranked = probabilities.sort_values(ascending=False)
    primary_label = LABEL_NAME_MAP[int(ranked.index[0].split("_")[-1])]
    secondary_label = LABEL_NAME_MAP[int(ranked.index[1].split("_")[-1])]
    return f"{primary_label}（{secondary_label}）"


def build_prediction_view_df(pred_df: pd.DataFrame, test_meta: pd.DataFrame) -> pd.DataFrame:
    """生成仿照业务展示口径的预测结果表。"""
    test_meta = test_meta.reset_index(drop=True).copy()
    probability_columns = ["prob_0", "prob_1", "prob_2"]
    view_df = pd.DataFrame(
        {
            "判断基点": pd.to_datetime(test_meta["Unnamed: 0"]).dt.strftime("%Y-%m-%d"),
            "当期10Y": test_meta["rate"].round(4),
            "AI研判": pred_df[probability_columns].apply(format_ai_view, axis=1),
            "预测时点": pd.to_datetime(test_meta["target_date_5d"]).dt.strftime("%Y-%m-%d"),
            "偏多": pred_df["prob_0"].map(lambda value: f"{value:.2%}"),
            "震荡": pred_df["prob_1"].map(lambda value: f"{value:.2%}"),
            "偏空": pred_df["prob_2"].map(lambda value: f"{value:.2%}"),
        }
    )
    return view_df


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    builder_script = base_dir / "43_2_华泰证券债券预测_训练数据构建.py"
    builder_module = load_data_builder_module(builder_script)

    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    merged, _x_train, y_train = builder_module.build_training_dataset(
        base_dir / args.data_file,
        base_dir / args.rate_file,
    )

    model_features = build_model_features(merged)
    model_features = load_genetic_features(base_dir / args.genetic_features_file, model_features)

    # --- 缩短训练集：按起始日期过滤 ---
    model_features, y_train, merged = filter_by_start_date(model_features, y_train, merged, args.start_date)

    # --- 根据训练模式选择路径 ---
    if args.train_mode == "rolling":
        # 滚动窗口训练
        y_pred, y_test_split, y_proba, test_dates, test_meta = rolling_window_train(
            model_features, y_train, merged, args,
        )
        train_info = f"滚动窗口 (window={args.rolling_window}, step={args.rolling_step})"
        val_info = f"滚动验证集 (val_size={args.rolling_val_size})"
        best_iter_info = "N/A (滚动多模型)"
        # 保存最终模型：用最后一段数据训练一个用于推理
        final_model = create_xgb_model(args)
        n = len(model_features)
        final_train_start = max(0, n - args.rolling_window)
        final_val_start = max(0, n - args.rolling_val_size)
        final_weights = get_time_decay_weight(len(model_features.iloc[final_train_start:final_val_start]))
        final_class_weight = get_training_sample_weight(
            y_train.iloc[final_train_start:final_val_start], args.class_weight_mode,
        )
        if final_class_weight is not None:
            final_weights = final_weights * final_class_weight
        final_model.fit(
            model_features.iloc[final_train_start:final_val_start],
            y_train.iloc[final_train_start:final_val_start],
            sample_weight=final_weights,
            eval_set=[(
                model_features.iloc[final_val_start:n],
                y_train.iloc[final_val_start:n],
            )],
            verbose=False,
        )
        final_model_path = output_dir / "43_xgb_model.joblib"
        joblib.dump(final_model, final_model_path)
        importance_df = pd.DataFrame(
            {"feature": model_features.columns, "importance": final_model.feature_importances_},
        ).sort_values("importance", ascending=False)
    else:
        # 固定切分训练（原逻辑）
        x_train_split, y_train_split, x_val_split, y_val_split, x_test_split, y_test_split, split_idx = (
            split_train_validation_test(
                model_features,
                y_train,
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
            )
        )
        test_dates = merged.iloc[split_idx:]["Unnamed: 0"].reset_index(drop=True)
        test_meta = merged.iloc[split_idx:][["Unnamed: 0", "rate", "target_date_5d"]].reset_index(drop=True)
        sample_weight = get_training_sample_weight(y_train_split, args.class_weight_mode)
        time_weights = get_time_decay_weight(len(x_train_split))
        if sample_weight is not None:
            sample_weight = sample_weight * time_weights
        else:
            sample_weight = time_weights

        model = create_xgb_model(args)
        model.fit(
            x_train_split, y_train_split,
            sample_weight=sample_weight,
            eval_set=[(x_val_split, y_val_split)],
            verbose=False,
        )
        y_pred = model.predict(x_test_split)
        y_proba = model.predict_proba(x_test_split)
        y_test_split = y_test_split.reset_index(drop=True)
        best_iter_info = str(getattr(model, "best_iteration", None))
        train_info = f"{len(x_train_split)}"
        val_info = f"{len(x_val_split)}"

        model_path = output_dir / "43_xgb_model.joblib"
        joblib.dump(model, model_path)
        importance_df = pd.DataFrame(
            {"feature": model_features.columns, "importance": model.feature_importances_},
        ).sort_values("importance", ascending=False)

    # --- 统一评估与输出 ---
    accuracy = accuracy_score(y_test_split, y_pred)
    macro_f1 = f1_score(y_test_split, y_pred, average="macro")
    report = classification_report(y_test_split, y_pred, digits=4)
    cm = confusion_matrix(y_test_split, y_pred)
    majority_label = int(y_test_split.mode().iloc[0])
    majority_accuracy = float((y_test_split == majority_label).mean())
    test_distribution = y_test_split.value_counts(normalize=True).sort_index()

    model_features.to_csv(output_dir / "43_xgb_train_features.csv")
    pd.DataFrame({"label": y_train}).to_csv(output_dir / "43_xgb_labels.csv", index=False)

    pred_df = pd.DataFrame(
        {
            "date": pd.to_datetime(test_dates).astype(str),
            "y_true": y_test_split.values if hasattr(y_test_split, "values") else y_test_split,
            "y_pred": y_pred.values if hasattr(y_pred, "values") else y_pred,
            "prob_0": y_proba[:, 0],
            "prob_1": y_proba[:, 1],
            "prob_2": y_proba[:, 2],
        }
    )
    pred_df.to_csv(output_dir / "43_xgb_test_predictions.csv", index=False)
    prediction_view_df = build_prediction_view_df(pred_df, test_meta)
    prediction_view_df.to_csv(output_dir / "43_xgb_test_predictions_view.csv", index=False, encoding="utf-8-sig")
    prediction_view_df.tail(1).to_csv(output_dir / "43_xgb_latest_prediction_view.csv", index=False, encoding="utf-8-sig")

    importance_df.to_csv(output_dir / "43_xgb_feature_importance.csv", index=False)

    metrics_text = "\n".join(
        [
            "XGBoost 三分类测试结果",
            f"训练模式: {args.train_mode}",
            f"训练集: {train_info}",
            f"验证集: {val_info}",
            f"测试样本数: {len(y_test_split)}",
            f"数据起始: {args.start_date}",
            f"Accuracy: {accuracy:.4f}",
            f"Macro F1: {macro_f1:.4f}",
            f"多数类基线标签: {majority_label}",
            f"多数类基线 Accuracy: {majority_accuracy:.4f}",
            f"样本权重模式: {args.class_weight_mode}",
            f"最佳迭代轮数: {best_iter_info}",
            "",
            "测试集标签分布:",
            test_distribution.to_string(),
            "",
            "Confusion Matrix:",
            pd.DataFrame(cm, index=["true_0", "true_1", "true_2"], columns=["pred_0", "pred_1", "pred_2"]).to_string(),
            "",
            "Classification Report:",
            report,
        ]
    )
    (output_dir / "43_xgb_metrics.txt").write_text(metrics_text, encoding="utf-8")

    print("XGBoost 训练完成")
    print(f"训练模式: {args.train_mode} | 数据起始: {args.start_date}")
    print(f"测试样本数: {len(y_test_split)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"多数类基线 Accuracy: {majority_accuracy:.4f}")
    print(f"模型文件: {output_dir / '43_xgb_model.joblib'}")
    print(f"展示结果文件: {output_dir / '43_xgb_test_predictions_view.csv'}")


if __name__ == "__main__":
    main()
