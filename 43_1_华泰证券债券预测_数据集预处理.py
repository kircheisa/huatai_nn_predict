"""
华泰证券债券预测 —— 数据集预处理流水线

================================================================================
整体设计目标：
  1. 对齐到日频 —— 所有特征最终统一为日度，适配预测模型
  2. 防止未来信息泄露 —— 低频数据使用"发布日期"而非"数据所属日期"做前向合并
  3. 鲁棒标准化 —— 中位数/IQR 替代均值/标准差，抵抗金融数据异常值
  4. 保留时序结构 —— 用滚动窗口标准化，避免跨时期信息串扰
  5. 完善的缺失值策略 —— 前向填充 + 特殊标记值(99.00)，区分"有数据但缺失"和"无数据"

================================================================================
输入：data/43_origin_data.xlsx（8 个 sheet）
  sheet 0 — 季频数据（如 GDP 同比）
  sheet 1 — 季频数据发布日
  sheet 2 — 月频数据
  sheet 3 — 月频数据发布日
  sheet 4 — 周频数据
  sheet 5 — 日频数据
  sheet 6 — 特殊频率（逆回购利率、存款准备金率）
  sheet 7 — 分类变量（牛熊阶段、政策周期等）

================================================================================
预处理流水线（9 个步骤）：

  步骤1：多频率数据滚动鲁棒标准化
    - 公式：标准化值 = (当前值 - 滚动窗口中位数) / 滚动窗口IQR
    - 日频窗口=252天、周频=52周、月频=12月、季频(GDP)=累进窗口
    - 原因：金融数据常有极端值，中位数/IQR 是鲁棒统计量

  步骤2：日频 + 周频按日期前向合并
  步骤3：合并季度 GDP（以发布日期为索引做前向匹配）
    - GDP 发布日滞后于数据所属季度，用发布日期避免未来信息
  步骤4：合并月频数据（同上逻辑，每列有独立发布日）
  步骤5：合并特殊频率数据（政策利率变动标记）
    - 逆回购利率、存准率变动时保留原值，不变时置 0
  步骤6：分类变量布尔化（One-Hot 编码）
  步骤7：缺失值处理 —— 前向填充 + 余缺填 99.00
  步骤8：日期列表化 —— [年, 月, 日] 格式
  步骤9：利率动量特征计算（产出 43_rate.csv）
    - 动量窗口: 1d, 7d, 30d, 60d, 180d
    - 动量再做滚动鲁棒标准化

================================================================================
输出：
  data/43_data.csv — 合并后的日频宽表
  data/43_rate.csv — 利率 + 动量特征
================================================================================
"""

import logging
import os
import json
import pickle
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
CHECKPOINT_VERSION = "2026-06-29-resume-v1"


def setup_logging():
    if LOGGER.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )


def validate_required_columns(df, required_columns, name):
    """校验数据表是否包含必需的列"""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少列: {missing}")


def ensure_datetime_column(df, column_name, name):
    """将指定列强制转换为 datetime 类型，用于后续排序和合并"""
    df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
    if df[column_name].isna().all():
        raise ValueError(f"{name} 的 {column_name} 无法解析为日期")
    return df


# ================================================================================
# 核心工具函数：滚动鲁棒标准化
#
# 公式：标准化值 = (当前值 - 滚动窗口中位数) / 滚动窗口 IQR
#
# 为什么用中位数/IQR 而不是均值/标准差？
#   金融数据常有极端值（如金融危机时刻），均值和标准差会被异常值扭曲。
#   中位数和 IQR（四分位距 = Q3 - Q1）是鲁棒统计量，不受极端值影响，
#   能更稳定地反映"当前值在历史分布中的相对位置"。
#
# 为什么用滚动窗口而不是全局标准化？
#   如果用全历史数据的均值和标准差做标准化，等于让模型"看到了未来"。
#   滚动窗口保证每个时间点的标准化只用到它之前的数据，不泄露未来信息。
# ================================================================================

def rolling_robust_standardize(arr, window):
    """
    对一维数组做滚动鲁棒标准化（基于中位数和 IQR）

    参数:
        arr:  原始数据数组
        window: 滚动窗口大小
    返回:
        标准化后的数组（前 window-1 个位置为 NaN，因为窗口数据不足）
    """
    if window <= 0:
        raise ValueError("window 必须为正整数")
    # 确保数据类型为 float64，非数值转为 NaN
    series = pd.to_numeric(pd.Series(arr), errors="coerce")
    arr = series.to_numpy(dtype=np.float64)
    if arr.size < window:
        return np.full(arr.shape, np.nan, dtype=np.float64)
    # 用滑动窗口视图一次性生成所有窗口，避免 Python 循环
    window_view = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window)
    # 窗口内全为 NaN 时抑制 RuntimeWarning: All-NaN slice encountered
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rolling_median = np.nanmedian(window_view, axis=1)    # 滚动中位数
            rolling_q1 = np.nanpercentile(window_view, 25, axis=1) # 滚动Q1
            rolling_q3 = np.nanpercentile(window_view, 75, axis=1) # 滚动Q3
    rolling_iqr = rolling_q3 - rolling_q1                          # 滚动 IQR
    rolling_iqr[rolling_iqr == 0] = np.nan                         # IQR=0 时避免除零
    standardized_values = (arr[window - 1:] - rolling_median) / rolling_iqr
    result = np.full(arr.shape, np.nan, dtype=np.float64)
    result[window - 1:] = standardized_values
    return result


def apply_rolling_robust_std_to_df(df_origin, window, date_col="Unnamed: 0"):
    """
    对 DataFrame 的每个数值列做滚动鲁棒标准化

    处理流程：
      1. 解析日期列、排序、设为索引
      2. 逐列调用 rolling_robust_standardize
    返回与原 DataFrame 相同 index 的结果
    """
    validate_required_columns(df_origin, [date_col], "数据表")
    df = df_origin.copy()
    df = ensure_datetime_column(df, date_col, "数据表")
    df = df.sort_values(by=date_col, ascending=True).set_index(date_col)
    result_df = pd.DataFrame(index=df.index)
    for column in df.columns:
        result_df[column] = rolling_robust_standardize(df[column].to_numpy(), window)
    print("apply_rolling_robust_std_to_df_stage done")
    return result_df


def safe_read_excel(file_name, sheet_name, name):
    """读取 Excel，优先用 openpyxl 引擎，失败时降级尝试默认引擎"""
    try:
        return pd.read_excel(file_name, sheet_name=sheet_name, engine="openpyxl")
    except Exception as exc:
        LOGGER.warning("读取 %s 失败，尝试降级读取: %s", name, exc)
        return pd.read_excel(file_name, sheet_name=sheet_name)


def build_checkpoint_state(file_name, rate_path):
    """构造当前输入文件状态，用于判断旧检查点是否仍然可复用"""
    state = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "file_name": os.path.abspath(file_name),
        "file_mtime": os.path.getmtime(file_name),
        "rate_path": os.path.abspath(rate_path)
    }
    if os.path.exists(rate_path):
        state["rate_mtime"] = os.path.getmtime(rate_path)
    else:
        state["rate_mtime"] = None
    return state


def load_checkpoint_state(state_path):
    """读取检查点状态文件，不存在时返回 None"""
    if not os.path.exists(state_path):
        return None
    with open(state_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_checkpoint_state(state_path, state):
    """保存当前检查点状态，供下次运行时校验是否可续跑"""
    with open(state_path, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)


def clear_stale_checkpoints(checkpoint_dir):
    """输入文件变更时清理旧检查点，避免加载过期中间结果"""
    if not os.path.exists(checkpoint_dir):
        return
    for entry in os.listdir(checkpoint_dir):
        entry_path = os.path.join(checkpoint_dir, entry)
        if os.path.isfile(entry_path):
            os.remove(entry_path)


def load_or_run_checkpoint(step_name, checkpoint_dir, builder):
    """
    带检查点执行步骤：
      - 若已有同名检查点，则直接加载
      - 否则执行 builder，并将结果持久化到磁盘
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"{step_name}.pkl")
    if os.path.exists(checkpoint_path):
        LOGGER.info("检查点命中，跳过步骤: %s", step_name)
        with open(checkpoint_path, "rb") as file:
            return pickle.load(file)
    LOGGER.info("开始执行步骤: %s", step_name)
    result = builder()
    with open(checkpoint_path, "wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.info("步骤完成并写入检查点: %s", step_name)
    return result


# ================================================================================
# 季度 GDP 数据处理
#
# 关键设计：
#   1. 累进窗口标准化 —— 第 i 期用 [0..i] 的数据做标准化，模拟"当时只知道历史数据"
#   2. 发布日期合并 —— GDP 数据所属季度与发布日不同，用发布日做索引对齐，
#      避免模型在发布日期之前就"看到"了尚未公布的 GDP
# ================================================================================

def standardize_quarterly(quar_data, quar_date):
    """
    对季度数据进行累进鲁棒标准化，并按发布日期对齐

    参数:
        quar_data: 季度经济数据（如 GDP 同比）
        quar_date: 季度数据对应的发布日期表（含 "GDP同比" 发布日列）
    返回:
        以发布日期为索引的标准化季度数据
    """
    validate_required_columns(quar_data, ["Unnamed: 0", "中国:GDP:不变价:当季同比"], "季度数据")
    validate_required_columns(quar_date, ["Unnamed: 0", "GDP同比"], "季度发布日期")
    quar_data = ensure_datetime_column(quar_data.copy(), "Unnamed: 0", "季度数据")
    quar_date = ensure_datetime_column(quar_date.copy(), "Unnamed: 0", "季度发布日期")
    quar_data = quar_data.sort_values(by="Unnamed: 0", ascending=True).set_index("Unnamed: 0")
    result = pd.DataFrame(index=quar_data.index, columns=quar_data.columns, dtype=np.float64)
    target = "中国:GDP:不变价:当季同比"
    # 累进窗口标准化：第 i 期只用 [0..i] 的数据（避免未来信息泄露）
    for i in range(1, len(quar_data)):
        window_data = quar_data[target].iloc[0:i + 1].astype(np.float64)
        median = window_data.median()
        iqr = window_data.quantile(0.75) - window_data.quantile(0.25)
        if pd.isna(iqr) or iqr == 0:
            result.loc[quar_data.index[i], target] = np.nan
        else:
            result.loc[quar_data.index[i], target] = (quar_data[target].iloc[i] - median) / iqr
    quar_data_std = result.reset_index()
    # 用发布日期做索引，为后续 merge_asof 前向匹配做准备
    quar_merge = pd.merge(quar_data_std, quar_date, on="Unnamed: 0", how="left")
    if "GDP同比" not in quar_merge.columns:
        raise ValueError("季度发布日期缺少 GDP同比 列")
    # 将 GDP同比 发布日强制转换为 datetime，避免索引中 str 与 datetime 混合导致 sort_index 失败
    quar_merge["GDP同比"] = pd.to_datetime(quar_merge["GDP同比"], errors="coerce")
    quar_merge = quar_merge.dropna(subset=["GDP同比"]).set_index("GDP同比")
    return quar_merge


# ================================================================================
# 月度数据合并
#
# 核心问题：月度指标有"发布日滞后"
#   例如 1 月 CPI 可能 2 月中旬才发布，模型在 1 月 31 日时不应该看到它。
#
# 处理方式：
#   - 每列月频数据在 month_date 中都对应一个发布日列
#   - 用 merge_asof 做前向匹配：日频日期 >= 发布日期时，该月数据才"可用"
#   - 逐列循环处理，因为不同指标的发布日可能不同
# ================================================================================

def merge_monthly(merge_base, month_data, month_date):
    """
    将月频数据按各自发布日做前向合并到日频基准

    参数:
        merge_base: 已有的日频 DataFrame（日期索引）
        month_data: 月频数据
        month_date: 月度数据发布日期表（每列月频数据对应一列发布日）
    返回:
        合并后的 DataFrame
    """
    # 月频数据滚动鲁棒标准化，窗口 12 个月
    month_data_std = apply_rolling_robust_std_to_df(month_data, 12)
    month_date = month_date.copy()
    month_date = ensure_datetime_column(month_date, "Unnamed: 0", "月度发布日期")
    # 将所有发布日列转为 datetime
    for col in month_date.columns:
        if col != "Unnamed: 0":
            month_date[col] = pd.to_datetime(month_date[col], errors="coerce")
    month_merge = pd.merge(
        month_data_std.reset_index(),
        month_date,
        on="Unnamed: 0",
        how="left"
    ).set_index("Unnamed: 0")
    if month_merge.empty:
        return merge_base
    month_merge = month_merge.sort_index()
    # n_features 为月频数据列数，后 n_features 列为对应的发布日列
    n_features = month_data_std.shape[1]
    total_cols = len(month_merge.columns)
    n_pairs = min(n_features, total_cols - n_features)
    if n_pairs <= 0:
        raise ValueError("月度数据列与发布日期列不匹配")
    merge_result = merge_base.sort_index()
    # 逐列按发布日做前向匹配
    for i in range(n_pairs):
        right_key = month_merge.columns[i + n_features]  # 发布日列
        col_to_merge = month_merge.columns[i]            # 月频数据列
        right_df = month_merge[[right_key, col_to_merge]].dropna(subset=[right_key]).sort_values(by=right_key)
        if right_df.empty:
            continue
        merge_result = pd.merge_asof(
            merge_result,
            right_df,
            left_index=True,
            right_on=right_key,
            direction="forward"        # 前向匹配：日频日期能看到的最近发布数据
        ).drop(columns=[right_key])
        merge_result = merge_result.set_index(merge_base.index)
    return merge_result


# ================================================================================
# 利率动量特征计算
#
# 计算多个时间窗口的动量（利率变化量），再做滚动鲁棒标准化。
#
# 动量窗口含义：
#   - 1天、7天：短期趋势
#   - 30天、60天、180天：中长期趋势
#
# 标准化窗口：
#   - 短期动量用 252 天标准化窗口（约 1 年），min_periods=66（约 3 个月）
#   - 长期动量用 756 天标准化窗口（约 3 年），min_periods=188（约 9 个月）
#
# 公式：动量 = rate(t) - rate(t - window)
#       标准化动量 = (动量 - 滚动中位数) / 滚动IQR
# ================================================================================

def preprocess_rate_data(rate_path):
    """
    读取原始利率数据，计算多窗口动量特征并标准化

    输出列：
        rate                        — 原始利率
        momentum_1_robust_scaled    — 1天动量（鲁棒标准化）
        momentum_7_robust_scaled    — 7天动量
        momentum_30_robust_scaled   — 30天动量
        momentum_60_robust_scaled   — 60天动量
        momentum_180_robust_scaled  — 180天动量
    """
    if not os.path.exists(rate_path):
        raise FileNotFoundError(f"未找到利率文件: {rate_path}")
    df = pd.read_csv(rate_path)
    if df.shape[1] < 2:
        raise ValueError("利率数据至少包含两列")
    df = df.iloc[:, 0:2].copy()
    df.columns = ["time", "rate"]
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    # 短期动量参数
    short_windows = [1, 7]
    # 长期动量参数
    long_windows = [30, 60, 180]
    # 短期动量标准化窗口：252 个交易日（约 1 年），最小观测 66 天（约 3 个月）
    short_window_size = 252
    short_min_periods = 66
    # 长期动量标准化窗口：756 个交易日（约 3 年），最小观测 188 天（约 9 个月）
    long_window_size = 756
    long_min_periods = 188

    def calculate_momentum(data_frame, window, rolling_window_size, min_periods):
        """
        计算单窗口动量并进行滚动鲁棒标准化

        参数:
            data_frame: 含 rate 列的 DataFrame
            window: 动量的时间窗口（如 1=日变化, 7=周变化）
            rolling_window_size: 标准化用的滚动窗口大小
            min_periods: 滚动窗口最少观测数
        """
        momentum = data_frame["rate"].diff(window)
        rolling_median = momentum.rolling(window=rolling_window_size, min_periods=min_periods).median()
        rolling_iqr = (
            momentum.rolling(window=rolling_window_size, min_periods=min_periods).quantile(0.75) -
            momentum.rolling(window=rolling_window_size, min_periods=min_periods).quantile(0.25)
        )
        return (momentum - rolling_median) / rolling_iqr

    # 计算短期动量
    for window in short_windows:
        df[f"momentum_{window}_robust_scaled"] = calculate_momentum(
            df, window, short_window_size, short_min_periods
        )
    # 计算长期动量
    for window in long_windows:
        df[f"momentum_{window}_robust_scaled"] = calculate_momentum(
            df, window, long_window_size, long_min_periods
        )
    return df


# ================================================================================
# 主预处理流程
#
# 按频率从高到低逐层合并：
#   日频 → +周频 → +季频(GDP) → +月频 → +特殊频率(政策利率) → +分类变量
#
# 合并统一使用 merge_asof + direction="forward"：
#   左表日期 >= 右表日期时，取最近一条右表数据，保证不看到未来。
# ================================================================================

def preprocess_dataset(
    file_name,
    output_data_path,
    output_rate_path,
    rate_source_path="data/43_origin_rate.csv",
    checkpoint_dir="data/43_preprocess_checkpoints"
):
    """
    执行完整的预处理流水线

    参数:
        file_name:         输入 Excel 路径（含 8 个 sheet）
        output_data_path:  输出合并数据 CSV 路径
        output_rate_path:  输出利率动量 CSV 路径
        rate_source_path:  原始利率 CSV 路径
        checkpoint_dir:    检查点目录，中途中断后可从此目录继续
    返回:
        (merge5_filled_total, rate_df)
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"未找到数据文件: {file_name}")
    pd.options.mode.string_storage = "python"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_state_path = os.path.join(checkpoint_dir, "checkpoint_state.json")
    current_state = build_checkpoint_state(file_name, rate_source_path)
    saved_state = load_checkpoint_state(checkpoint_state_path)
    if saved_state != current_state:
        LOGGER.info("检测到输入文件或版本变化，重建检查点目录: %s", checkpoint_dir)
        clear_stale_checkpoints(checkpoint_dir)
        save_checkpoint_state(checkpoint_state_path, current_state)
    else:
        LOGGER.info("检查点状态有效，尝试从已完成步骤继续")

    # ===== 步骤1：读取 8 个 sheet =====
    sheet_cache = load_or_run_checkpoint(
        "sheet_cache",
        checkpoint_dir,
        lambda: {
            "quar_data": safe_read_excel(file_name, sheet_name=0, name="季度数据"),
            "quar_date": safe_read_excel(file_name, sheet_name=1, name="季度发布日期"),
            "month_data": safe_read_excel(file_name, sheet_name=2, name="月度数据"),
            "month_date": safe_read_excel(file_name, sheet_name=3, name="月度发布日期"),
            "week_data": safe_read_excel(file_name, sheet_name=4, name="周度数据"),
            "daily_data": safe_read_excel(file_name, sheet_name=5, name="日度数据"),
            "spec_data": safe_read_excel(file_name, sheet_name=6, name="特殊频率数据"),
            "cate_data": safe_read_excel(file_name, sheet_name=7, name="分类时代变量")
        }
    )
    quar_data = sheet_cache["quar_data"]
    quar_date = sheet_cache["quar_date"]
    month_data = sheet_cache["month_data"]
    month_date = sheet_cache["month_date"]
    week_data = sheet_cache["week_data"]
    daily_data = sheet_cache["daily_data"]
    spec_data = sheet_cache["spec_data"]
    cate_data = sheet_cache["cate_data"]
    daily_data = daily_data.bfill()  # 日频数据缺失值后向填充（少量缺失用后一日补）

    # ===== 步骤1续：各频率数据滚动鲁棒标准化 =====
    quar_merge = load_or_run_checkpoint(
        "quarterly_standardized",
        checkpoint_dir,
        lambda: standardize_quarterly(quar_data, quar_date)
    )  # 季频：累进窗口
    week_data_std = load_or_run_checkpoint(
        "weekly_standardized",
        checkpoint_dir,
        lambda: apply_rolling_robust_std_to_df(week_data, 52)
    )  # 周频：52周窗口
    daily_data_std = load_or_run_checkpoint(
        "daily_standardized",
        checkpoint_dir,
        lambda: apply_rolling_robust_std_to_df(daily_data, 252)
    )  # 日频：252天窗口

    # ===== 步骤2：日频 + 周频前向合并 =====
    merge1 = load_or_run_checkpoint(
        "merge_daily_weekly",
        checkpoint_dir,
        lambda: pd.merge_asof(
            daily_data_std.sort_index(),
            week_data_std.sort_index(),
            left_index=True,
            right_index=True,
            direction="forward"
        )
    )

    # ===== 步骤3：合并季度 GDP 数据（按发布日索引前向匹配） =====
    merge2 = load_or_run_checkpoint(
        "merge_quarterly_gdp",
        checkpoint_dir,
        lambda: pd.merge_asof(
            merge1.sort_index(),
            quar_merge["中国:GDP:不变价:当季同比"].sort_index(),
            left_index=True,
            right_index=True,
            direction="forward"   # 日频日期 >= GDP 发布日时，获取最新已发布的 GDP
        )
    )

    # ===== 步骤4：合并月频数据（每列独立发布日 + 前向匹配） =====
    merge3 = load_or_run_checkpoint(
        "merge_monthly_data",
        checkpoint_dir,
        lambda: merge_monthly(merge2, month_data, month_date)
    )

    # ===== 步骤5：合并特殊频率数据（政策利率变动标记） =====
    # 逆回购利率和存准率不是定期变动的，关键不是当前值而是"是否变了"
    # 处理方式：变动时保留原值，不变时置 0
    merge4 = load_or_run_checkpoint(
        "merge_special_frequency",
        checkpoint_dir,
        lambda: _merge_special_frequency(merge3, spec_data)
    )

    # ===== 步骤6：分类变量布尔化（One-Hot 编码） =====
    # sheet 7 包含 3 列分类变量，分别有 2/5/4 个类别
    # 如：牛熊阶段、政策周期等——转为 0/1 布尔列，供模型使用
    merge5 = load_or_run_checkpoint(
        "merge_category_data",
        checkpoint_dir,
        lambda: _merge_category_data(merge4, cate_data)
    )

    # ===== 步骤7：缺失值处理 =====
    # 策略：
    #   - 前向填充(ffill)：低频数据在发布后维持不变直到下次更新
    #     （如季度 GDP 发布后，直到下一季度前都沿用当前值）
    #   - 剩余缺失填 99.00：作为特殊标记值，让模型能识别"从未有数据的特征"
    merge5_filled_total = load_or_run_checkpoint(
        "fill_missing_and_dates",
        checkpoint_dir,
        lambda: _finalize_merged_dataset(merge5)
    )

    # ===== 步骤8：日期列表化 =====
    # 输出主数据
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    merge5_filled_total.to_csv(output_data_path)

    # ===== 步骤9：利率动量特征计算 =====
    rate_df = load_or_run_checkpoint(
        "rate_features",
        checkpoint_dir,
        lambda: preprocess_rate_data(rate_source_path)
    )
    os.makedirs(os.path.dirname(output_rate_path), exist_ok=True)
    rate_df.to_csv(output_rate_path, index=False)

    return merge5_filled_total, rate_df


def _merge_special_frequency(merge3, spec_data):
    """处理特殊频率数据，并合并到主表"""
    validate_required_columns(spec_data, ["Unnamed: 0", "逆回购利率:7天", "存款准备金率:大型金融机构(变动公告日期)"], "特殊频率数据")
    spec_data = ensure_datetime_column(spec_data.copy(), "Unnamed: 0", "特殊频率数据")
    # 如果逆回购利率值与下一期相同 → 置 0（未变动），否则保留原值
    spec_data["A_modified"] = spec_data["逆回购利率:7天"].where(
        spec_data["逆回购利率:7天"] != spec_data["逆回购利率:7天"].shift(-1), 0
    )
    # 同上逻辑处理存款准备金率
    spec_data["B_modified"] = spec_data["存款准备金率:大型金融机构(变动公告日期)"].where(
        spec_data["存款准备金率:大型金融机构(变动公告日期)"] != spec_data["存款准备金率:大型金融机构(变动公告日期)"].shift(-1), 0
    )
    spec_data = spec_data.sort_values(by="Unnamed: 0")
    merged = pd.merge_asof(
        merge3.sort_index(),
        spec_data[["Unnamed: 0", "A_modified", "B_modified"]].sort_values(by="Unnamed: 0"),
        left_index=True,
        right_on="Unnamed: 0",
        direction="forward"
    ).drop(columns=["Unnamed: 0"])
    return merged.set_index(merge3.index)


def _merge_category_data(merge4, cate_data):
    """对分类变量做独热编码，并合并到主表"""
    validate_required_columns(cate_data, ["Unnamed: 0"], "分类时代变量")
    if len(cate_data.columns) < 4:
        raise ValueError("分类时代变量列数不足")
    cate_data = ensure_datetime_column(cate_data.copy(), "Unnamed: 0", "分类时代变量")
    col_1 = cate_data.columns[1]  # 第1个分类变量（2类）
    col_2 = cate_data.columns[2]  # 第2个分类变量（5类）
    col_3 = cate_data.columns[3]  # 第3个分类变量（4类）
    # One-Hot 编码：每类一列，属于该类为 1，否则为 0
    for i in range(1, 3):
        cate_data[f"{col_1}-{i}"] = (cate_data[col_1] == i).astype(int)
    for i in range(1, 6):
        cate_data[f"{col_2}-{i}"] = (cate_data[col_2] == i).astype(int)
    for i in range(1, 5):
        cate_data[f"{col_3}-{i}"] = (cate_data[col_3] == i).astype(int)
    # 删除原始分类列，保留独热编码列
    cate_data = cate_data.drop(columns=[col_1, col_2, col_3]).sort_values(by="Unnamed: 0")
    merged = pd.merge_asof(
        merge4.sort_index(),
        cate_data.sort_values(by="Unnamed: 0"),
        left_index=True,
        right_on="Unnamed: 0",
        direction="forward"
    ).drop(columns=["Unnamed: 0"])
    return merged.set_index(merge4.index)


def _finalize_merged_dataset(merge5):
    """完成最终缺失值处理，并补充日期列表特征"""
    merge5_filled = merge5.ffill()
    merge5_filled_total = merge5_filled.fillna(99.00)
    if not isinstance(merge5_filled_total.index, pd.DatetimeIndex):
        raise ValueError("输出索引不是日期类型")
    merge5_filled_total["date_list"] = merge5_filled_total.index.to_series().apply(
        lambda x: [x.year, x.month, x.day]
    )
    return merge5_filled_total


def main():
    setup_logging()
    file_name = "data/43_origin_data.xlsx"
    output_data_path = "data/43_data.csv"
    output_rate_path = "data/43_rate.csv"
    checkpoint_dir = "data/43_preprocess_checkpoints"
    try:
        preprocess_dataset(
            file_name,
            output_data_path,
            output_rate_path,
            rate_source_path="data/43_origin_rate.csv",
            checkpoint_dir=checkpoint_dir
        )
        LOGGER.info("预处理完成: %s, %s", output_data_path, output_rate_path)
    except Exception:
        LOGGER.exception("预处理失败")
        raise


if __name__ == "__main__":
    main()
