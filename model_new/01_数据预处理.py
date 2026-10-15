# -*- coding: utf-8 -*-
"""
华泰证券债券预测 —— 新数据集预处理流水线（2026版）

================================================================================
基于 43_1_华泰证券债券预测_数据集预处理.py 改造，适配新的 Excel 数据结构。

与原版的核心区别：
  1. Excel 从 8 sheet → 10 sheet（新增"时代变量整理明细"和"切券时间"）
  2. 列名大量简化（去掉"前一个观测值"等后缀）
  3. 新增多列日频特征（美债、美股、商品、净买入等）
  4. 利率 CSV 倒序排列且含尾部空格，需特殊处理
  5. 月频数据列(19)与发布日期列(17)不再一一对应，需显式映射

================================================================================
输入：
  data_ori/2026_origin_data.xlsx（10 个 sheet）
  data_ori/2026_origin_rate.csv（利率，倒序）

输出：
  model_new/data/data.csv  — 合并后的日频宽表
  model_new/data/rate.csv  — 利率 + 动量特征
================================================================================
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# =====================================================================
# 工具函数
# =====================================================================

def setup_logging():
    """配置日志格式"""
    if LOGGER.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )


def ensure_datetime_column(df, col, name):
    """将指定列转为 datetime 类型"""
    df[col] = pd.to_datetime(df[col], errors="coerce")
    if df[col].isna().all():
        raise ValueError(f"{name} 的 {col} 列无法解析为日期")
    return df


def safe_read_excel(file_name, sheet_name, name):
    """读取 Excel sheet，优先 openpyxl 引擎"""
    try:
        return pd.read_excel(file_name, sheet_name=sheet_name, engine="openpyxl")
    except Exception as exc:
        LOGGER.warning("读取 %s 失败，尝试降级: %s", name, exc)
        return pd.read_excel(file_name, sheet_name=sheet_name)


# =====================================================================
# 滚动鲁棒标准化
#
# 公式：标准化值 = (当前值 - 滚动窗口中位数) / 滚动窗口 IQR
# 使用中位数/IQR 而非均值/标准差，抵抗金融数据极端值
# 使用滚动窗口而非全局统计，避免未来信息泄露
# =====================================================================

def rolling_robust_standardize(arr, window):
    """
    对一维数组做滚动鲁棒标准化

    参数:
        arr: 原始数据
        window: 滚动窗口大小
    返回:
        标准化后的数组（前 window-1 个位置为 NaN）
    """
    if window <= 0:
        raise ValueError("window 必须为正整数")
    series = pd.to_numeric(pd.Series(arr), errors="coerce")
    arr = series.to_numpy(dtype=np.float64)
    if arr.size < window:
        return np.full(arr.shape, np.nan, dtype=np.float64)

    window_view = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window)
    # 抑制全 NaN 窗口的 RuntimeWarning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rolling_median = np.nanmedian(window_view, axis=1)
        rolling_q1 = np.nanpercentile(window_view, 25, axis=1)
        rolling_q3 = np.nanpercentile(window_view, 75, axis=1)
    rolling_iqr = rolling_q3 - rolling_q1
    rolling_iqr[rolling_iqr == 0] = np.nan  # IQR=0 时避免除零
    standardized = (arr[window - 1:] - rolling_median) / rolling_iqr
    result = np.full(arr.shape, np.nan, dtype=np.float64)
    result[window - 1:] = standardized
    return result


def apply_rolling_robust_std(df_origin, window, date_col="Unnamed: 0"):
    """对 DataFrame 的每个数值列做滚动鲁棒标准化"""
    df = df_origin.copy()
    df = ensure_datetime_column(df, date_col, "数据表")
    df = df.sort_values(by=date_col, ascending=True).set_index(date_col)
    result = pd.DataFrame(index=df.index)
    for col in df.columns:
        result[col] = rolling_robust_standardize(df[col].to_numpy(), window)
    return result


# =====================================================================
# 季度数据标准化（累进窗口 + 发布日对齐）
# =====================================================================

def standardize_quarterly(quar_data, quar_date):
    """
    对季度数据做累进鲁棒标准化（第 i 期只用 [0..i] 的数据），
    然后按发布日期做索引对齐，避免未来信息泄露。

    参数:
        quar_data: 季度数据（GDP、产能利用率等）
        quar_date: 对应的发布日期表
    返回:
        以发布日期为索引的标准化季度数据
    """
    quar_data = ensure_datetime_column(quar_data.copy(), "Unnamed: 0", "季度数据")
    quar_date = ensure_datetime_column(quar_date.copy(), "Unnamed: 0", "季度发布日期")
    quar_data = quar_data.sort_values(by="Unnamed: 0", ascending=True).set_index("Unnamed: 0")
    quar_date = quar_date.sort_values(by="Unnamed: 0", ascending=True)

    # 累进窗口标准化所有数值列
    result = pd.DataFrame(index=quar_data.index, dtype=np.float64)
    data_cols = [c for c in quar_data.columns if c != "Unnamed: 0"]
    for col in data_cols:
        for i in range(1, len(quar_data)):
            window_data = quar_data[col].iloc[0:i + 1].astype(np.float64)
            median = window_data.median()
            iqr = window_data.quantile(0.75) - window_data.quantile(0.25)
            if pd.isna(iqr) or iqr == 0:
                result.loc[quar_data.index[i], col] = np.nan
            else:
                result.loc[quar_data.index[i], col] = (quar_data[col].iloc[i] - median) / iqr

    # 合并发布日期，以发布日为索引
    quar_merge = result.reset_index().merge(quar_date, on="Unnamed: 0", how="left")
    # 发布日期列（非 "Unnamed: 0" 且非数据列）
    publish_cols = [c for c in quar_date.columns if c != "Unnamed: 0"]
    # 取第一个非空的发布日期列作为索引
    # 多列发布日时，取每行最早可用的发布日期
    for pc in publish_cols:
        quar_merge[pc] = pd.to_datetime(quar_merge[pc], errors="coerce")

    # 使用每行最大发布日期作为索引（所有指标发布完毕的日期）
    quar_merge["publish_date"] = quar_merge[publish_cols].max(axis=1)
    quar_merge = quar_merge.dropna(subset=["publish_date"]).set_index("publish_date")
    # 保留标准化后的数据列
    quar_merge = quar_merge[data_cols]
    return quar_merge


# =====================================================================
# 月度数据合并（显式列名映射 + 发布日前向匹配）
# =====================================================================

# 月频数据列 → 发布日期列的映射
# 键 = 月频数据列名，值 = 发布日期列名；None = 无独立发布日（用月末）
MONTHLY_PUBLISH_MAP = {
    "工业增加值同比": "工业增加值同比",
    "社会消费品零售总额同比": "社会消费品零售总额同比",
    "固定资产投资完成额:基础设施建设:累计同比": "固定资产投资完成额:基础设施建设:累计同比",
    "房地产开发投资完成额:累计同比": "房地产开发投资完成额:累计同比",
    "CPI同比": "CPI同比",
    "PPI同比": "PPI同比",
    "M2同比": "M2同比",
    "PMI-50": "PMI-50",
    "出口金额:当月同比": "出口金额",
    "进口金额:当月同比": "进口金额",
    "商品房销售面积:累计同比": None,
    "70个大中城市:房屋销售价格指数:新建商品住宅:当月同比": None,
    "社融同比": "社融同比",
    "M1旧口径": "M1同比",
    "商业银行托管同比": "商业银行托管同比",
    "广义基金托管同比": "广义基金托管同比",
    "证券公司托管同比": "证券公司托管同比",
    "保险托管同比": "保险托管同比",
}


def merge_monthly(merge_base, month_data, month_date):
    """
    将月频数据按各自发布日做前向合并到日频基准。

    改进：使用显式列名映射（MONTHLY_PUBLISH_MAP）替代原版的位置匹配，
    避免数据列(19)与发布日期列(17)数量不一致导致的错位。
    """
    # 月频数据滚动鲁棒标准化，窗口 12 个月
    month_data_std = apply_rolling_robust_std(month_data, 12)
    month_date = ensure_datetime_column(month_date.copy(), "Unnamed: 0", "月度发布日期")
    # 发布日期列全部转 datetime
    for col in month_date.columns:
        if col != "Unnamed: 0":
            month_date[col] = pd.to_datetime(month_date[col], errors="coerce")

    merge_result = merge_base.sort_index()

    for data_col, publish_col in MONTHLY_PUBLISH_MAP.items():
        if data_col not in month_data_std.columns:
            continue

        if publish_col and publish_col in month_date.columns:
            # 有独立发布日：用发布日做前向匹配
            tmp = month_data_std[[data_col]].reset_index().copy()
            tmp["publish_date"] = month_date.set_index("Unnamed: 0")[publish_col].reindex(
                month_data_std.index
            ).values
            tmp = tmp.dropna(subset=["publish_date"]).sort_values("publish_date")
            if tmp.empty:
                continue
            merge_result = pd.merge_asof(
                merge_result,
                tmp[["publish_date", data_col]],
                left_index=True,
                right_on="publish_date",
                direction="forward"
            ).drop(columns=["publish_date"])
            merge_result = merge_result.set_index(merge_base.index)
        else:
            # 无独立发布日：用数据所属月末作为发布日
            tmp = month_data_std[[data_col]].copy()
            tmp.index = tmp.index + pd.offsets.MonthEnd(0)  # 月末
            merge_result = pd.merge_asof(
                merge_result,
                tmp,
                left_index=True,
                right_index=True,
                direction="forward"
            )

    return merge_result


# =====================================================================
# 特殊频率数据处理（政策利率变动标记）
# =====================================================================

def merge_special_frequency(merge_base, spec_data):
    """
    处理逆回购利率和存款准备金率，生成变动标记。

    逻辑：当利率/准备金率与下一期不同时（即发生了变动），
    保留原值；否则置 0。这样模型能识别"政策调整事件"。

    新数据区别：准备金率列名为"中小金融机构"而非"大型金融机构"。
    """
    spec_data = ensure_datetime_column(spec_data.copy(), "Unnamed: 0", "特殊频率数据")
    spec_data = spec_data.sort_values(by="Unnamed: 0").reset_index(drop=True)

    # 逆回购利率变动标记
    rev_col = "逆回购利率:7天"
    if rev_col in spec_data.columns:
        spec_data["A_modified"] = spec_data[rev_col].where(
            spec_data[rev_col] != spec_data[rev_col].shift(-1), 0
        )
    else:
        spec_data["A_modified"] = 0.0

    # 存款准备金率变动标记
    rr_col = "存款准备金率:中小金融机构(变动公告日期)"
    if rr_col not in spec_data.columns:
        # 回退：找包含"存款准备金率"的列
        rr_candidates = [c for c in spec_data.columns if "存款准备金率" in c]
        rr_col = rr_candidates[0] if rr_candidates else None
    if rr_col:
        spec_data["B_modified"] = spec_data[rr_col].where(
            spec_data[rr_col] != spec_data[rr_col].shift(-1), 0
        )
    else:
        spec_data["B_modified"] = 0.0

    merged = pd.merge_asof(
        merge_base.sort_index(),
        spec_data[["Unnamed: 0", "A_modified", "B_modified"]].sort_values(by="Unnamed: 0"),
        left_index=True,
        right_on="Unnamed: 0",
        direction="forward"
    ).drop(columns=["Unnamed: 0"])
    return merged.set_index(merge_base.index)


# =====================================================================
# 时代变量独热编码
# =====================================================================

def merge_category_data(merge_base, cate_data):
    """
    对时代变量做独热编码并合并到主表。

    新数据有 3 列分类变量：
    - 中国各领导人时期（2 类）
    - 美国各总统时期（5 类）
    - 基本面主题变量（4 类）
    """
    cate_data = ensure_datetime_column(cate_data.copy(), "Unnamed: 0", "时代变量")
    cate_data = cate_data.sort_values(by="Unnamed: 0").reset_index(drop=True)

    # 获取分类列（排除日期列）
    cat_cols = [c for c in cate_data.columns if c != "Unnamed: 0"]

    for col_idx, col in enumerate(cat_cols):
        max_val = int(cate_data[col].dropna().max())
        for k in range(1, max_val + 1):
            cate_data[f"{col}-{k}"] = (cate_data[col] == k).astype(int)

    # 删除原始分类列
    cate_data = cate_data.drop(columns=cat_cols)
    merged = pd.merge_asof(
        merge_base.sort_index(),
        cate_data.sort_values(by="Unnamed: 0"),
        left_index=True,
        right_on="Unnamed: 0",
        direction="forward"
    ).drop(columns=["Unnamed: 0"])
    return merged.set_index(merge_base.index)


# =====================================================================
# 利率动量特征计算
# =====================================================================

def preprocess_rate_data(rate_path):
    """
    读取原始利率数据，计算多窗口动量特征并标准化。

    新数据区别：
    - 利率 CSV 倒序排列（最新在前），需升序排序
    - 利率值含尾部空格，需 strip 后转 float
    - 列名为"日期"和"中国国债收益率10年"

    输出列：
        rate, momentum_{1,7,30,60,180}_robust_scaled
    """
    df = pd.read_csv(rate_path)
    # 取前两列，重命名为标准格式
    df = df.iloc[:, 0:2].copy()
    df.columns = ["time", "rate"]
    # 清理利率值中的尾部空格并转数值
    df["rate"] = pd.to_numeric(df["rate"].astype(str).str.strip(), errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    # 升序排列（新数据为倒序）
    df = df.sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=["rate"])

    # 动量窗口参数
    short_windows = [1, 7]    # 短期：日变化、周变化
    long_windows = [30, 60, 180]  # 中长期
    # 标准化窗口
    short_win, short_min = 252, 66    # ~1年，最少~3个月
    long_win, long_min = 756, 188     # ~3年，最少~9个月

    def calc_momentum(window, rolling_win, min_periods):
        """计算单窗口动量并做滚动鲁棒标准化"""
        momentum = df["rate"].diff(window)
        med = momentum.rolling(rolling_win, min_periods=min_periods).median()
        q1 = momentum.rolling(rolling_win, min_periods=min_periods).quantile(0.25)
        q3 = momentum.rolling(rolling_win, min_periods=min_periods).quantile(0.75)
        iqr = q3 - q1
        return (momentum - med) / iqr

    for w in short_windows:
        df[f"momentum_{w}_robust_scaled"] = calc_momentum(w, short_win, short_min)
    for w in long_windows:
        df[f"momentum_{w}_robust_scaled"] = calc_momentum(w, long_win, long_min)

    return df


# =====================================================================
# 检查点机制（中途中断后可续跑）
# =====================================================================

def load_or_run_checkpoint(step_name, checkpoint_dir, builder):
    """带检查点执行步骤：有缓存则加载，否则执行 builder 并缓存"""
    checkpoint_path = os.path.join(checkpoint_dir, f"{step_name}.pkl")
    if os.path.exists(checkpoint_path):
        LOGGER.info("检查点命中，跳过: %s", step_name)
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)
    LOGGER.info("开始执行: %s", step_name)
    result = builder()
    with open(checkpoint_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.info("完成: %s", step_name)
    return result


# =====================================================================
# 主预处理流程
# =====================================================================

def preprocess_dataset(
    excel_path,
    rate_path,
    output_data_path,
    output_rate_path,
    checkpoint_dir,
):
    """
    执行完整预处理流水线。

    合并顺序（从高频到低频）：
      日频 → +周频 → +季频(GDP) → +月频 → +特殊频率 → +时代变量
    所有合并使用 merge_asof + direction="forward"，保证不看到未来。
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

    # ===== 步骤1：读取 10 个 sheet =====
    LOGGER.info("读取 Excel: %s", excel_path)
    sheets = {}
    sheet_names = [
        "quar_data", "quar_date", "month_data", "month_date",
        "week_data", "daily_data", "spec_data", "cate_data"
    ]
    for i, name in enumerate(sheet_names):
        sheets[name] = safe_read_excel(excel_path, sheet_name=i, name=name)

    daily_data = sheets["daily_data"]
    daily_data = daily_data.bfill()  # 日频少量缺失后向填充

    # ===== 步骤2：各频率滚动鲁棒标准化 =====
    quar_std = load_or_run_checkpoint(
        "quarterly_standardized", checkpoint_dir,
        lambda: standardize_quarterly(sheets["quar_data"], sheets["quar_date"])
    )
    week_std = load_or_run_checkpoint(
        "weekly_standardized", checkpoint_dir,
        lambda: apply_rolling_robust_std(sheets["week_data"], 52)
    )
    daily_std = load_or_run_checkpoint(
        "daily_standardized", checkpoint_dir,
        lambda: apply_rolling_robust_std(daily_data, 252)
    )

    # ===== 步骤3：日频 + 周频前向合并 =====
    merge1 = load_or_run_checkpoint(
        "merge_daily_weekly", checkpoint_dir,
        lambda: pd.merge_asof(
            daily_std.sort_index(),
            week_std.sort_index(),
            left_index=True, right_index=True,
            direction="forward"
        )
    )

    # ===== 步骤4：合并季度数据 =====
    merge2 = load_or_run_checkpoint(
        "merge_quarterly", checkpoint_dir,
        lambda: pd.merge_asof(
            merge1.sort_index(),
            quar_std.sort_index(),
            left_index=True, right_index=True,
            direction="forward"
        )
    )

    # ===== 步骤5：合并月频数据 =====
    merge3 = load_or_run_checkpoint(
        "merge_monthly", checkpoint_dir,
        lambda: merge_monthly(merge2, sheets["month_data"], sheets["month_date"])
    )

    # ===== 步骤6：合并特殊频率 =====
    merge4 = load_or_run_checkpoint(
        "merge_special", checkpoint_dir,
        lambda: merge_special_frequency(merge3, sheets["spec_data"])
    )

    # ===== 步骤7：时代变量独热编码 =====
    merge5 = load_or_run_checkpoint(
        "merge_category", checkpoint_dir,
        lambda: merge_category_data(merge4, sheets["cate_data"])
    )

    # ===== 步骤8：缺失值处理 =====
    # 前向填充 + 余缺填 99.00
    merge5_filled = merge5.ffill().fillna(99.00)

    # 补充日期列表特征
    if isinstance(merge5_filled.index, pd.DatetimeIndex):
        merge5_filled["date_list"] = merge5_filled.index.to_series().apply(
            lambda x: [x.year, x.month, x.day]
        )

    # 输出主数据
    merge5_filled.to_csv(output_data_path)
    LOGGER.info("输出主数据: %s (%d 行, %d 列)", output_data_path, *merge5_filled.shape)

    # ===== 步骤9：利率动量特征 =====
    rate_df = preprocess_rate_data(rate_path)
    rate_df.to_csv(output_rate_path, index=False)
    LOGGER.info("输出利率数据: %s (%d 行)", output_rate_path, len(rate_df))

    return merge5_filled, rate_df


def main():
    setup_logging()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 相对于 model_new 目录的路径
    project_root = os.path.dirname(base_dir)

    excel_path = os.path.join(project_root, "data_ori", "2026_origin_data.xlsx")
    rate_path = os.path.join(project_root, "data_ori", "2026_origin_rate.csv")
    output_data = os.path.join(base_dir, "data", "data.csv")
    output_rate = os.path.join(base_dir, "data", "rate.csv")
    checkpoint_dir = os.path.join(base_dir, "data", "checkpoints")

    try:
        preprocess_dataset(excel_path, rate_path, output_data, output_rate, checkpoint_dir)
        LOGGER.info("预处理完成")
    except Exception:
        LOGGER.exception("预处理失败")
        raise


if __name__ == "__main__":
    main()
