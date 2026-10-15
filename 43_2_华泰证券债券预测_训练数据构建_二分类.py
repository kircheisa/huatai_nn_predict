"""
10年期国债收益率未来5天方向预测 —— 二分类训练数据构建

与三分类版本的核心区别：
  标签定义从「下行超3BP / 震荡 / 上行超3BP」改为「下行 / 上行」二分类。
  - label = 1：未来5日利率下行（rate_delta_5d < 0），债市偏多
  - label = 0：未来5日利率上行（rate_delta_5d >= 0），债市偏空

  这样做的理由：在2024年以来的单边下行市场中，三分类的「震荡」类里混杂了大量
  小幅下行样本，与「偏多」方向一致却被打上不同标签，形成严重的标签噪音。
  二分类只需判断方向（涨/跌），回避了人为幅度边界（±3BP）带来的不可分问题。

输出文件（全部加 _bin 后缀，避免覆盖三分类文件）：
  data/43_train_merged_bin.csv   -> 合并后的训练底表（含 rate_delta_5d 和 label）
  data/43_x_raw_bin.csv          -> 原始训练特征
  data/43_y_bin.csv              -> 二分类标签
  data/43_x_bin.csv              -> 遗传算法生成的 10 个特征（基于二分类标签）
  data/43_x_seq_bin.npy          -> 时序样本特征
  data/43_y_seq_bin.npy          -> 时序样本标签
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# =====================================================================
# 延迟导入 gplearn（仅在运行遗传算法时需要）
# =====================================================================
def _require_gplearn():
    """
    延迟导入 gplearn，避免未安装时影响标签构建和模型训练流程。
    """
    try:
        from gplearn.functions import make_function
        from gplearn.genetic import SymbolicTransformer
    except ImportError as exc:
        raise ImportError(
            "未安装 gplearn，请先执行 `pip install gplearn` 后再运行遗传特征工程。"
        ) from exc
    return make_function, SymbolicTransformer


# =====================================================================
# 二分类标签
# =====================================================================
def gen_label(value: float) -> float:
    """
    二分类标签：只看方向，不看幅度。

    - 利率下行（value < 0）→ 1（债市偏多）
    - 利率上行（value >= 0）→ 0（债市偏空）

    与三分类的关键区别：不再设置 ±3BP 阈值，避免在单边市场中
    把同方向的样本人为切割成「偏多」和「震荡」两个类别。
    """
    if pd.isna(value):
        return np.nan
    return 1 if value < 0 else 0


# =====================================================================
# 数据合并与底表构建
# =====================================================================
def build_training_dataset(data_file: Path, rate_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    构建监督学习底表。

    步骤：
    1. 读取宏观/流动性特征表（43_data.csv）和利率动量特征表（43_rate.csv）
    2. 以日期为键合并
    3. 构造 future 5d 利率变化 rate_delta_5d
    4. 生成二分类标签 label（1=下行, 0=上行）
    5. 输出原始特征 x_train 与标签 y_train
    """
    df_data = pd.read_csv(data_file)
    df_rate = pd.read_csv(rate_file)

    df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"], errors="coerce")
    df_rate["time"] = pd.to_datetime(df_rate["time"], errors="coerce")
    df_rate = df_rate.iloc[:, 0:].copy()

    merged = pd.merge(
        df_data,
        df_rate,
        left_on="Unnamed: 0",
        right_on="time",
        how="inner",
    ).sort_values("Unnamed: 0")

    # 目标变量：未来 5 天利率变化（正=上行，负=下行）
    merged["target_date_5d"] = merged["Unnamed: 0"].shift(-5)
    merged["rate_delta_5d"] = merged["rate"].shift(-5) - merged["rate"]
    # 二分类标签：下行=1, 上行=0
    merged["label"] = merged["rate_delta_5d"].apply(gen_label)

    # 最后 5 天没有未来标签，剔除
    merged = merged.dropna(subset=["label"]).copy()
    merged["label"] = merged["label"].astype(int)

    # 防止目标泄露：rate、label、rate_delta_5d 不能进入训练特征
    x_train = merged.drop(
        columns=["time", "rate", "label", "rate_delta_5d", "target_date_5d", "date_list"],
        errors="ignore",
    ).copy()
    x_train = x_train.set_index("Unnamed: 0")
    x_train = x_train.apply(pd.to_numeric, errors="coerce").fillna(99.00)
    y_train = merged["label"].copy()

    # 打印二分类标签分布，便于确认类别平衡情况
    print("=== 二分类标签分布 ===")
    dist = y_train.value_counts(normalize=True).sort_index()
    for lbl, ratio in dist.items():
        tag = "下行(偏多)" if lbl == 1 else "上行(偏空)"
        print(f"  label {lbl} ({tag}): {y_train.sum() if lbl == 1 else len(y_train) - y_train.sum()} ({ratio:.1%})")
    return merged, x_train, y_train


# =====================================================================
# 遗传算法函数库（与三分类版本完全一致）
# =====================================================================
def _pad_rolling_result(values: np.ndarray, window: int, full_length: int) -> np.ndarray:
    """将滚动计算结果前面补零，保持与输入等长。"""
    if full_length < window:
        return np.zeros(full_length, dtype=np.float64)
    return np.concatenate([np.zeros(window - 1, dtype=np.float64), values])


def _make_ts_delay(window: int):
    make_function, _ = _require_gplearn()
    def ts_delay(arr):
        arr = np.asarray(arr, dtype=np.float64)
        return np.roll(arr, window)
    return make_function(function=ts_delay, name=f"ts_delay_{window}", arity=1)


def _make_ts_corr(window: int):
    make_function, _ = _require_gplearn()
    def ts_corr(arr1, arr2):
        arr1 = np.asarray(arr1, dtype=np.float64)
        arr2 = np.asarray(arr2, dtype=np.float64)
        if len(arr1) < window or len(arr2) < window:
            return np.zeros_like(arr1)
        win1 = sliding_window_view(arr1, window_shape=window)
        win2 = sliding_window_view(arr2, window_shape=window)
        mean1, mean2 = np.mean(win1, axis=1), np.mean(win2, axis=1)
        std1, std2 = np.std(win1, axis=1), np.std(win2, axis=1)
        cov = np.mean(win1 * win2, axis=1) - mean1 * mean2
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov / (std1 * std2)
            corr[np.isnan(corr) | np.isinf(corr)] = 0
        return _pad_rolling_result(corr, window, len(arr1))
    return make_function(function=ts_corr, name=f"ts_corr_{window}", arity=2)


def _make_ts_cov(window: int):
    make_function, _ = _require_gplearn()
    def ts_cov(arr1, arr2):
        arr1 = np.asarray(arr1, dtype=np.float64)
        arr2 = np.asarray(arr2, dtype=np.float64)
        if len(arr1) < window or len(arr2) < window:
            return np.zeros_like(arr1)
        win1 = sliding_window_view(arr1, window_shape=window)
        win2 = sliding_window_view(arr2, window_shape=window)
        mean1, mean2 = np.mean(win1, axis=1), np.mean(win2, axis=1)
        cov = np.mean(win1 * win2, axis=1) - mean1 * mean2
        cov[np.isnan(cov) | np.isinf(cov)] = 0
        return _pad_rolling_result(cov, window, len(arr1))
    return make_function(function=ts_cov, name=f"ts_cov_{window}", arity=2)


def _make_single_series_stat(window: int, stat_name: str):
    """生成单序列滚动统计量（max/min/mean/std/zscore/decay_linear 等）。"""
    make_function, _ = _require_gplearn()
    def _calc(arr):
        arr = np.asarray(arr, dtype=np.float64)
        if len(arr) < window:
            return np.zeros_like(arr)
        windows = sliding_window_view(arr, window_shape=window)
        if stat_name == "max":       values = np.max(windows, axis=1)
        elif stat_name == "min":     values = np.min(windows, axis=1)
        elif stat_name == "maxarg":  values = np.argmax(windows, axis=1)
        elif stat_name == "minarg":  values = np.argmin(windows, axis=1)
        elif stat_name == "mean":    values = np.mean(windows, axis=1)
        elif stat_name == "median":  values = np.median(windows, axis=1)
        elif stat_name == "sum":     values = np.sum(windows, axis=1)
        elif stat_name == "stddev":  values = np.std(windows, axis=1)
        elif stat_name == "product": values = np.prod(windows, axis=1)
        elif stat_name == "kurt":
            stds = np.std(windows, axis=1)
            means = np.mean(windows, axis=1, keepdims=True)
            mu4 = np.mean((windows - means) ** 4, axis=1)
            stds[stds == 0] = 1
            values = mu4 / (stds ** 4) - 3
        elif stat_name == "skew":
            stds = np.std(windows, axis=1)
            means = np.mean(windows, axis=1, keepdims=True)
            mu3 = np.mean((windows - means) ** 3, axis=1)
            stds[stds == 0] = 1
            values = mu3 / (stds ** 3)
        elif stat_name == "zscore":
            stds = np.std(windows, axis=1, keepdims=True)
            means = np.mean(windows, axis=1, keepdims=True)
            stds[stds == 0] = 1
            values = ((windows - means) / stds)[:, -1]
        elif stat_name == "decay_linear":
            weights = np.arange(window, 0, -1, dtype=np.float64)
            weights /= np.sum(weights)
            values = np.sum(windows * weights, axis=1)
        else:
            raise ValueError(f"未知统计函数: {stat_name}")
        values[np.isnan(values) | np.isinf(values)] = 0
        return _pad_rolling_result(values, window, len(arr))
    name = f"decay_linear_{window}" if stat_name == "decay_linear" else f"ts_{stat_name}_{window}"
    return make_function(function=_calc, name=name, arity=1)


def _make_sort_cut(window: int, n: int, desc: bool):
    """滚动窗口内取最大/最小的 n 个值的均值。"""
    make_function, _ = _require_gplearn()
    def _calc(arr):
        arr = np.asarray(arr, dtype=np.float64)
        if len(arr) < window:
            return np.zeros_like(arr)
        windows = sliding_window_view(arr, window_shape=window, axis=0)
        if desc:
            values = np.mean(np.partition(windows, -n, axis=-1)[..., -n:], axis=-1)
        else:
            values = np.mean(np.partition(windows, n - 1, axis=-1)[..., :n], axis=-1)
        return _pad_rolling_result(values, window, len(arr))
    prefix = "ts_decsortcut" if desc else "ts_ascsortcut"
    return make_function(function=_calc, name=f"{prefix}_{window}_{n}", arity=1)


def _make_const(offset: int):
    make_function, _ = _require_gplearn()
    def _calc(arr):
        return np.asarray(arr, dtype=np.float64) + offset
    return make_function(function=_calc, name=f"const_{offset}", arity=1)


def build_function_set():
    """构建遗传算法函数库（ts_delay/corr/cov/max/min/mean/std/zscore/decay/sort/logical 等）。"""
    make_function, _ = _require_gplearn()
    functions = []
    # 基础滚动窗口统计
    for window in [5, 20, 60, 120]:
        functions.append(_make_ts_delay(window))
        functions.append(_make_ts_corr(window))
        functions.append(_make_ts_cov(window))
        for stat_name in [
            "max", "min", "maxarg", "minarg", "mean", "median",
            "sum", "stddev", "product", "kurt", "skew", "zscore", "decay_linear",
        ]:
            functions.append(_make_single_series_stat(window, stat_name))
    # 排序截断统计
    for window, n in zip([20, 60, 120], [5, 15, 30]):
        functions.append(_make_sort_cut(window, n, desc=False))
        functions.append(_make_sort_cut(window, n, desc=True))
    # 常数偏移
    for offset in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
        functions.append(_make_const(offset))
    # 逻辑运算符
    def _logical1(x1, x2, x3, x4): return np.where(x1 > x2, x3, x4)
    def _logical2(x1, x2):         return np.where(x1 > x2, x1, x2)
    def _logical3(x1, x2):         return np.where(x1 > x2, 1, -1)
    def _logical4(x1, x2):         return np.where(x1 > x2, 1, 0)
    functions.extend([
        "add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv",
        make_function(function=_logical1, name="logical1", arity=4),
        make_function(function=_logical2, name="logical2", arity=2),
        make_function(function=_logical3, name="logical3", arity=2),
        make_function(function=_logical4, name="logical4", arity=2),
    ])
    return functions


def run_symbolic_transformer(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    population_size: int,
    generations: int,
    n_components: int,
    max_samples: float,
    tournament_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, object]:
    """
    运行遗传算法特征工程。

    与三分类版本的区别：y_train 是二分类标签（0/1），
    SymbolicTransformer 用 pearson 相关性来筛选特征，
    二分类标签同样适用。
    """
    _, SymbolicTransformer = _require_gplearn()
    transformer = SymbolicTransformer(
        population_size=population_size,
        generations=generations,
        function_set=build_function_set(),
        n_components=n_components,
        metric="pearson",
        parsimony_coefficient=0.01,
        max_samples=max_samples,
        tournament_size=tournament_size,
        verbose=1,
        random_state=random_state,
    )
    x_new = transformer.fit_transform(x_train, y_train)
    feature_names = [f"gp_feature_{i + 1}" for i in range(x_new.shape[1])]
    x_new_df = pd.DataFrame(x_new, index=x_train.index, columns=feature_names)
    return x_new_df, transformer


# =====================================================================
# 时序样本构建（与三分类版本一致）
# =====================================================================
def build_sequence_samples(
    dataset_x: pd.DataFrame,
    dataset_y: pd.Series,
    seq_length: int,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构建时序样本：用过去 seq_length 天特征作为一个样本，
    标签取窗口最后一个时点对应的 label。
    """
    x_values = dataset_x.to_numpy(dtype=np.float32)
    y_values = dataset_y.to_numpy(dtype=np.int64)

    groups_x, groups_y = [], []
    for start in range(0, len(dataset_x) - seq_length + 1, step):
        end = start + seq_length
        groups_x.append(x_values[start:end, :])
        groups_y.append(y_values[end - 1])

    if not groups_x:
        raise ValueError("样本数量不足，无法构造时序窗口，请检查 seq_length 设置。")
    return np.asarray(groups_x, dtype=np.float32), np.asarray(groups_y, dtype=np.int64)


def save_formula_summary(transformer, output_file: Path) -> None:
    """保存遗传算法生成特征的表达式。"""
    lines = [f"gp_feature_{i + 1}: {program}" for i, program in enumerate(transformer)]
    output_file.write_text("\n".join(lines), encoding="utf-8")


# =====================================================================
# CLI 入口
# =====================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建10年期国债收益率未来5天方向预测 —— 二分类训练数据")
    parser.add_argument("--data-file", default="data/43_data.csv", help="预处理后的环境特征文件")
    parser.add_argument("--rate-file", default="data/43_rate.csv", help="预处理后的利率特征文件")
    parser.add_argument("--output-dir", default="data", help="输出目录")
    parser.add_argument("--seq-length", type=int, default=60, help="时序窗口长度")
    parser.add_argument("--step", type=int, default=1, help="滑动窗口步长")
    parser.add_argument("--skip-genetic", action="store_true",
                        help="跳过遗传算法（仅生成标签和底表，复用已有遗传特征）")
    parser.add_argument("--population-size", type=int, default=100000, help="遗传算法种群大小")
    parser.add_argument("--generations", type=int, default=5, help="遗传算法迭代代数")
    parser.add_argument("--n-components", type=int, default=10, help="生成新特征数量")
    parser.add_argument("--max-samples", type=float, default=0.9, help="每代拟合样本比例")
    parser.add_argument("--tournament-size", type=int, default=20, help="锦标赛大小")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / args.data_file
    rate_file = base_dir / args.rate_file
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: 合并数据 + 构建二分类标签 ----
    print("[Step 1] 构建训练底表与二分类标签...")
    merged, x_train, y_train = build_training_dataset(data_file, rate_file)
    merged.to_csv(output_dir / "43_train_merged_bin.csv", index=False)
    x_train.to_csv(output_dir / "43_x_raw_bin.csv")
    pd.DataFrame({"label": y_train}).to_csv(output_dir / "43_y_bin.csv", index=False)
    print(f"  底表: {len(merged)} 行, 特征: {x_train.shape[1]} 列\n")

    # ---- Step 2: 遗传算法特征工程（可跳过） ----
    if args.skip_genetic:
        print("[Step 2] 跳过遗传算法（--skip-genetic）")
        print("  如需使用遗传特征，请在模型训练脚本中加载 data/43_x.csv（三分类版本）")
        print("  或去掉 --skip-genetic 重新运行以生成 data/43_x_bin.csv\n")
        return

    print(f"[Step 2] 运行遗传算法特征工程 (population={args.population_size}, generations={args.generations})...")
    x_new_df, transformer = run_symbolic_transformer(
        x_train=x_train,
        y_train=y_train,
        population_size=args.population_size,
        generations=args.generations,
        n_components=args.n_components,
        max_samples=args.max_samples,
        tournament_size=args.tournament_size,
        random_state=args.random_state,
    )
    x_new_df.to_csv(output_dir / "43_x_bin.csv")
    save_formula_summary(transformer, output_dir / "43_x_formula_bin.txt")
    print(f"  遗传特征: {x_new_df.shape[1]} 列 -> data/43_x_bin.csv\n")

    # ---- Step 3: 构建时序样本 ----
    print("[Step 3] 构建时序样本...")
    x_seq, y_seq = build_sequence_samples(
        dataset_x=x_new_df,
        dataset_y=y_train,
        seq_length=args.seq_length,
        step=args.step,
    )
    np.save(output_dir / "43_x_seq_bin.npy", x_seq)
    np.save(output_dir / "43_y_seq_bin.npy", y_seq)
    print(f"  时序样本 X: {x_seq.shape}")
    print(f"  时序样本 y: {y_seq.shape}")
    print("\n二分类训练数据构建完成。")


if __name__ == "__main__":
    main()
