"""
10年期国债收益率未来5天方向预测 —— 训练数据构建

本脚本完成三部分工作：
1. 读取预处理后的 `43_data.csv` 与 `43_rate.csv`，构建监督学习训练特征与标签
2. 使用 SymbolicTransformer 做遗传算法特征工程，生成 10 个新特征
3. 将特征表切分为时序样本，生成过去 60 天 -> 当前标签 的序列数据

输出文件：
  data/43_train_merged.csv   -> 合并后的训练底表（含 rate_delta_5d 和 label）
  data/43_x_raw.csv          -> 原始训练特征
  data/43_y.csv              -> 标签
  data/43_x.csv              -> 遗传算法生成的 10 个特征
  data/43_x_seq.npy          -> 时序样本特征，形状 [N, seq_length, feature_dim]
  data/43_y_seq.npy          -> 时序样本标签，形状 [N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _require_gplearn():
    """
    延迟导入 gplearn。

    这样做的好处是：
    1. 仅构建原始训练特征或训练 XGBoost 时，不会因为缺少 gplearn 而失败
    2. 只有在执行遗传算法特征工程时，才要求安装 gplearn
    """
    try:
        from gplearn.functions import make_function
        from gplearn.genetic import SymbolicTransformer
    except ImportError as exc:  # pragma: no cover - 运行期依赖提示
        raise ImportError(
            "未安装 gplearn，请先执行 `pip install gplearn` 后再运行遗传特征工程。"
        ) from exc
    return make_function, SymbolicTransformer


def gen_label(value: float) -> float:
    """将未来 5 天利率变化离散为三分类标签。"""
    if pd.isna(value):
        return np.nan
    if value < -0.03:
        return 0
    if value < 0.03:
        return 1
    return 2


def build_training_dataset(data_file: Path, rate_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    构建监督学习底表。

    处理逻辑：
    1. 读取宏观/流动性特征表和利率动量特征表
    2. 以日期为键进行合并
    3. 构造 future 5d 利率变化 `rate_delta_5d`
    4. 生成三分类标签 `label`
    5. 输出原始特征 `x_train` 与标签 `y_train`
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

    # 目标变量：未来 5 天利率变化
    # 同时保留未来第 5 个交易日日期，供预测结果展示表使用
    merged["target_date_5d"] = merged["Unnamed: 0"].shift(-5)
    merged["rate_delta_5d"] = merged["rate"].shift(-5) - merged["rate"]
    merged["label"] = merged["rate_delta_5d"].apply(gen_label)

    # 最后 5 天没有未来标签，需要剔除，避免被误分类
    merged = merged.dropna(subset=["label"]).copy()
    merged["label"] = merged["label"].astype(int)

    # 注意：rate、label、rate_delta_5d 都不能进入训练特征，防止目标泄露
    x_train = merged.drop(
        columns=["time", "rate", "label", "rate_delta_5d", "target_date_5d", "date_list"],
        errors="ignore",
    ).copy()
    x_train = x_train.set_index("Unnamed: 0")
    x_train = x_train.apply(pd.to_numeric, errors="coerce").fillna(99.00)
    y_train = merged["label"].copy()
    print("stage build_training_dataset ")
    return merged, x_train, y_train


def _pad_rolling_result(values: np.ndarray, window: int, full_length: int) -> np.ndarray:
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
        mean1 = np.mean(win1, axis=1)
        mean2 = np.mean(win2, axis=1)
        std1 = np.std(win1, axis=1)
        std2 = np.std(win2, axis=1)
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
        mean1 = np.mean(win1, axis=1)
        mean2 = np.mean(win2, axis=1)
        cov = np.mean(win1 * win2, axis=1) - mean1 * mean2
        cov[np.isnan(cov) | np.isinf(cov)] = 0
        return _pad_rolling_result(cov, window, len(arr1))
    print("_make_ts_cov")
    return make_function(function=ts_cov, name=f"ts_cov_{window}", arity=2)


def _make_single_series_stat(window: int, stat_name: str):
    make_function, _ = _require_gplearn()

    def _calc(arr):
        arr = np.asarray(arr, dtype=np.float64)
        if len(arr) < window:
            return np.zeros_like(arr)
        windows = sliding_window_view(arr, window_shape=window)

        if stat_name == "max":
            values = np.max(windows, axis=1)
        elif stat_name == "min":
            values = np.min(windows, axis=1)
        elif stat_name == "maxarg":
            values = np.argmax(windows, axis=1)
        elif stat_name == "minarg":
            values = np.argmin(windows, axis=1)
        elif stat_name == "mean":
            values = np.mean(windows, axis=1)
        elif stat_name == "median":
            values = np.median(windows, axis=1)
        elif stat_name == "sum":
            values = np.sum(windows, axis=1)
        elif stat_name == "stddev":
            values = np.std(windows, axis=1)
        elif stat_name == "product":
            values = np.prod(windows, axis=1)
        elif stat_name == "kurt":
            stds = np.std(windows, axis=1)
            means = np.mean(windows, axis=1, keepdims=True)
            deviations = windows - means
            mu4 = np.mean(deviations ** 4, axis=1)
            stds[stds == 0] = 1
            values = mu4 / (stds ** 4) - 3
        elif stat_name == "skew":
            stds = np.std(windows, axis=1)
            means = np.mean(windows, axis=1, keepdims=True)
            deviations = windows - means
            mu3 = np.mean(deviations ** 3, axis=1)
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
    print("stage_make_single_series_stat")
    return make_function(function=_calc, name=f"ts_{stat_name}_{window}" if stat_name != "decay_linear" else f"decay_linear_{window}", arity=1)


def _make_sort_cut(window: int, n: int, desc: bool):
    make_function, _ = _require_gplearn()

    def _calc(arr):
        arr = np.asarray(arr, dtype=np.float64)
        if len(arr) < window:
            return np.zeros_like(arr)
        windows = sliding_window_view(arr, window_shape=window, axis=0)
        if desc:
            partitioned = np.partition(windows, -n, axis=-1)
            values = np.mean(partitioned[..., -n:], axis=-1)
        else:
            partitioned = np.partition(windows, n - 1, axis=-1)
            values = np.mean(partitioned[..., :n], axis=-1)
        return _pad_rolling_result(values, window, len(arr))
    print("_make_sort_cut_stage")
    prefix = "ts_decsortcut" if desc else "ts_ascsortcut"
    return make_function(function=_calc, name=f"{prefix}_{window}_{n}", arity=1)


def _make_const(offset: int):
    make_function, _ = _require_gplearn()

    def _calc(arr):
        arr = np.asarray(arr, dtype=np.float64)
        return arr + offset

    return make_function(function=_calc, name=f"const_{offset}", arity=1)


def build_function_set():
    """构建参考资料中的遗传算法函数库。"""
    make_function, _ = _require_gplearn()
    functions = []
    windows = [5, 20, 60, 120]

    for window in windows:
        functions.append(_make_ts_delay(window))
        functions.append(_make_ts_corr(window))
        functions.append(_make_ts_cov(window))
        for stat_name in [
            "max",
            "min",
            "maxarg",
            "minarg",
            "mean",
            "median",
            "sum",
            "stddev",
            "product",
            "kurt",
            "skew",
            "zscore",
            "decay_linear",
        ]:
            functions.append(_make_single_series_stat(window, stat_name))

    for window, n in zip([20, 60, 120], [5, 15, 30]):
        functions.append(_make_sort_cut(window, n, desc=False))
        functions.append(_make_sort_cut(window, n, desc=True))

    for offset in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
        functions.append(_make_const(offset))

    def _logical1(x1, x2, x3, x4):
        return np.where(x1 > x2, x3, x4)

    def _logical2(x1, x2):
        return np.where(x1 > x2, x1, x2)

    def _logical3(x1, x2):
        return np.where(x1 > x2, 1, -1)

    def _logical4(x1, x2):
        return np.where(x1 > x2, 1, 0)

    functions.extend(
        [
            "add",
            "sub",
            "mul",
            "div",
            "sqrt",
            "log",
            "abs",
            "neg",
            "inv",
            make_function(function=_logical1, name="logical1", arity=4),
            make_function(function=_logical2, name="logical2", arity=2),
            make_function(function=_logical3, name="logical3", arity=2),
            make_function(function=_logical4, name="logical4", arity=2),
        ]
    )
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
) -> tuple[pd.DataFrame, SymbolicTransformer]:
    """运行遗传算法特征工程。"""
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


def build_sequence_samples(
    dataset_x: pd.DataFrame,
    dataset_y: pd.Series,
    seq_length: int,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构建时序样本。

    逻辑与参考资料一致：
    - 用过去 seq_length 天特征作为一个样本
    - 样本标签取窗口最后一个时点对应的 label
    """
    x_values = dataset_x.to_numpy(dtype=np.float32)
    y_values = dataset_y.to_numpy(dtype=np.int64)

    groups_x = []
    groups_y = []
    for start in range(0, len(dataset_x) - seq_length + 1, step):
        end = start + seq_length
        groups_x.append(x_values[start:end, :])
        groups_y.append(y_values[end - 1])

    if not groups_x:
        raise ValueError("样本数量不足，无法构造时序窗口，请检查 seq_length 设置。")

    return np.asarray(groups_x, dtype=np.float32), np.asarray(groups_y, dtype=np.int64)


def save_formula_summary(transformer: SymbolicTransformer, output_file: Path) -> None:
    """保存遗传算法生成特征的表达式，便于后续查看。"""
    lines = []
    for i, program in enumerate(transformer):
        lines.append(f"gp_feature_{i + 1}: {program}")
    output_file.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建10年期国债收益率未来5天方向预测训练数据")
    parser.add_argument("--data-file", default="data/43_data.csv", help="预处理后的环境特征文件")
    parser.add_argument("--rate-file", default="data/43_rate.csv", help="预处理后的利率特征文件")
    parser.add_argument("--output-dir", default="data", help="输出目录")
    parser.add_argument("--seq-length", type=int, default=60, help="时序窗口长度")
    parser.add_argument("--step", type=int, default=1, help="滑动窗口步长")
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

    merged, x_train, y_train = build_training_dataset(data_file, rate_file)
    merged.to_csv(output_dir / "43_train_merged.csv", index=False)
    x_train.to_csv(output_dir / "43_x_raw.csv")
    pd.DataFrame({"label": y_train}).to_csv(output_dir / "43_y.csv", index=False)

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
    x_new_df.to_csv(output_dir / "43_x.csv")
    save_formula_summary(transformer, output_dir / "43_x_formula.txt")

    x_seq, y_seq = build_sequence_samples(
        dataset_x=x_new_df,
        dataset_y=y_train,
        seq_length=args.seq_length,
        step=args.step,
    )
    np.save(output_dir / "43_x_seq.npy", x_seq)
    np.save(output_dir / "43_y_seq.npy", y_seq)

    print("训练数据构建完成")
    print(f"原始训练特征: {output_dir / '43_x_raw.csv'}")
    print(f"遗传算法特征: {output_dir / '43_x.csv'}")
    print(f"标签文件: {output_dir / '43_y.csv'}")
    print(f"时序样本 X: {output_dir / '43_x_seq.npy'} -> {x_seq.shape}")
    print(f"时序样本 y: {output_dir / '43_y_seq.npy'} -> {y_seq.shape}")


if __name__ == "__main__":
    main()
