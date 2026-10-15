# -*- coding: utf-8 -*-
"""
10年期国债收益率未来5天方向预测 —— 二分类训练数据构建（2026新版）

基于 43_2_华泰证券债券预测_训练数据构建_二分类.py 改造。
核心逻辑不变，主要更新输入/输出路径以适配 model_new 目录结构。

标签定义：
  - label = 1：未来5日利率下行（rate_delta_5d < 0），债市偏多
  - label = 0：未来5日利率上行（rate_delta_5d >= 0），债市偏空

输出文件：
  model_new/data/train_merged_bin.csv  -> 合并后的训练底表
  model_new/data/x_raw_bin.csv         -> 原始训练特征
  model_new/data/y_bin.csv             -> 二分类标签
  model_new/data/x_genetic_bin.csv     -> 遗传算法生成的特征
  model_new/data/x_seq_bin.npy         -> 时序样本特征
  model_new/data/y_seq_bin.npy         -> 时序样本标签
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# =====================================================================
# 延迟导入 gplearn
# =====================================================================

def _require_gplearn():
    """延迟导入 gplearn，避免未安装时影响标签构建"""
    try:
        from gplearn.functions import make_function
        from gplearn.genetic import SymbolicTransformer
    except ImportError as exc:
        raise ImportError(
            "未安装 gplearn，请先执行 `pip install gplearn`"
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
    """
    if pd.isna(value):
        return np.nan
    return 1 if value < 0 else 0


# =====================================================================
# 数据合并与底表构建
# =====================================================================

def build_training_dataset(data_file: Path, rate_file: Path):
    """
    构建监督学习底表：
    1. 读取特征表和利率动量表
    2. 以日期为键合并
    3. 构造 future 5d 利率变化
    4. 生成二分类标签
    """
    df_data = pd.read_csv(data_file)
    df_rate = pd.read_csv(rate_file)

    # 日期列解析
    df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"], errors="coerce")
    df_rate["time"] = pd.to_datetime(df_rate["time"], errors="coerce")

    merged = pd.merge(
        df_data, df_rate,
        left_on="Unnamed: 0", right_on="time",
        how="inner"
    ).sort_values("Unnamed: 0")

    # 目标变量：未来 5 天利率变化
    merged["target_date_5d"] = merged["Unnamed: 0"].shift(-5)
    merged["rate_delta_5d"] = merged["rate"].shift(-5) - merged["rate"]
    merged["label"] = merged["rate_delta_5d"].apply(gen_label)

    # 剔除最后 5 天（无未来标签）
    merged = merged.dropna(subset=["label"]).copy()
    merged["label"] = merged["label"].astype(int)

    # 防止目标泄露：rate、label 等不进入特征
    drop_cols = ["time", "rate", "label", "rate_delta_5d", "target_date_5d", "date_list"]
    x_train = merged.drop(columns=[c for c in drop_cols if c in merged.columns]).copy()
    x_train = x_train.set_index("Unnamed: 0")
    x_train = x_train.apply(pd.to_numeric, errors="coerce").fillna(99.00)
    y_train = merged["label"].copy()

    # 打印标签分布
    print("=== 二分类标签分布 ===")
    dist = y_train.value_counts(normalize=True).sort_index()
    for lbl, ratio in dist.items():
        tag = "下行(偏多)" if lbl == 1 else "上行(偏空)"
        count = y_train.sum() if lbl == 1 else len(y_train) - y_train.sum()
        print(f"  label {lbl} ({tag}): {count} ({ratio:.1%})")

    return merged, x_train, y_train


# =====================================================================
# 遗传算法函数库（与原版完全一致）
# =====================================================================

def _pad_rolling_result(values, window, full_length):
    """将滚动计算结果前面补零，保持与输入等长"""
    if full_length < window:
        return np.zeros(full_length, dtype=np.float64)
    return np.concatenate([np.zeros(window - 1, dtype=np.float64), values])


def _make_ts_delay(make_function, window):
    def ts_delay(arr):
        return np.roll(np.asarray(arr, dtype=np.float64), window)
    return make_function(function=ts_delay, name=f"ts_delay_{window}", arity=1)


def _make_ts_corr(make_function, window):
    def ts_corr(arr1, arr2):
        try:
            arr1 = np.asarray(arr1, dtype=np.float64)
            arr2 = np.asarray(arr2, dtype=np.float64)
            n = len(arr1)
            if n < window:
                return np.zeros(n, dtype=np.float64)
            # pandas rolling corr 避免 (N,W) 中间数组
            s1, s2 = pd.Series(arr1), pd.Series(arr2)
            corr = s1.rolling(window, min_periods=window).corr(s2).to_numpy(copy=True)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            return _pad_rolling_result(corr[window - 1:], window, n)
        except Exception:
            return np.zeros(len(arr1), dtype=np.float64)
    return make_function(function=ts_corr, name=f"ts_corr_{window}", arity=2)


def _make_ts_cov(make_function, window):
    def ts_cov(arr1, arr2):
        try:
            arr1 = np.asarray(arr1, dtype=np.float64)
            arr2 = np.asarray(arr2, dtype=np.float64)
            n = len(arr1)
            if n < window:
                return np.zeros(n, dtype=np.float64)
            # pandas rolling cov(ddof=0) 与原版 numpy 一致，避免 (N,W) 中间数组
            s1, s2 = pd.Series(arr1), pd.Series(arr2)
            cov = s1.rolling(window, min_periods=window).cov(s2, ddof=0).to_numpy(copy=True)
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            return _pad_rolling_result(cov[window - 1:], window, n)
        except Exception:
            return np.zeros(len(arr1), dtype=np.float64)
    return make_function(function=ts_cov, name=f"ts_cov_{window}", arity=2)


def _make_single_series_stat(make_function, window, stat_name):
    """单序列滚动统计量（max/min/mean/std/zscore/decay_linear 等）

    内存优化：zscore / decay_linear / stddev 使用滚动累量法或卷积，
    避免 sliding_window_view 展开 (N, W) 大数组导致 OOM。
    """
    # 低内存统计量：用 pandas rolling 或 np.convolve，避免 (N,W) 中间数组
    _LOW_MEM_STATS = ("kurt", "skew", "zscore", "decay_linear", "stddev", "mean", "sum")

    def _calc(arr):
        try:
            arr = np.asarray(arr, dtype=np.float64)
            n = len(arr)
            if n < window:
                return np.zeros(n, dtype=np.float64)

            # ---- 低内存路径：pandas rolling / convolve（O(N) 额外内存）----
            if stat_name in _LOW_MEM_STATS:
                s = pd.Series(arr)
                if stat_name in ("kurt", "skew"):
                    s1 = s.rolling(window, min_periods=window).sum()
                    s2 = (s ** 2).rolling(window, min_periods=window).sum()
                    s3 = (s ** 3).rolling(window, min_periods=window).sum()
                    n_w = float(window)
                    mu = s1 / n_w
                    m2 = s2 / n_w - mu ** 2
                    m2_safe = np.where(m2 <= 0, np.nan, m2)
                    if stat_name == "skew":
                        m3 = s3 / n_w - 3 * mu * (s2 / n_w) + 2 * mu ** 3
                        v = (m3 / (m2_safe ** 1.5)).to_numpy(copy=True)
                    else:
                        s4 = (s ** 4).rolling(window, min_periods=window).sum()
                        m4 = s4 / n_w - 4 * mu * (s3 / n_w) + 6 * mu ** 2 * (s2 / n_w) - 3 * mu ** 4
                        v = (m4 / (m2_safe ** 2) - 3).to_numpy(copy=True)
                    v[:window - 1] = 0.0
                    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

                elif stat_name == "zscore":
                    roll_mean = s.rolling(window, min_periods=window).mean()
                    # ddof=0 与原版 numpy np.std 一致
                    roll_std = s.rolling(window, min_periods=window).std(ddof=0)
                    roll_std = roll_std.replace(0, 1.0)
                    v = ((s - roll_mean) / roll_std).to_numpy(copy=True)
                    v[:window - 1] = 0.0
                    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

                elif stat_name == "decay_linear":
                    weights = np.arange(window, 0, -1, dtype=np.float64)
                    weights /= np.sum(weights)
                    # np.correlate 不翻转 kernel，等价于原版 windows*weights
                    v = np.correlate(arr, weights, mode='valid')
                    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                    return _pad_rolling_result(v, window, n)

                elif stat_name == "stddev":
                    # ddof=0 与原版 numpy np.std 一致
                    v = s.rolling(window, min_periods=window).std(ddof=0).to_numpy(copy=True)
                    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                    return _pad_rolling_result(v[window - 1:], window, n)

                elif stat_name == "mean":
                    v = s.rolling(window, min_periods=window).mean().to_numpy(copy=True)
                    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                    return _pad_rolling_result(v[window - 1:], window, n)

                elif stat_name == "sum":
                    v = s.rolling(window, min_periods=window).sum().to_numpy(copy=True)
                    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                    return _pad_rolling_result(v[window - 1:], window, n)

            # ---- 高内存路径：仍需 sliding_window_view 的统计量 ----
            windows = sliding_window_view(arr, window)
            if stat_name == "max":       v = np.max(windows, axis=1)
            elif stat_name == "min":     v = np.min(windows, axis=1)
            elif stat_name == "maxarg":  v = np.argmax(windows, axis=1)
            elif stat_name == "minarg":  v = np.argmin(windows, axis=1)
            elif stat_name == "median":  v = np.median(windows, axis=1)
            elif stat_name == "product": v = np.prod(windows, axis=1)
            else:
                raise ValueError(f"未知统计函数: {stat_name}")
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            return _pad_rolling_result(v, window, n)
        except Exception:
            # 单 program 异常不影响整体遗传算法，返回零向量
            return np.zeros(len(arr), dtype=np.float64)
    name = f"decay_linear_{window}" if stat_name == "decay_linear" else f"ts_{stat_name}_{window}"
    return make_function(function=_calc, name=name, arity=1)


def _make_sort_cut(make_function, window, n, desc):
    """滚动窗口内取最大/最小的 n 个值的均值"""
    def _calc(arr):
        arr = np.asarray(arr, dtype=np.float64)
        if len(arr) < window:
            return np.zeros_like(arr)
        windows = sliding_window_view(arr, window)
        if desc:
            v = np.mean(np.partition(windows, -n, axis=-1)[..., -n:], axis=-1)
        else:
            v = np.mean(np.partition(windows, n - 1, axis=-1)[..., :n], axis=-1)
        return _pad_rolling_result(v, window, len(arr))
    prefix = "ts_decsortcut" if desc else "ts_ascsortcut"
    return make_function(function=_calc, name=f"{prefix}_{window}_{n}", arity=1)


def _make_const(make_function, offset):
    def _calc(arr):
        return np.asarray(arr, dtype=np.float64) + offset
    return make_function(function=_calc, name=f"const_{offset}", arity=1)


def build_function_set():
    """构建遗传算法函数库"""
    make_function, _ = _require_gplearn()
    functions = []
    for window in [5, 20, 60, 120]:
        functions.append(_make_ts_delay(make_function, window))
        functions.append(_make_ts_corr(make_function, window))
        functions.append(_make_ts_cov(make_function, window))
        for stat in ["max", "min", "maxarg", "minarg", "mean", "median",
                      "sum", "stddev", "product", "kurt", "skew", "zscore", "decay_linear"]:
            functions.append(_make_single_series_stat(make_function, window, stat))
    for window, n in zip([20, 60, 120], [5, 15, 30]):
        functions.append(_make_sort_cut(make_function, window, n, desc=False))
        functions.append(_make_sort_cut(make_function, window, n, desc=True))
    for offset in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
        functions.append(_make_const(make_function, offset))
    # 逻辑运算符
    def _l1(x1, x2, x3, x4): return np.where(x1 > x2, x3, x4)
    def _l2(x1, x2):         return np.where(x1 > x2, x1, x2)
    def _l3(x1, x2):         return np.where(x1 > x2, 1, -1)
    def _l4(x1, x2):         return np.where(x1 > x2, 1, 0)
    functions.extend([
        "add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv",
        make_function(function=_l1, name="logical1", arity=4),
        make_function(function=_l2, name="logical2", arity=2),
        make_function(function=_l3, name="logical3", arity=2),
        make_function(function=_l4, name="logical4", arity=2),
    ])
    return functions


def run_symbolic_transformer(x_train, y_train, args):
    """运行遗传算法特征工程"""
    _, SymbolicTransformer = _require_gplearn()
    transformer = SymbolicTransformer(
        population_size=args.population_size,
        generations=args.generations,
        function_set=build_function_set(),
        n_components=args.n_components,
        metric="pearson",
        parsimony_coefficient=0.01,
        max_samples=args.max_samples,
        tournament_size=args.tournament_size,
        verbose=1,
        random_state=args.random_state,
    )
    x_new = transformer.fit_transform(x_train, y_train)
    names = [f"gp_feature_{i + 1}" for i in range(x_new.shape[1])]
    return pd.DataFrame(x_new, index=x_train.index, columns=names), transformer


# =====================================================================
# 时序样本构建
# =====================================================================

def build_sequence_samples(dataset_x, dataset_y, seq_length, step=1):
    """构建时序样本：过去 seq_length 天特征作为一个样本"""
    x_vals = dataset_x.to_numpy(dtype=np.float32)
    y_vals = dataset_y.to_numpy(dtype=np.int64)
    groups_x, groups_y = [], []
    for start in range(0, len(dataset_x) - seq_length + 1, step):
        end = start + seq_length
        groups_x.append(x_vals[start:end, :])
        groups_y.append(y_vals[end - 1])
    if not groups_x:
        raise ValueError("样本不足，请检查 seq_length")
    return np.asarray(groups_x, dtype=np.float32), np.asarray(groups_y, dtype=np.int64)


# =====================================================================
# CLI 入口
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="二分类训练数据构建（2026新版）")
    parser.add_argument("--data-file", default="model_new/data/data.csv")
    parser.add_argument("--rate-file", default="model_new/data/rate.csv")
    parser.add_argument("--output-dir", default="model_new/data")
    parser.add_argument("--seq-length", type=int, default=60)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--skip-genetic", action="store_true",
                        help="跳过遗传算法（仅生成标签和底表）")
    parser.add_argument("--reuse-cache", action="store_true",
                        help="复用已有 x_raw_bin.csv/y_bin.csv，跳过 Step1 直接重试遗传算法")
    parser.add_argument("--population-size", type=int, default=100000)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--max-samples", type=float, default=0.9)
    parser.add_argument("--tournament-size", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent.parent  # 项目根目录
    data_file = base_dir / args.data_file
    rate_file = base_dir / args.rate_file
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 合并数据 + 标签（支持缓存复用，避免遗传算法报错后从头重跑）
    x_raw_cache = output_dir / "x_raw_bin.csv"
    y_cache = output_dir / "y_bin.csv"
    if args.reuse_cache and x_raw_cache.exists() and y_cache.exists():
        print("[Step 1] 复用缓存（--reuse-cache），跳过底表构建...")
        x_train = pd.read_csv(x_raw_cache, index_col=0)
        y_train = pd.read_csv(y_cache)["label"]
        print(f"  特征: {x_train.shape[1]} 列, 样本: {len(x_train)} 行\n")
    else:
        print("[Step 1] 构建训练底表与二分类标签...")
        merged, x_train, y_train = build_training_dataset(data_file, rate_file)
        merged.to_csv(output_dir / "train_merged_bin.csv", index=False)
        x_train.to_csv(x_raw_cache)
        pd.DataFrame({"label": y_train}).to_csv(y_cache, index=False)
        print(f"  底表: {len(merged)} 行, 特征: {x_train.shape[1]} 列\n")

    # Step 2: 遗传算法
    if args.skip_genetic:
        print("[Step 2] 跳过遗传算法（--skip-genetic）\n")
        return

    print(f"[Step 2] 遗传算法 (pop={args.population_size}, gen={args.generations})...")
    x_new_df, transformer = run_symbolic_transformer(x_train, y_train, args)
    x_new_df.to_csv(output_dir / "x_genetic_bin.csv")

    # 保存公式
    formulas = [f"gp_feature_{i+1}: {p}" for i, p in enumerate(transformer)]
    (output_dir / "x_formula_bin.txt").write_text("\n".join(formulas), encoding="utf-8")
    print(f"  遗传特征: {x_new_df.shape[1]} 列\n")

    # Step 3: 时序样本
    print("[Step 3] 构建时序样本...")
    x_seq, y_seq = build_sequence_samples(x_new_df, y_train, args.seq_length, args.step)
    np.save(output_dir / "x_seq_bin.npy", x_seq)
    np.save(output_dir / "y_seq_bin.npy", y_seq)
    print(f"  X: {x_seq.shape}, y: {y_seq.shape}")
    print("\n完成。")


if __name__ == "__main__":
    main()
