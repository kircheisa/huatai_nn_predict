# -*- coding: utf-8 -*-
"""
特征筛选脚本 —— 基于训练后的稳定性统计，生成三套特征保留列表

依赖文件（由 03_XGB训练_二分类.py 首次运行后生成）：
  xgb_feature_stability_bin.csv  -> 每个 feature 的 appear_ratio / avg_importance
  xgb_never_nonzero_bin.csv      -> 从未 importance>0 的特征
  xgb_train_features_bin.csv     -> 全量特征列名（用于交叉验证）

输出文件（model_new/data/ 目录）：
  keep_cols_full.txt         -> 仅删噪音（约180列）
  keep_cols_conservative.txt -> 保守筛选 appear_ratio≥0.10（约90列）
  keep_cols_aggressive.txt   -> 激进筛选 appear_ratio≥0.15（约60列）

使用方式（配合 03 脚本）：
  python model_new/03_XGB训练_二分类.py --keep-cols-file model_new/data/keep_cols_conservative.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "model_new" / "data"

# 相关性剪枝阈值：Spearman |corr| 超过此值的特征对，删除稳定性较低的那个
CORR_THRESHOLD = 0.85


# =====================================================================
# 噪音特征黑名单（硬规则，无论稳定性如何都删除）
# =====================================================================

# 分类编码列（无序变量当连续值用，纯噪音）
CATEGORICAL_NOISE = {
    "中国各领导人时期-1", "中国各领导人时期-2",
    "美国各总统时期-1", "美国各总统时期-2", "美国各总统时期-3",
    "美国各总统时期-4", "美国各总统时期-5",
    "基本面主题变量-1", "基本面主题变量-2",
    "基本面主题变量-3", "基本面主题变量-4",
}

# 目标泄露 / 无意义列
LEAKAGE_OR_MEANINGLESS = {
    "target_date_5d",  # 预测目标日期，泄露
}

# 高共线性的托管同比（保留"广义基金托管同比"作为代表，其余删除）
COLLINEAR_CUSTODY = {
    "保险托管同比", "商业银行托管同比", "证券公司托管同比",
}


def load_stability_data(suffix="_bin"):
    """加载稳定性统计和全量特征列"""
    stab_path = DATA_DIR / f"xgb_feature_stability{suffix}.csv"
    never_path = DATA_DIR / f"xgb_never_nonzero{suffix}.csv"
    feat_path = DATA_DIR / f"xgb_train_features{suffix}.csv"

    if not stab_path.exists():
        raise FileNotFoundError(
            f"未找到 {stab_path.name}，请先运行 03_XGB训练_二分类.py 生成稳定性统计"
        )

    stab = pd.read_csv(stab_path)
    never = pd.read_csv(never_path)["never_nonzero_feature"].tolist() if never_path.exists() else []

    # 从全量特征文件获取所有列名
    if feat_path.exists():
        all_features = pd.read_csv(feat_path, index_col=0, nrows=0).columns.tolist()
    else:
        # 兜底：从稳定性统计 + never 推断
        all_features = stab["feature"].tolist() + never

    return stab, set(never), all_features


def build_blacklist(never_set: set[str]) -> set[str]:
    """构建硬删除黑名单：从未非零 + 分类编码 + 泄露 + 共线性"""
    blacklist = set()
    blacklist |= never_set
    blacklist |= CATEGORICAL_NOISE
    blacklist |= LEAKAGE_OR_MEANINGLESS
    blacklist |= COLLINEAR_CUSTODY
    return blacklist


def filter_by_appearance(stab: pd.DataFrame, threshold: float, all_features: list[str]) -> list[str]:
    """按 appear_ratio 阈值筛选"""
    keep_stab = stab.loc[stab["appear_ratio"] >= threshold, "feature"].tolist()
    return keep_stab


def correlation_pruning(keep_list: list[str], stab: pd.DataFrame, threshold: float = CORR_THRESHOLD, suffix: str = "_bin") -> tuple[list[str], int]:
    """
    基于 Spearman 相关性的全局剪枝（中信建投"全局相关性剪枝"同款手法）。

    规则：对任意两个特征，若 |spearman_corr| > threshold，删除 avg_importance 较低的那个
         （即保留稳定性得分更高的代表特征）。

    参数：
      keep_list  : 待剪枝的特征列表（已过稳定性/黑名单筛选）
      stab       : 稳定性统计表，含 feature / avg_importance 列
      threshold  : 相关系数阈值，默认 0.85

    返回：(剪枝后的特征列表, 删除数量)
    """
    feat_path = DATA_DIR / f"xgb_train_features{suffix}.csv"
    if not feat_path.exists():
        print(f"  [跳过相关性剪枝] 未找到 xgb_train_features{suffix}.csv", flush=True)
        return keep_list, 0

    # 读取实际特征值用于计算相关性
    feat_df = pd.read_csv(feat_path, index_col=0)
    available = [c for c in keep_list if c in feat_df.columns]
    if len(available) < 2:
        return keep_list, 0

    # Spearman 相关性矩阵（对 NaN 安全）
    sub = feat_df[available].select_dtypes(include=[np.number])
    corr = sub.corr(method="spearman").abs()

    # 构建稳定性映射：avg_importance 越高代表特征越稳定
    imp_map = dict(zip(stab["feature"], stab["avg_importance"]))
    # 未在 stability 表中的特征（如新增 STL）赋予低优先级 0
    imp_map_default = lambda f: imp_map.get(f, 0.0)

    # 贪心删除：扫描上三角，高相关对中删除稳定性较低者
    to_drop = set()
    cols = list(corr.columns)
    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue
            if corr.iloc[i, j] > threshold:
                # 保留稳定性更高的，删除另一个
                if imp_map_default(cols[i]) >= imp_map_default(cols[j]):
                    to_drop.add(cols[j])
                else:
                    to_drop.add(cols[i])
                    break  # cols[i] 被删，跳出内层循环

    pruned = [f for f in keep_list if f not in to_drop]
    return pruned, len(to_drop)


def main():
    import sys
    # 支持 --suffix 参数（双周模型用 _10d）
    SUF = "_bin"
    if "--suffix" in sys.argv:
        idx = sys.argv.index("--suffix")
        SUF = sys.argv[idx + 1]
    out_tag = SUF if SUF != "_bin" else ""

    print("=" * 60)
    print(f"特征筛选 —— 基于稳定性的三套方案 (suffix={SUF})")
    print("=" * 60)

    stab, never_set, all_features = load_stability_data(SUF)
    blacklist = build_blacklist(never_set)

    print(f"\n全量特征: {len(all_features)} 列")
    print(f"  - 从未非零 (appear_ratio=0): {len(never_set)}")
    print(f"  - 分类编码噪音: {len(CATEGORICAL_NOISE)}")
    print(f"  - 目标泄露/无意义: {len(LEAKAGE_OR_MEANINGLESS)}")
    print(f"  - 共线性托管(留1个): {len(COLLINEAR_CUSTODY)}")
    print(f"  硬删除黑名单合计: {len(blacklist)} 个特征（去重后）\n")

    # === 方案1：仅删噪音 + 相关性剪枝（FULL）===
    keep_full = [f for f in all_features if f not in blacklist]
    keep_full, n_prune_full = correlation_pruning(keep_full, stab, suffix=SUF)

    # === 方案2：保守（CONSERVATIVE, appear_ratio>=0.10）+ 相关性剪枝 ===
    keep_conservative_stab = filter_by_appearance(stab, 0.10, all_features)
    keep_conservative = [f for f in keep_conservative_stab if f not in blacklist]
    keep_conservative, n_prune_cons = correlation_pruning(keep_conservative, stab, suffix=SUF)

    # === 方案3：激进（AGGRESSIVE, appear_ratio>=0.15）+ 相关性剪枝 ===
    keep_aggressive_stab = filter_by_appearance(stab, 0.15, all_features)
    keep_aggressive = [f for f in keep_aggressive_stab if f not in blacklist]
    keep_aggressive, n_prune_aggr = correlation_pruning(keep_aggressive, stab, suffix=SUF)

    # 打印各方案详情
    schemes = [
        ("FULL (删噪音+去相关)", keep_full, f"keep_cols_full{out_tag}.txt", n_prune_full),
        ("CONSERVATIVE (ratio>=0.10+去相关)", keep_conservative, f"keep_cols_conservative{out_tag}.txt", n_prune_cons),
        ("AGGRESSIVE (ratio>=0.15+去相关)", keep_aggressive, f"keep_cols_aggressive{out_tag}.txt", n_prune_aggr),
    ]

    print(f"{'方案':<36} {'保留列数':>8} {'相关性删除':>10} {'总删除':>6}")
    print("-" * 66)
    for name, keep_list, outfile, n_prune in schemes:
        n_drop = len(all_features) - len(keep_list)
        print(f"{name:<36} {len(keep_list):>8} {n_prune:>10} {n_drop:>6}")
        out_path = DATA_DIR / outfile
        out_path.write_text("\n".join(keep_list), encoding="utf-8")
        print(f"  -> 已保存: {out_path.relative_to(BASE)}")
    print()

    # 详细列出各方案删除的特征
    print("=" * 60)
    print("方案对比（被删除的特征）")
    print("=" * 60)

    keep_full_set = set(keep_full)
    keep_cons_set = set(keep_conservative)
    keep_aggr_set = set(keep_aggressive)

    # CONSERVATIVE 比 FULL 多删了什么
    extra_cons = keep_full_set - keep_cons_set
    extra_aggr = keep_full_set - keep_aggr_set

    print(f"\n[FULL -> CONSERVATIVE] 额外删除 {len(extra_cons)} 列（appear_ratio<0.10 的低频特征）:")
    for f in sorted(extra_cons):
        row = stab.loc[stab["feature"] == f]
        ratio = row.iloc[0]["appear_ratio"] if len(row) > 0 else 0
        print(f"  {f}  (ratio={ratio:.3f})")

    print(f"\n[CONSERVATIVE -> AGGRESSIVE] 额外删除 {len(keep_cons_set - keep_aggr_set)} 列（0.10≤ratio<0.15）:")
    for f in sorted(keep_cons_set - keep_aggr_set):
        row = stab.loc[stab["feature"] == f]
        ratio = row.iloc[0]["appear_ratio"] if len(row) > 0 else 0
        print(f"  {f}  (ratio={ratio:.3f})")

    # 推荐的对比实验命令
    print("\n" + "=" * 60)
    print("对比实验命令")
    print("=" * 60)
    for label, _, outfile, _ in schemes:
        print(f"\n# {label}")
        cmd = f"python model_new/03_XGB训练_二分类.py --keep-cols-file model_new/data/{outfile}"
        print(f"  {cmd}")

    # 推荐组合：大窗口 + 保守筛选
    print("\n" + "=" * 60)
    print("推荐组合（大窗口 + 保守筛选）")
    print("=" * 60)
    print("\npython model_new/03_XGB训练_二分类.py "
          "--rolling-window 504 --learning-rate 0.05 "
          "--keep-cols-file model_new/data/keep_cols_conservative.txt")
    print("  W=504(2年) 训练样本441天，配合90列特征，样本/特征比≈4.9")
    print()


if __name__ == "__main__":
    main()
