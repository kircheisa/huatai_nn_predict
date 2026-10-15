# -*- coding: utf-8 -*-
"""
美元指数与上证A股指数相关性分析
重点：美元指数突破 100 点对上证A股指数的影响
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

# ---------- 中文字体配置（兼容 Windows 常见中文字体） ----------
for f in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
    try:
        rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
rcParams["axes.unicode_minus"] = False

# ---------- 路径配置 ----------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE, "美元指数相关性分析.txt")
OUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    """读取制表符分隔的 txt 数据，清洗千分位逗号与多余空格"""
    df = pd.read_csv(DATA_FILE, sep="\t", dtype=str)
    # 统一列名，便于后续引用
    df.columns = [c.strip() for c in df.columns]
    # 数值列清洗：去除逗号、空格
    num_cols = ["美元指数", "上证A股指数", "上证所:股票成交金额", "上证所:主板A股:成交金额"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c].str.replace(",", "", regex=False).str.strip(), errors="coerce")
    # 日期解析并按时间升序
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["日期"] + num_cols).sort_values("日期").reset_index(drop=True)
    return df


def overall_corr(df):
    """整体相关性矩阵（Pearson 与 Spearman）"""
    cols = ["美元指数", "上证A股指数", "上证所:股票成交金额", "上证所:主板A股:成交金额"]
    pear = df[cols].corr(method="pearson")
    spe = df[cols].corr(method="spearman")
    return pear, spe


def plot_heatmap(corr, title, fname):
    """绘制相关性热力图"""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(corr.columns, fontsize=10)
    # 标注数值
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color="black" if abs(corr.values[i, j]) < 0.6 else "white", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)


def group_compare(df):
    """按美元指数是否 >=100 分组对比"""
    df = df.copy()
    df["美元>=100"] = np.where(df["美元指数"] >= 100, "≥100", "<100")
    # 分组描述统计
    desc = df.groupby("美元>=100")[["上证A股指数", "上证所:股票成交金额", "上证所:主板A股:成交金额"]].describe()
    # 分组均值汇总
    mean_tbl = df.groupby("美元>=100")[
        ["美元指数", "上证A股指数", "上证所:股票成交金额", "上证所:主板A股:成交金额"]
    ].mean().round(2)
    # 分组相关性（美元指数 vs 上证A股指数）
    grp_corr = {}
    for name, g in df.groupby("美元>=100"):
        if len(g) >= 3:
            r_p, p_p = stats.pearsonr(g["美元指数"], g["上证A股指数"])
            grp_corr[name] = {"pearson_r": r_p, "p_value": p_p, "n": len(g)}
    return df, desc, mean_tbl, grp_corr


def plot_timeseries(df):
    """双轴时间序列：美元指数 vs 上证A股指数"""
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax1.plot(df["日期"], df["美元指数"], color="#d62728", marker="o", ms=3, lw=1.2, label="美元指数")
    ax1.axhline(100, color="#d62728", ls="--", lw=1, alpha=0.7)
    ax1.set_ylabel("美元指数", color="#d62728", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#d62728")

    ax2 = ax1.twinx()
    ax2.plot(df["日期"], df["上证A股指数"], color="#1f77b4", marker="s", ms=3, lw=1.2, label="上证A股指数")
    ax2.set_ylabel("上证A股指数", color="#1f77b4", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    # 标注美元指数>=100的区域
    over = df[df["美元指数"] >= 100]
    if not over.empty:
        ax1.axvspan(over["日期"].min(), over["日期"].max(), color="#ffcccc", alpha=0.3, label="美元≥100区间")

    ax1.set_title("美元指数与上证A股指数走势（日期升序）", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "timeseries.png"), dpi=150)
    plt.close(fig)


def plot_scatter(df):
    """散点图 + 回归线：美元指数 vs 上证A股指数（按100分组着色）"""
    df = df.copy()
    df["组"] = np.where(df["美元指数"] >= 100, "≥100", "<100")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = {"≥100": "#d62728", "<100": "#1f77b4"}
    for name, g in df.groupby("组"):
        ax.scatter(g["美元指数"], g["上证A股指数"], s=45, alpha=0.75,
                   color=colors[name], edgecolor="white", label=f"美元{name}（n={len(g)}）")

    # 全样本拟合回归线
    x = df["美元指数"].values
    y = df["上证A股指数"].values
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="green", lw=1.5, ls="--",
            label=f"回归线: y={slope:.2f}x+{intercept:.2f}")

    ax.axvline(100, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("美元指数", fontsize=11)
    ax.set_ylabel("上证A股指数", fontsize=11)
    ax.set_title("美元指数 vs 上证A股指数 散点回归", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scatter.png"), dpi=150)
    plt.close(fig)


def main():
    df = load_data()
    print("=" * 70)
    print(f"样本量: {len(df)} 行   日期范围: {df['日期'].min().date()} ~ {df['日期'].max().date()}")
    print("=" * 70)

    # 1. 整体相关性
    pear, spe = overall_corr(df)
    print("\n【整体 Pearson 相关性矩阵】")
    print(pear.round(3).to_string())
    print("\n【整体 Spearman 相关性矩阵】")
    print(spe.round(3).to_string())
    plot_heatmap(pear, "Pearson 相关性矩阵", "heatmap_pearson.png")
    plot_heatmap(spe, "Spearman 相关性矩阵", "heatmap_spearman.png")

    # 2. 分组对比
    df2, desc, mean_tbl, grp_corr = group_compare(df)
    print("\n【按美元指数是否≥100 分组均值】")
    print(mean_tbl.to_string())
    print("\n【分组内 美元指数 vs 上证A股指数 相关性】")
    for k, v in grp_corr.items():
        print(f"  美元{k}: Pearson r={v['pearson_r']:.3f}, p={v['p_value']:.4f}, n={v['n']}")
    print("\n【上证A股指数描述统计（按组）】")
    print(desc["上证A股指数"].round(2).to_string())

    # 3. 可视化
    plot_timeseries(df2)
    plot_scatter(df2)

    # 4. 导出结果
    pear.to_csv(os.path.join(OUT_DIR, "corr_pearson.csv"), encoding="utf-8-sig")
    spe.to_csv(os.path.join(OUT_DIR, "corr_spearman.csv"), encoding="utf-8-sig")
    mean_tbl.to_csv(os.path.join(OUT_DIR, "group_mean.csv"), encoding="utf-8-sig")

    # 5. 关键结论
    r_pear = pear.loc["美元指数", "上证A股指数"]
    over100 = df[df["美元指数"] >= 100]
    below100 = df[df["美元指数"] < 100]
    print("\n" + "=" * 70)
    print("【关键结论】")
    print(f"1) 全样本 美元指数与上证A股指数 Pearson 相关系数 = {r_pear:.3f}")
    print(f"2) 美元≥100 样本数 {len(over100)}，均值 上证A股={over100['上证A股指数'].mean():.2f}")
    print(f"3) 美元<100  样本数 {len(below100)}，均值 上证A股={below100['上证A股指数'].mean():.2f}")
    diff = over100['上证A股指数'].mean() - below100['上证A股指数'].mean()
    print(f"4) 美元≥100 较 <100 的上证A股指数均值差 = {diff:+.2f} 点")
    print(f"5) 输出目录: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
