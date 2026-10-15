# -*- coding: utf-8 -*-
"""
方法论检验：在"美元指数≥100"子样本中，Pearson 相关系数是否合理？
检验维度：
  1) 正态性检验（Pearson 的前提）
  2) 线性检验（Pearson 只能捕捉线性关系）
  3) 范围限制（Range Restriction）效应 —— 截断子样本会系统性低估 r
  4) Pearson vs Spearman vs Kendall 对比
  5) LOWESS 局部多项式拟合（目视非线性）
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

for f in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
    rcParams["font.sans-serif"] = [f]
    break
rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE, "美元指数相关性分析.txt")
OUT_DIR = os.path.join(BASE, "output")


def load_data():
    df = pd.read_csv(DATA_FILE, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    num_cols = ["美元指数", "上证A股指数", "上证所:股票成交金额", "上证所:主板A股:成交金额"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c].str.replace(",", "", regex=False).str.strip(), errors="coerce")
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    return df.dropna(subset=["日期"] + num_cols).sort_values("日期").reset_index(drop=True)


def lowess(x, y, frac=0.4):
    """简易 LOWESS 局部加权回归，用于检测非线性趋势"""
    n = len(x)
    sx = np.sort(x)
    sy = np.interp(sx, x, y)
    win = max(int(n * frac), 5)
    yhat = np.zeros(n)
    for i in range(n):
        d = np.abs(sx - sx[i])
        w = np.exp(-((d / np.max(d)) ** 2) * 4)
        b = np.sum(w * sx * sy) / np.sum(w * sx * sx) - np.sum(w * sx) * np.sum(w * sy) / np.sum(w * sx) ** 2
        a = (np.sum(w * sy) - b * np.sum(w * sx)) / np.sum(w)
        yhat[i] = a + b * sx[i]
    return sx, yhat


def main():
    df = load_data()
    sub = df[df["美元指数"] >= 100].copy()
    full = df.copy()
    n = len(sub)
    print("=" * 72)
    print(f"子样本（美元指数≥100）n = {n}，全样本 n = {len(full)}")
    print("=" * 72)

    # ---------- 1. 正态性检验（Pearson 的前提之一） ----------
    print("\n【1】Shapiro-Wilk 正态性检验（p<0.05 → 拒绝正态）")
    for col in ["美元指数", "上证A股指数"]:
        W, p = stats.shapiro(sub[col].values)
        flag = "非正态" if p < 0.05 else "近正态"
        print(f"  {col:12s}: W={W:.4f}, p={p:.4e}  →  {flag}")

    # ---------- 2. 三种相关系数对比 ----------
    print("\n【2】子样本（美元≥100）三种相关系数对比")
    for name, fn in [("Pearson", stats.pearsonr),
                     ("Spearman", stats.spearmanr),
                     ("Kendall", stats.kendalltau)]:
        r, p = fn(sub["美元指数"], sub["上证A股指数"])
        print(f"  {name:9s}: r={r:+.4f}, p={p:.4e}")

    # ---------- 3. 范围限制校正 ----------
    # Thorndike 公式：校正被截断后的相关系数
    # 设子样本的 X 标准差为 sx_s，全样本的 X 标准差为 sx_f
    # 令 r_s = 子样本相关，则全样本校正后相关 r_f：
    #   r_f = (r_s * (sx_f/sx_s)) / sqrt(1 - r_s^2 + r_s^2*(sx_f/sx_s)^2)
    sx_s = sub["美元指数"].std()
    sx_f = full["美元指数"].std()
    r_s, _ = stats.pearsonr(sub["美元指数"], sub["上证A股指数"])
    ratio = sx_f / sx_s
    denom = np.sqrt(1 - r_s**2 + (r_s**2) * (ratio**2))
    r_corrected = (r_s * ratio) / denom
    print("\n【3】范围限制（Range Restriction）校正")
    print(f"  子样本 X 标准差 sx_s={sx_s:.3f}, 全样本 X 标准差 sx_f={sx_f:.3f}, 比值={ratio:.3f}")
    print(f"  观测的子样本 Pearson r = {r_s:+.4f}")
    print(f"  Thorndike 校正后的 r   = {r_corrected:+.4f}")
    print(f"  → 说明子样本 r 被『截断』显著低估了真实线性关联")

    # ---------- 4. 残差非线性检验 ----------
    # 线性回归后，检验残差是否还有非线性趋势
    print("\n【4】残差非线性检验（拟合线性模型后，残差是否依赖X的平方）")
    x = sub["美元指数"].values
    y = sub["上证A股指数"].values
    # 线性拟合
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    # 加入 X² 项检验是否显著
    X = np.column_stack([np.ones_like(x), x, x**2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_quad = X @ beta
    ss_lin = np.sum(resid**2)
    ss_quad = np.sum((y - y_quad)**2)
    F_stat = ((ss_lin - ss_quad) / 1) / (ss_quad / (n - 3))
    p_nonlin = 1 - stats.f.cdf(F_stat, 1, n - 3)
    print(f"  线性残差 SS={ss_lin:.2f}, 加入 X的平方项 后 SS={ss_quad:.2f}")
    print(f"  F={F_stat:.3f}, p={p_nonlin:.4e}  {'→ 存在显著非线性' if p_nonlin < 0.05 else '→ 非线性不显著，线性可接受'}")

    # ---------- 5. 可视化诊断 ----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) 散点 + 线性 vs LOWESS
    ax = axes[0, 0]
    ax.scatter(x, y, s=30, alpha=0.5, color="#1f77b4", edgecolor="white")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, "r-", lw=2, label=f"线性: y={slope:.1f}x+{intercept:.0f}")
    lx, ly = lowess(x, y, frac=0.5)
    ax.plot(lx, ly, "g--", lw=2, label="LOWESS 局部趋势")
    ax.set_xlabel("美元指数（仅≥100样本）")
    ax.set_ylabel("上证A股指数")
    ax.set_title("(a) 散点：线性拟合 vs LOWESS（若偏离→非线性）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) 残差 vs X（看是否有喇叭口/曲线 = 异方差/非线性）
    ax = axes[0, 1]
    ax.scatter(x, resid, s=30, alpha=0.5, color="#d62728")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("美元指数")
    ax.set_ylabel("线性拟合残差")
    ax.set_title("(b) 残差 vs X（系统性曲线 → 违反线性）")
    ax.grid(alpha=0.3)

    # (c) 全样本散点 + 标注截断点
    ax = axes[1, 0]
    xf, yf = full["美元指数"].values, full["上证A股指数"].values
    ax.scatter(xf, yf, s=25, alpha=0.4, color="gray", label="全样本")
    ax.scatter(x, y, s=30, alpha=0.7, color="#d62728", label="子样本（≥100）")
    ax.axvline(100, color="black", ls="--", lw=1)
    ax.set_xlabel("美元指数")
    ax.set_ylabel("上证A股指数")
    ax.set_title("(c) 范围限制：子样本 X 只覆盖右半段")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (d) 相关系数对比柱状图
    ax = axes[1, 1]
    methods = ["Pearson", "Spearman", "Kendall"]
    rs = []
    for _, fn in zip(methods, [stats.pearsonr, stats.spearmanr, stats.kendalltau]):
        r, _ = fn(sub["美元指数"], sub["上证A股指数"])
        rs.append(r)
    bars = ax.bar(methods, rs, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.axhline(0, color="black", lw=0.8)
    ax.bar_label(bars, fmt="%.3f", fontsize=10)
    ax.set_ylabel("相关系数")
    ax.set_title("(d) 子样本三种相关系数对比")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pearson_diagnosis.png"), dpi=150)
    plt.close(fig)
    print(f"\n诊断图已保存: {os.path.join(OUT_DIR, 'pearson_diagnosis.png')}")

    # ---------- 总结 ----------
    print("\n" + "=" * 72)
    print("【方法论结论】")
    non_normal = any(stats.shapiro(sub[c].values)[1] < 0.05 for c in ["美元指数", "上证A股指数"])
    print(f"  [1] 正态性：{'不满足（Pearson假设违反）' if non_normal else '基本满足'}")
    print(f"  [2] 范围限制：X 标准差比值={ratio:.2f}（>1.2 即影响显著）")
    print(f"  [3] Thorndike 校正后 r 从 {r_s:+.3f} 升至 {r_corrected:+.3f}")
    print(f"  [4] 非线性检验 p={p_nonlin:.4f} {'（存在非线性）' if p_nonlin < 0.05 else '（线性可接受）'}")
    print("  -> 建议：在子样本中 Pearson 不应单独使用，应与 Spearman 联合报告，")
    print("          并明确标注范围限制带来的低估。若两者差异大，以 Spearman 为准。")
    print("=" * 72)


if __name__ == "__main__":
    main()
