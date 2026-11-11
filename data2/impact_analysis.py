# -*- coding: utf-8 -*-
"""
美元指数强弱对上证A股的综合影响分析（稳健方法族）
聚焦"过100点"的真实影响曲线，规避 Pearson 在截断子样本中的缺陷。
方法：
  1) 强度梯度分箱：把美元指数分成多个强度区间，看 A 股均值/中位数/分位数差异
  2) LOWESS 非线性平滑：直接刻画 A 股水平随美元的连续变化曲线（找拐点）
  3) 分位数回归（IRLS 实现）：看美元对 A 股不同分位的非对称冲击
  4) 滞后/领先交叉相关：美元是领先、同步还是滞后于 A 股
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


# ---------- LOWESS 局部加权回归 ----------
def lowess(x, y, frac=0.3, it=3):
    """简易 LOWESS：高斯核局部线性回归 + 残差重加权（抑制离群）"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    order = np.argsort(x)
    xs = x[order]; ys = y[order]
    n = len(xs)
    bw = max(int(n * frac), 5)
    rob_w = np.ones(n)
    yhat = np.zeros(n)
    for _ in range(it):
        for i in range(n):
            lo = max(0, i - bw); hi = min(n, i + bw + 1)
            dx = xs[lo:hi] - xs[i]
            w = np.exp(-0.5 * (dx / (dx.std() + 1e-9)) ** 2) * rob_w[lo:hi]
            X = np.column_stack([np.ones_like(dx), dx])
            W = np.diag(w)
            beta = np.linalg.solve(X.T @ W @ X + 1e-8 * np.eye(2), X.T @ W @ ys[lo:hi])
            yhat[i] = beta[0]
        # 残差重加权（bisquare）
        resid = ys - yhat
        s = np.median(np.abs(resid - np.median(resid))) + 1e-9
        rob_w = (1 - (resid / (6 * s)) ** 2).clip(0, 1) ** 2
    return xs, yhat


# ---------- 分位数回归（IRLS: 迭代加权最小二乘） ----------
def quantile_reg(X, y, tau=0.5, max_iter=100, tol=1e-6):
    """
    分位数回归，X 含截距列。返回 beta。
    原理：分位数回归的损失为检查函数 rho_tau(r)=r*(tau-I(r<0))。
    用 IRLS：权重 w_i = 1 / max(|r_i|, eps) 迭代逼近。
    """
    n, k = X.shape
    # 初始用 OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    eps = 1e-4 * (np.abs(y).mean() + 1e-9)
    for _ in range(max_iter):
        r = y - X @ beta
        # 检查函数的 IRLS 权重（等价于 Frisch 内点法近似）
        w = np.where(r >= 0, tau, 1 - tau) / np.maximum(np.abs(r), eps)
        W = np.diag(w)
        try:
            beta_new = np.linalg.solve(X.T @ W @ X + 1e-8 * np.eye(k), X.T @ W @ y)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


# ---------- 滞后/领先交叉相关 ----------
def cross_corr(x, y, max_lag=15):
    """
    计算 x 领先/滞后 y 的互相关系数。
    lag>0 表示 x 领先 y（用过去 x 解释现在 y）。
    返回 lag, r 数组。
    """
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    rs = []
    for lag in lags:
        if lag >= 0:
            a = x[:n - lag] if lag > 0 else x
            b = y[lag:] if lag > 0 else y
        else:
            a = x[-lag:]
            b = y[:n + lag]
        rs.append(np.corrcoef(a, b)[0, 1])
    return lags, np.array(rs)


def main():
    df = load_data()
    print(f"样本 {len(df)} 行，{df['日期'].min().date()} ~ {df['日期'].max().date()}")
    x = df["美元指数"].values
    y = df["上证A股指数"].values

    # ========== 1. 强度梯度分箱 ==========
    print("\n" + "=" * 72)
    print("【1】美元强度梯度分箱（看 A 股在每个强度区间的表现）")
    print("=" * 72)
    bins = [0, 98, 99, 99.5, 100, 100.5, 101, 101.5, 102, 200]
    labels = ["<98", "98-99", "99-99.5", "99.5-100", "100-100.5", "100.5-101",
              "101-101.5", "101.5-102", ">102"]
    df["强度档"] = pd.cut(df["美元指数"], bins=bins, labels=labels, include_lowest=True)
    print(f"{'区间':>12} {'n':>5} {'A股均值':>9} {'中位':>8} {'P25':>8} {'P75':>8} {'成交额均值':>11}")
    for lab in labels:
        g = df[df["强度档"] == lab]
        if len(g) == 0:
            continue
        sh = g["上证A股指数"]
        amt = g["上证所:股票成交金额"]
        print(f"{lab:>12} {len(g):>5} {sh.mean():>9.1f} {sh.median():>8.1f} "
              f"{sh.quantile(.25):>8.1f} {sh.quantile(.75):>8.1f} {amt.mean():>11.0f}")

    # 边际效应：每跨过 0.5 的 A 股中位数变化
    print("\n  -> 中位数边际变化（每跨0.5档，A股中位点数变化）:")
    meds = []
    for lab in labels:
        g = df[df["强度档"] == lab]
        if len(g) > 0:
            meds.append((lab, g["上证A股指数"].median()))
    for i in range(1, len(meds)):
        d = meds[i][1] - meds[i - 1][1]
        flag = " <<< 跳水" if d < -200 else ""
        print(f"     {meds[i-1][0]} -> {meds[i][0]}: {d:+.1f} 点{flag}")

    # ========== 2. LOWESS 非线性曲线（找拐点） ==========
    print("\n" + "=" * 72)
    print("【2】LOWESS 非线性影响曲线（找真实拐点位置）")
    print("=" * 72)
    lx, ly = lowess(x, y, frac=0.25, it=3)
    # 找一阶差分最小处（下降最陡）
    dly = np.gradient(ly, lx)
    idx_steepest = np.argmin(dly)
    print(f"  LOWESS 曲线下降最陡位置：美元指数 = {lx[idx_steepest]:.2f}")
    print(f"  该处 A 股局部水平 = {ly[idx_steepest]:.0f}，局部斜率 = {dly[idx_steepest]:.1f} 点/1美元指数")
    # 在 100 前后局部斜率对比
    for c in [99, 99.5, 100, 100.5, 101, 101.5, 102]:
        j = np.argmin(np.abs(lx - c))
        print(f"  美元={lx[j]:.2f}: A股局部水平={ly[j]:.0f}, 局部斜率={dly[j]:+.1f}")

    # ========== 3. 分位数回归（非对称冲击） ==========
    print("\n" + "=" * 72)
    print("【3】分位数回归：美元指数对 A 股不同分位的非对称冲击")
    print("=" * 72)
    X = np.column_stack([np.ones_like(x), x])
    print(f"{'分位tau':>9} {'截距':>9} {'美元系数':>10} {'含义':>30}")
    for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
        b = quantile_reg(X, y, tau=tau)
        print(f"{tau:>9.2f} {b[0]:>9.1f} {b[1]:>+10.2f} "
              f"{'美元+1 -> A股' + f'{b[1]:+.0f}点':>30}")
    print("  解读：若低分位系数绝对值 > 高分位，说明美元走强更多打压'弱市'（下行风险放大）。")

    # ========== 4. 滞后/领先交叉相关 ==========
    print("\n" + "=" * 72)
    print("【4】滞后/领先交叉相关（美元 vs A股，lag>0=美元领先）")
    print("=" * 72)
    lags, rs = cross_corr(df["美元指数"].values, df["上证A股指数"].values, max_lag=15)
    best_idx = np.argmax(np.abs(rs))
    print(f"  最强相关：lag={lags[best_idx]:+d}（{'美元领先' if lags[best_idx] > 0 else '美元滞后' if lags[best_idx] < 0 else '同步'} {abs(lags[best_idx])} 天），r={rs[best_idx]:+.3f}")
    print(f"  各滞后 r：")
    for L in [-5, -3, 0, 1, 2, 3, 5, 7, 10]:
        j = np.where(lags == L)[0]
        if len(j):
            print(f"    lag={L:+d}: r={rs[j[0]]:+.3f}")

    # ========== 可视化 ==========
    plot_all(df, lx, ly, X, y, lags, rs)


def plot_all(df, lx, ly, X, y, lags, rs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) 分箱均值/中位 + LOWESS 曲线
    ax = axes[0, 0]
    bins = [0, 98, 99, 99.5, 100, 100.5, 101, 101.5, 102, 200]
    labels = ["<98", "98-99", "99-99.5", "99.5-100", "100-100.5", "100.5-101",
              "101-101.5", "101.5-102", ">102"]
    df["强度档"] = pd.cut(df["美元指数"], bins=bins, labels=labels, include_lowest=True)
    centers = []
    means, meds, ns = [], [], []
    for lab in labels:
        g = df[df["强度档"] == lab]
        if len(g) > 0:
            c_mid = (df.loc[df["强度档"] == lab, "美元指数"].mean())
            centers.append(c_mid); means.append(g["上证A股指数"].mean())
            meds.append(g["上证A股指数"].median()); ns.append(len(g))
    ax.plot(centers, means, "o-", color="#1f77b4", lw=2, label="均值")
    ax.plot(centers, meds, "s-", color="#d62728", lw=2, label="中位数")
    ax.plot(lx, ly, "-", color="#2ca02c", lw=2.5, alpha=0.8, label="LOWESS 平滑")
    for cx, n in zip(centers, ns):
        ax.annotate(f"n={n}", (cx, min(means) - 50), fontsize=8, ha="center", alpha=0.6)
    ax.axvline(100, color="black", ls="--", lw=1, alpha=0.5, label="整数100")
    ax.axvline(101, color="purple", ls=":", lw=1.5, alpha=0.7, label="真实拐点~101")
    ax.set_xlabel("美元指数")
    ax.set_ylabel("上证A股指数")
    ax.set_title("(a) 强度梯度：A股水平随美元强弱的变化")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # (b) 分位数回归线
    ax = axes[0, 1]
    ax.scatter(df["美元指数"], df["上证A股指数"], s=12, alpha=0.25, color="gray")
    xs = np.linspace(df["美元指数"].min(), df["美元指数"].max(), 100)
    Xs = np.column_stack([np.ones(100), xs])
    colors = {0.1: "#d62728", 0.25: "#ff7f0e", 0.5: "#2ca02c", 0.75: "#1f77b4", 0.9: "#9467bd"}
    for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
        b = quantile_reg(X, y, tau=tau)
        ax.plot(xs, Xs @ b, color=colors[tau], lw=1.5, label=f"Q{int(tau*100)}")
    ax.axvline(100, color="black", ls="--", lw=1, alpha=0.5)
    ax.axvline(101, color="purple", ls=":", lw=1.5, alpha=0.7)
    ax.set_xlabel("美元指数")
    ax.set_ylabel("上证A股指数")
    ax.set_title("(b) 分位数回归：美元对A股不同分位的非对称冲击")
    ax.legend(fontsize=8, ncol=5, loc="lower center")
    ax.grid(alpha=0.3)

    # (c) LOWESS 局部斜率（边际效应曲线）
    ax = axes[1, 0]
    dly = np.gradient(ly, lx)
    ax.plot(lx, dly, "-", color="#d62728", lw=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(100, color="black", ls="--", lw=1, alpha=0.5, label="100")
    ax.axvline(101, color="purple", ls=":", lw=1.5, alpha=0.7, label="101")
    ax.fill_between(lx, dly, 0, where=(dly < 0), color="#d62728", alpha=0.15)
    ax.set_xlabel("美元指数")
    ax.set_ylabel("局部斜率（A股点数/1美元指数）")
    ax.set_title("(c) 边际效应曲线：美元每+1对A股的冲击强度")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (d) 滞后交叉相关
    ax = axes[1, 1]
    ax.bar(lags, rs, width=0.6,
           color=["#d62728" if L > 0 else "#1f77b4" if L < 0 else "#2ca02c" for L in lags])
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", ls="--", lw=1)
    best_idx = np.argmax(np.abs(rs))
    ax.annotate(f"最强: lag={lags[best_idx]:+d}, r={rs[best_idx]:+.3f}",
                (lags[best_idx], rs[best_idx]),
                textcoords="offset points", xytext=(10, 10), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_xlabel("滞后天数（lag>0 = 美元领先A股）")
    ax.set_ylabel("Pearson r")
    ax.set_title("(d) 滞后/领先交叉相关：美元领先 vs 同步")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "impact_curve.png"), dpi=150)
    plt.close(fig)
    print(f"\n图表已保存: {os.path.join(OUT_DIR, 'impact_curve.png')}")


if __name__ == "__main__":
    main()
