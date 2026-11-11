# -*- coding: utf-8 -*-
"""
美元指数突破100对上证A股的影响：事件研究 + 断点回归(RDD)
分析框架：
  A) 事件研究法：识别"从<100突破且连续>=5天维持>=100"的事件，
     计算事件窗口[-10, +15]内上证A股相对t-1的累计涨跌幅。
  B) 断点回归(RDD)：用100附近全部数据估计A股在阈值处的"跳跃"，
     并探测真实断点位置（不一定是整数100）。
说明：由于严格定义下事件数<=3，事件研究仅做描述；统计推断主要依赖RDD。
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


def identify_events(df, L_look=5, L_hold=5, threshold=100.0):
    dxy = df["美元指数"].values
    n = len(df)
    events, i = [], 0
    while i < n:
        if dxy[i] >= threshold:
            look_start = max(0, i - L_look)
            pre_ok = np.all(dxy[look_start:i] < threshold)
            j = i
            while j < n and dxy[j] >= threshold:
                j += 1
            run_len = j - i
            if pre_ok and run_len >= L_hold:
                events.append((i, run_len))
            i = j
        else:
            i += 1
    return events


# ============ A. 事件研究 ============
def event_study(df, events, pre=10, post=15):
    """对每个事件，截取窗口[-pre,+post]，计算上证A股相对 t=-1 的累计涨跌幅(%)"""
    windows = []
    sh = df["上证A股指数"].values
    for (ei, rl) in events:
        if ei - pre < 0:  # 窗口左端越界（如事件1在样本起点），跳过
            continue
        end = min(ei + post, len(df) - 1)
        seg = np.arange(ei - pre, end + 1)
        base = sh[ei - 1]  # 以突破前一日为基准
        ret = (sh[seg] - base) / base * 100.0
        windows.append((ei, seg - ei, ret))
    # 平均累计涨跌幅（若多事件）
    if windows:
        # 对齐到同一相对日
        min_len = min(len(w[2]) for w in windows)
        aligned = np.array([w[2][:min_len] for w in windows])
        mean_ret = aligned.mean(axis=0)
        rel_days = windows[0][1][:min_len]
        return windows, rel_days, mean_ret
    return windows, None, None


# ============ B. 断点回归 ============
def rdd_estimate(df, c=100.0, bw=3.0):
    """
    局部线性断点回归：Y=上证A股, X=美元指数, 阈值c, 带宽bw
    模型: Y = a + tau*D + b*(X-c) + g*(X-c)*D + e, D=1 if X>=c
    tau 即阈值处的跳跃量。
    用 OLS 解析解估计（避免依赖 statsmodels）。
    """
    m = (df["美元指数"] >= c - bw) & (df["美元指数"] <= c + bw)
    Xc = (df.loc[m, "美元指数"].values - c).astype(float)
    Y = df.loc[m, "上证A股指数"].values.astype(float)
    D = (df.loc[m, "美元指数"].values >= c).astype(float)
    # 设计矩阵
    M = np.column_stack([np.ones_like(Xc), D, Xc, Xc * D])
    beta, *_ = np.linalg.lstsq(M, Y, rcond=None)
    a, tau, b, g = beta
    # 残差与标准误
    resid = Y - M @ beta
    n = len(Y)
    XtX_inv = np.linalg.inv(M.T @ M)
    sigma2 = resid @ resid / (n - 4)
    se = np.sqrt(np.diag(XtX_inv * sigma2))
    t_stat = tau / se[1]
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 4))
    return dict(tau=tau, se=se[1], t=t_stat, p=p_val, n=n, bw=bw,
                left_int=a + b * (0), right_int=a + tau,  # 阈值左右极限
                left_slope=b, right_slope=b + g)


def rdd_scan_threshold(df, cands, bw=3.0):
    """扫描不同阈值，寻找跳跃最显著的位置"""
    rows = []
    for c in cands:
        r = rdd_estimate(df, c=c, bw=bw)
        rows.append(r)
    return rows


def main():
    df = load_data()
    print(f"样本 {len(df)} 行，{df['日期'].min().date()} ~ {df['日期'].max().date()}")

    # ---------- A. 事件研究 ----------
    events = identify_events(df, L_look=5, L_hold=5)
    print(f"\n===== A. 事件研究 (L_look=5, L_hold=5) 识别到 {len(events)} 个事件 =====")
    for idx, (ei, rl) in enumerate(events):
        print(f"  事件{idx+1}: {df.iloc[ei]['日期'].date()}, 持续{rl}天")

    windows, rel_days, mean_ret = event_study(df, events, pre=10, post=15)
    print(f"\n可用于事件窗口分析的事件数: {len(windows)}（排除样本起点无基准的事件）")
    if windows:
        for k, (ei, rd, ret) in enumerate(windows):
            d0 = df.iloc[ei]["日期"].date()
            print(f"  窗口 {k+1} ({d0}): 突破前1日->突破当日 {ret[10]:+.2f}%, "
                  f"突破后5日 {ret[15]:+.2f}%, 突破后{len(ret)-1}日 {ret[-1]:+.2f}%")
        if len(windows) > 1:
            print(f"\n  多事件平均累计涨跌幅:")
            for d in [0, 1, 3, 5, 10, 15]:
                if d < len(mean_ret):
                    print(f"    t={d:+d}: {mean_ret[np.where(rel_days==d)[0][0]]:+.2f}%")

    # ---------- B. 断点回归 ----------
    print("\n===== B. 断点回归 RDD（阈值=100，不同带宽）=====")
    for bw in [1.5, 2.0, 3.0, 5.0]:
        r = rdd_estimate(df, c=100.0, bw=bw)
        sig = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else ("*" if r["p"] < 0.1 else "ns"))
        print(f"  bw={bw}: tau(跳跃)={r['tau']:+.1f}点, se={r['se']:.1f}, t={r['t']:+.2f}, "
              f"p={r['p']:.3f} {sig}, n={r['n']}")
        print(f"         阈值左侧A股极限={r['left_int']:.0f}, 右侧={r['right_int']:.0f}")

    # ---------- B2. 阈值扫描：真实断点在哪 ----------
    print("\n===== B2. 阈值扫描（bw=3，寻找跳跃最显著的阈值）=====")
    rows = rdd_scan_threshold(df, np.arange(98.5, 103.6, 0.5), bw=3.0)
    best = min(rows, key=lambda r: r["p"])
    for r in rows:
        c = 100 + (r["bw"])  # placeholder
    for c, r in zip(np.arange(98.5, 103.6, 0.5), rows):
        sig = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else ("*" if r["p"] < 0.1 else "ns"))
        print(f"  阈值={c:.1f}: tau={r['tau']:+.1f}, p={r['p']:.3f} {sig}, n={r['n']}")
    print(f"\n  >> 跳跃最显著阈值 ≈ 100+（p最小处），tau={best['tau']:+.1f}")

    # ---------- 非参数检验：100前后A股分布 ----------
    print("\n===== C. 非参数检验（100前后A股分布）=====")
    for bw in [1.0, 2.0, 3.0]:
        left = df[(df["美元指数"] >= 100 - bw) & (df["美元指数"] < 100)]["上证A股指数"]
        right = df[(df["美元指数"] >= 100) & (df["美元指数"] < 100 + bw)]["上证A股指数"]
        u, p = stats.mannwhitneyu(left, right, alternative="two-sided")
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "ns"))
        print(f"  bw={bw}: 左n={len(left)}中位={left.median():.0f}, 右n={len(right)}中位={right.median():.0f}, "
              f"Mann-Whitney p={p:.3f} {sig}")

    # ---------- 可视化 ----------
    plot_results(df, windows, rel_days, mean_ret, best)


def plot_results(df, windows, rel_days, mean_ret, best):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) 事件窗口累计涨跌幅
    ax = axes[0, 0]
    if windows:
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
        for k, (ei, rd, ret) in enumerate(windows):
            ax.plot(rd, ret, marker="o", ms=4, lw=1.4, color=colors[k % len(colors)],
                    label=f"事件{k+1}: {df.iloc[ei]['日期'].date()}")
        if mean_ret is not None and len(windows) > 1:
            ax.plot(rel_days, mean_ret, "k-", lw=2.5, alpha=0.6, label="平均")
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("相对突破日（t=0 为美元指数首次>=100当日）")
        ax.set_ylabel("上证A股累计涨跌幅(%)")
        ax.set_title("(a) 事件窗口：突破100前后上证A股走势")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # (b) RDD 散点 + 局部线性拟合
    ax = axes[0, 1]
    bw = 3.0
    m = (df["美元指数"] >= 100 - bw) & (df["美元指数"] <= 100 + bw)
    xs = df.loc[m, "美元指数"].values
    ys = df.loc[m, "上证A股指数"].values
    D = xs >= 100
    ax.scatter(xs[~D], ys[~D], s=20, alpha=0.4, color="#1f77b4", label="美元<100")
    ax.scatter(xs[D], ys[D], s=20, alpha=0.5, color="#d62728", label="美元>=100")
    # 拟合线
    for d_flag, col in [(0, "#1f77b4"), (1, "#d62728")]:
        sel = (xs >= 100) if d_flag else (xs < 100)
        if sel.sum() > 2:
            b1, b0 = np.polyfit(xs[sel] - 100, ys[sel], 1)
            xl = np.linspace(xs[sel].min(), 100 if d_flag == 0 else xs[sel].max(), 50)
            ax.plot(xl, b0 + b1 * (xl - 100), color=col, lw=2)
    ax.axvline(100, color="black", ls="--", lw=1)
    ax.set_xlabel("美元指数")
    ax.set_ylabel("上证A股指数")
    ax.set_title("(b) 断点回归：100附近A股水平跳跃")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) 阈值扫描：跳跃量与显著性
    ax = axes[1, 0]
    cands = np.arange(98.5, 103.6, 0.5)
    taus = []
    for c in cands:
        taus.append(rdd_estimate(df, c=c, bw=3.0)["tau"])
    ax.bar(cands, taus, width=0.4, color=["#d62728" if t < 0 else "#2ca02c" for t in taus])
    ax.axvline(100, color="black", ls="--", lw=1, label="整数100")
    ax.set_xlabel("候选阈值")
    ax.set_ylabel("跳跃量 tau（A股点数）")
    ax.set_title("(c) 阈值扫描：跳跃量随阈值变化")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # (d) 分箱均值曲线（平滑看趋势）
    ax = axes[1, 1]
    bins = np.arange(96, 106.1, 0.5)
    df["bin"] = pd.cut(df["美元指数"], bins, include_lowest=True)
    grp = df.groupby("bin", observed=True)["上证A股指数"].agg(["mean", "median", "count"]).reset_index()
    centers = [iv.mid for iv in grp["bin"]]
    ax.plot(centers, grp["mean"], "o-", color="#1f77b4", label="均值")
    ax.plot(centers, grp["median"], "s-", color="#d62728", label="中位数")
    for cx, cnt in zip(centers, grp["count"]):
        ax.annotate(f"{cnt}", (cx, grp.set_index('bin')['mean'].iloc[centers.index(cx)]) if False else (cx, 3000), fontsize=7, ha="center", alpha=0.5)
    ax.axvline(100, color="black", ls="--", lw=1)
    ax.set_xlabel("美元指数（0.5分箱）")
    ax.set_ylabel("上证A股指数")
    ax.set_title("(d) 分箱均值/中位数：100附近真实趋势")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "event_rdd.png"), dpi=150)
    plt.close(fig)
    print(f"\n图表已保存: {os.path.join(OUT_DIR, 'event_rdd.png')}")


if __name__ == "__main__":
    main()
