# -*- coding: utf-8 -*-
"""
项目可视化分析套件（配套 模型迭代报告_BiGRU.md）
=====================================================
三类图像，统一风格 / 中文标注 / 透明背景 PNG(≥1080P) + SVG 矢量双输出：

  [支撑图]
    fig01 原始数据样本图      —— 收益率走势 / 分布 / 特征热力图 / 特征相关矩阵
    fig02 预处理过程可视化    —— 10天标签生成示意 / 28-fold滚动验证切分 / 死区样本权重
    fig03 模型结构示意图      —— BiGRU→Inception→SelfAttention→GAP→FC 全流程

  [预测 vs 真实对比图]
    fig04 时间序列对比折线图  —— 真实利率走势(下行区间着色) vs 模型概率/预测正误
    fig05 散点拟合分布图      —— 概率-标签散点(统一[0,1]坐标) + 分箱校准曲线
                                 标注 MSE(Brier)/R²(BSS)/LogLoss/AUC
    fig06 误差热力图          —— 混淆矩阵 + 年份×预测方向 准确率热力图

  [补充分析图]
    fig07 评估指标雷达图      —— XGB基线 / 5天最优 / 10天FINAL5 / +固定阈值0.39
    fig08 参数性能对比柱状图  —— horizon / seq_len / 训练窗口 三面板
    fig09 误差分布箱线图      —— 各配置逐fold准确率 + 分年概率误差

数据来源: model_new/data/(rate.csv, 特征, XGB fold指标) + model_nn/out/(NN预测与fold指标)
运行: python 03_可视化分析.py          （全部图像输出至 model_nn/figs/）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss

# ============================================================
# 全局统一风格（所有图共用，保证风格一致、可二次编辑）
# ============================================================
BASE = Path(r"f:\43_华泰证券预处理\43_华泰证券预处理")
DATA = BASE / "model_new" / "data"
OUT = BASE / "model_nn" / "out"
FIGS = BASE / "model_nn" / "figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 100, "savefig.dpi": 200,          # 12.8x7.2 @200dpi = 2560x1440 (>1080P)
    "axes.edgecolor": "#666666", "axes.labelcolor": "#333333",
    "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "legend.frameon": True, "legend.framealpha": 0.75, "legend.edgecolor": "#CCCCCC",
    "axes.grid": True, "grid.color": "#D9D9D9", "grid.linewidth": 0.6, "grid.alpha": 0.65,
})

# 统一配色（与项目报告一致：蓝=真实/主, 红=误差/预测, 绿=正确, 橙=辅助, 紫=对比模型）
C = {"blue": "#2C7FB8", "red": "#E34A33", "green": "#41AB5D",
     "orange": "#FEB24C", "purple": "#8C6BB1", "gray": "#969696", "dark": "#253494"}
CMAP_BIN = LinearSegmentedColormap.from_list("proj", ["#FFF5F0", "#FDBB84", "#E34A33", "#8B0000"])
CMAP_BLUE = LinearSegmentedColormap.from_list("projb", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5"])


def save(name: str, fig) -> None:
    """统一输出: 透明背景 PNG(≥1080P) + SVG 矢量（可二次编辑）"""
    fig.savefig(FIGS / f"{name}.png", transparent=True, bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {name}.png/.svg", flush=True)


def newfig(w=12.8, h=7.2):
    return plt.subplots(figsize=(w, h))


# ============================================================
# 公共数据加载
# ============================================================
print("[加载] 数据 ...", flush=True)
rate = pd.read_csv(DATA / "rate.csv", parse_dates=["time"]).rename(columns={"time": "date"})
feats = pd.read_csv(DATA / "xgb_train_features_bin.csv").drop(columns=["Unnamed: 0"], errors="ignore")
pred = pd.read_csv(OUT / "H10_S40_FINAL5_nn_test_predictions.csv",
                   parse_dates=["date"]).sort_values("date").reset_index(drop=True)
pred["year"] = pred["date"].dt.year
folds = pd.read_csv(OUT / "H10_S40_FINAL5_nn_fold_metrics.csv", parse_dates=["date_start", "date_end"])


# ============================================================
# 阈值层复算（与修正后的 02_阈值后校准_2025.py 一致）
#   方案A: 固定阈值 th=0.39（≤2024历史最优, 无泄漏, 交付采用）
#   滚动校准: W=250, lag=10 防标签前视（已验证无效, 仅画阈值漂移路径作诊断）
# ============================================================
FIX_TH = 0.39
pred["pred_fix"] = (pred["proba_down"] >= FIX_TH).astype(int)
calib_acc = accuracy_score(pred["y_true"], pred["pred_fix"])
calib_f1 = f1_score(pred["y_true"], pred["pred_fix"], average="macro")
p25 = pred[pred["year"] == 2025]
calib_2025 = accuracy_score(p25["y_true"], p25["pred_fix"])
print(f"  固定th={FIX_TH}: 全局Acc={calib_acc:.4f} F1={calib_f1:.4f} 2025Acc={calib_2025:.4f}", flush=True)


def rolling_threshold(df: pd.DataFrame, W: int = 250, lag: int = 10, ths=None):
    ths = ths if ths is not None else np.round(np.arange(0.35, 0.6501, 0.01), 2)
    p, yv = df["proba_down"].to_numpy(), df["y_true"].to_numpy()
    used = np.full(len(df), np.nan)
    pv = df["proba_down"].to_numpy()
    correct = ((pv[None, :] >= ths[:, None]).astype(int) == yv[None, :]).astype(int)
    cum = np.zeros((len(ths), len(df) + 1), dtype=int)
    cum[:, 1:] = np.cumsum(correct, axis=1)
    for i in range(len(df)):
        hi, lo = max(0, i - lag), max(0, max(0, i - lag) - W)
        if hi - lo < 30:
            continue
        k = int(np.argmax((cum[:, hi] - cum[:, lo]) / (hi - lo)))
        used[i] = ths[k]
    return used


pred["th_calib"] = rolling_threshold(pred)

# ============================================================
# fig01 原始数据样本图
# ============================================================
print("[图1] 原始数据样本图 ...", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.4))
ax = axes[0, 0]
ax.plot(rate["date"], rate["rate"], color=C["blue"], lw=1.0)
ax.axvspan(pd.Timestamp("2018-01-01"), pd.Timestamp("2026-01-01"), color=C["orange"], alpha=0.10)
ax.annotate("样本外验证区间\n(2018-2026, 28折滚动)", xy=(pd.Timestamp("2022-01-01"), 4.15),
            xytext=(pd.Timestamp("2013-06-01"), 4.35), fontsize=10.5, color="#8B4513",
            arrowprops=dict(arrowstyle="->", color="#8B4513", lw=1.2))
ax.set_title("(a) 10年期国债收益率全期走势（2007-12 至 2026）")
ax.set_xlabel("日期"); ax.set_ylabel("收益率 (%)")
ax.annotate("2008金融危机\n2020疫情\n2022-2023加息", xy=(pd.Timestamp("2015-06-01"), 3.05),
            fontsize=9.5, color=C["gray"], ha="center")

ax = axes[0, 1]
ax.hist(rate["rate"].dropna(), bins=60, color=C["blue"], alpha=0.85, edgecolor="white")
ax.axvline(rate["rate"].mean(), color=C["red"], ls="--", lw=1.4,
           label=f"均值 {rate['rate'].mean():.2f}%")
ax.set_title("(b) 收益率水平分布直方图")
ax.set_xlabel("收益率 (%)"); ax.set_ylabel("频数"); ax.legend(loc="upper right")

ax = axes[1, 0]
samp = feats.iloc[-120:, :14]           # 最近120交易日 × 前14特征
im = ax.imshow(samp.T, aspect="auto", cmap=CMAP_BLUE, interpolation="nearest")
ax.set_yticks(range(14)); ax.set_yticklabels(samp.columns[:14], fontsize=8)
ax.set_xticks([0, 59, 119]); ax.set_xticklabels(["T-120", "T-60", "T-0"])
ax.set_title("(c) 特征矩阵样本热力图（最近120交易日 × 前14列）")
ax.set_xlabel("交易日（相对最新样本）"); ax.set_ylabel("特征")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

ax = axes[1, 1]
sub = feats.iloc[:, :12]
corr = sub.corr()
im = ax.imshow(corr, cmap=CMAP_BIN, vmin=-1, vmax=1)
ax.set_xticks(range(12)); ax.set_xticklabels(sub.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(12)); ax.set_yticklabels(sub.columns, fontsize=8)
ax.set_title("(d) 特征相关性矩阵（前12列，资金利率类）")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
fig.suptitle("图1  原始数据样本概览：目标序列 / 分布 / 特征空间", fontsize=16, fontweight="bold", y=1.00)
fig.tight_layout(rect=[0, 0, 1, 0.97])
save("fig01_原始数据样本图", fig)

# ============================================================
# fig02 预处理过程可视化
# ============================================================
print("[图2] 预处理过程可视化 ...", flush=True)
fig, axes = plt.subplots(3, 1, figsize=(12.8, 8.6))

ax = axes[0]
seg = rate[(rate["date"] >= "2020-06-01") & (rate["date"] <= "2021-06-30")].reset_index(drop=True)
H = 10
seg["fwd"] = seg["rate"].shift(-H) - seg["rate"]
ax.plot(seg["date"], seg["rate"], color=C["blue"], lw=1.4, label="10年期国债收益率")
i0 = 60
ax.scatter([seg.loc[i0, "date"]], [seg.loc[i0, "rate"]], color=C["dark"], zorder=5, s=42)
ax.scatter([seg.loc[i0 + H, "date"]], [seg.loc[i0 + H, "rate"]], color=C["red"], zorder=5, s=42)
ax.annotate("", xy=(seg.loc[i0 + H, "date"], seg.loc[i0 + H, "rate"]),
            xytext=(seg.loc[i0, "date"], seg.loc[i0, "rate"]),
            arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.8))
d = seg.loc[i0 + H, "rate"] - seg.loc[i0, "rate"]
lbl = "未来10天变化 Δ=+%.0fbp\nΔ<0 → 标签=1(下行)" % (d * 100) if d < 0 else \
      "未来10天变化 Δ=+%dbp\nΔ≥0 → 标签=0(上行)" % int(d * 100)
ax.annotate(lbl, xy=(seg.loc[i0 + 3, "date"], seg.loc[i0, "rate"] + 0.10),
            fontsize=10.5, color=C["red"])
down = seg[seg["fwd"] < 0]
ax.scatter(down["date"], down["rate"], color=C["green"], s=6, alpha=0.5, zorder=3,
           label="未来10天下行样本（标签=1）")
ax.set_title("(a) 标签动态生成：未来10个交易日利率变化 Δ = rate(t+10) − rate(t)（shift(-10)，下行=1）")
ax.set_xlabel("日期"); ax.set_ylabel("收益率 (%)"); ax.legend(loc="upper right", fontsize=9.5)

ax = axes[1]
for i, r in folds.iterrows():
    y = r["fold"]
    ax.plot([r["date_start"], r["date_end"]], [y, y], lw=3.2,
            color=plt.cm.viridis(i / len(folds)), solid_capstyle="butt")
    # 训练窗示意（fold测试段向前1260交易日, 只画首尾若干fold避免过密）
    if i % 4 == 0 or i == len(folds) - 1:
        tr_start = r["date_start"] - pd.Timedelta(days=int(1260 * 365 / 252))
        ax.plot([tr_start, r["date_start"]], [y, y], lw=3.2, color="#BDD7E7",
                solid_capstyle="butt")
ax.set_yticks([1, 7, 14, 21, 28])
ax.set_title("(b) 28-fold 滚动验证切分（彩色=各fold样本外测试段；浅蓝=滚动训练窗≤1260交易日）")
ax.set_xlabel("日期"); ax.set_ylabel("fold 编号")
ax.set_xlim(folds["date_start"].min() - pd.Timedelta(days=2600), folds["date_end"].max())
ax.annotate("测试段随时间滚动前移\n训练窗与测试段之间按序列长度purge", xy=(0.02, 0.15),
            xycoords="axes fraction", fontsize=10, color=C["gray"])

ax = axes[2]
rr = rate.dropna(subset=["rate"]).copy()
rr["fwd"] = rr["rate"].shift(-H) - rr["rate"]
dz = rr["fwd"].dropna() * 100
bins = np.linspace(-40, 40, 41)
ax.hist(dz[(dz >= -40) & (dz <= 40)], bins=bins, color=C["blue"], alpha=0.75, edgecolor="white")
ax.axvspan(-2, 2, color=C["orange"], alpha=0.45, label="死区 |Δ|<2bp（权重0.3）")
ax.axvline(0, color=C["red"], ls="--", lw=1.3)
ax.annotate("Δ<0 → 标签1\n(权重1.0)", xy=(-26, ax.get_ylim()[1] * 0.86), fontsize=10.5, color=C["green"])
ax.annotate("Δ≥0 → 标签0\n(权重1.0)", xy=(12, ax.get_ylim()[1] * 0.86), fontsize=10.5, color=C["dark"])
ax.set_title("(c) 未来10天利率变化分布与死区样本权重（|Δ|<2bp 方向模糊，降权至0.3）")
ax.set_xlabel("未来10天利率变化 Δ (bp)"); ax.set_ylabel("频数"); ax.legend(loc="upper right")
fig.suptitle("图2  预处理流程：标签生成 / 滚动验证切分 / 死区加权", fontsize=16, fontweight="bold", y=1.00)
fig.tight_layout(rect=[0, 0, 1, 0.97])
save("fig02_预处理过程可视化", fig)

# ============================================================
# fig03 模型结构示意图
# ============================================================
print("[图3] 模型结构示意图 ...", flush=True)
fig, ax = plt.subplots(figsize=(15.0, 7.6))   # 模型结构横幅图，保证≥1080P
ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 46)
ax.grid(False)

blocks = [
    ("输入序列\n30特征 × 40交易日\n(XGB重要性Top-30)", C["gray"], 1.5),
    ("双向BiGRU\nhidden=64\n序列编码", C["blue"], 16),
    ("Inception模块\n4分支1D卷积 k=3/5/7/9\n+1×1残差 +BN +MaxPool", C["green"], 30.5),
    ("自注意力\nSelf-Attention\n(Q·K/√d)·V", C["orange"], 45),
    ("全局平均池化\nGAP\n(时间步聚合)", C["purple"], 59.5),
    ("全连接 FC\n64 → 2", C["blue"], 74),
    ("Sigmoid\n下行概率\nP(下行|历史)", C["red"], 88),
]
for txt, col, x in blocks:
    ax.add_patch(FancyBboxPatch((x, 22), 13, 13, boxstyle="round,pad=0.6",
                                fc=col + "33", ec=col, lw=2.0))
    ax.text(x + 6.5, 28.5, txt, ha="center", va="center", fontsize=10.2, color="#222222")
for x in [14.5, 29, 43.5, 58, 72.5, 87]:
    ax.add_patch(FancyArrowPatch((x, 28.5), (x + 1.6, 28.5),
                                 arrowstyle="-|>", mutation_scale=22, color="#555555", lw=2))
notes = [
    ("训练协议", "28-fold滚动验证 | 窗口内标准化(防泄漏) | purged验证切分 | 验证AUC早停"),
    ("正则与加权", "label smoothing 0.05 | 类别权重×死区权重(0.3) | AdamW+余弦退火 | 梯度裁剪1.0"),
    ("集成与决策", f"5-seed概率平均 | fold验证集自动选阈值 | 采用固定判决阈值 th={FIX_TH}（历史最优, 无泄漏）"),
]
for j, (t1, t2) in enumerate(notes):
    y = 15.5 - j * 6.4
    ax.add_patch(FancyBboxPatch((1.5, y - 2.4), 13, 5.2, boxstyle="round,pad=0.4",
                                fc="#F0F0F0", ec="#AAAAAA", lw=1.2))
    ax.text(8, y, t1, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#444444")
    ax.text(16.5, y, t2, ha="left", va="center", fontsize=10.2, color="#444444")
ax.set_title("图3  BiGRU + Inception + Self-Attention 模型结构与训练协议（10天方向二分类）",
             fontsize=15, fontweight="bold", pad=14)
save("fig03_模型结构示意图", fig)

# ============================================================
# fig04 时间序列对比折线图（预测 vs 真实）
# ============================================================
print("[图4] 时间序列对比 ...", flush=True)
show = pred[pred["date"] >= "2023-07-01"].copy()
rs = rate[rate["date"] >= "2023-07-01"]
fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.6), sharex=True,
                         gridspec_kw={"height_ratios": [1.15, 1]})
ax = axes[0]
ax.plot(rs["date"], rs["rate"], color=C["blue"], lw=1.5, label="真实观测：10年期国债收益率")
# 真实标签=1（未来10天下行）区间着色
lab = show.set_index("date")["y_true"]
grp = (lab != lab.shift()).cumsum()
for _, g in lab.groupby(grp):
    if g.iloc[0] == 1 and len(g) >= 2:
        ax.axvspan(g.index.min(), g.index.max(), color=C["green"], alpha=0.16)
ax.set_ylabel("收益率 (%)")
ax.set_title("(a) 真实观测：收益率走势（绿色底=未来10天实际下行区间，标签=1）")
ax.legend(loc="upper right")

ax = axes[1]
ax.plot(show["date"], show["proba_down"], color=C["red"], lw=1.1, alpha=0.9,
        label="模型预测：下行概率 P(下行)")
ax.axhline(FIX_TH, color=C["dark"], ls="--", lw=1.6,
           label=f"采用判决阈值 th={FIX_TH}（历史最优固定线）")
ax.axhline(0.50, color=C["gray"], ls=":", lw=1.1, label="默认阈值 0.50")
ax.plot(show["date"], show["th_calib"], color=C["orange"], lw=1.0, alpha=0.75,
        label="滚动最优阈值路径（诊断：双峰摆动，校准无效）")
ok = show[show["pred_fix"] == show["y_true"]]
bad = show[show["pred_fix"] != show["y_true"]]
ax.scatter(ok["date"], ok["proba_down"], s=9, color=C["green"], zorder=4,
           label=f"预测正确 ({len(ok)}例, acc={len(ok)/len(show):.1%})")
ax.scatter(bad["date"], bad["proba_down"], s=14, color=C["red"], marker="x", zorder=5,
           label=f"预测错误 ({len(bad)}例)")
ax.set_ylim(-0.04, 1.04); ax.set_ylabel("下行概率")
ax.set_xlabel("日期")
ax.set_title(f"(b) 模型预测：下行概率与判决阈值 th={FIX_TH}（10天版 5-seed 集成，样本外）")
ax.legend(loc="upper right", ncols=2, fontsize=9)
fig.suptitle("图4  预测 vs 真实：时间序列对比（2023-07 至 2026）", fontsize=16,
             fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.965])
save("fig04_预测对比_时间序列", fig)

# ============================================================
# fig05 散点拟合分布图 + R²/MSE 标注
# ============================================================
print("[图5] 散点拟合分布 ...", flush=True)
rng = np.random.default_rng(42)
p_all, y_all = pred["proba_down"].to_numpy(), pred["y_true"].to_numpy()
brier = float(np.mean((p_all - y_all) ** 2))                      # MSE（二分类=Brier分数）
ref = float(np.mean((y_all.mean() - y_all) ** 2))                 # 基准: 样本正类率常数预测
bss = 1.0 - brier / ref                                           # 决定系数 R²（Brier技巧分）
auc = roc_auc_score(y_all, p_all)
ll = log_loss(y_all, p_all, labels=[0, 1])

fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.0))
ax = axes[0]
jy = y_all + rng.uniform(-0.055, 0.055, len(y_all))
ax.scatter(p_all[y_all == 0], jy[y_all == 0], s=13, color=C["blue"], alpha=0.45,
           label="真实标签=0（未来10天上行）")
ax.scatter(p_all[y_all == 1], jy[y_all == 1], s=13, color=C["red"], alpha=0.45,
           label="真实标签=1（未来10天下行）")
ax.plot([0, 1], [0, 1], color=C["gray"], ls="--", lw=1.2)
ax.set_xlim(0, 1); ax.set_ylim(-0.12, 1.12)                       # 统一坐标轴范围 [0,1]
ax.set_xticks(np.arange(0, 1.01, 0.1)); ax.set_yticks([0, 1])
ax.set_xlabel("模型预测下行概率"); ax.set_ylabel("真实标签（含抖动便于观察）")
ax.set_title("(a) 预测概率 vs 真实标签 散点分布（坐标统一[0,1]）")
box = (f"评估指标（样本外, n={len(y_all)}）\n"
       f"MSE (Brier) = {brier:.4f}\n"
       f"决定系数 R² (BSS) = {bss:.4f}\n"
       f"LogLoss = {ll:.4f}\n"
       f"AUC = {auc:.4f}")
ax.text(0.025, 0.965, box, transform=ax.transAxes, va="top", ha="left", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=C["blue"], alpha=0.92))
ax.legend(loc="lower right", fontsize=9.5)

ax = axes[1]
nb, bins_ = 10, np.linspace(0, 1, 11)
bid = np.digitize(p_all, bins_) - 1
xs, ys = [], []
for b in range(nb):
    msk = bid == b
    if msk.sum() >= 15:
        xs.append(p_all[msk].mean()); ys.append(y_all[msk].mean())
ax.plot([0, 1], [0, 1], color=C["gray"], ls="--", lw=1.2, label="理想校准（概率=频率）")
ax.plot(xs, ys, "o-", color=C["red"], lw=2, ms=8, label="模型分箱校准曲线")
ax.fill_between([0, 1], [0, 1], [0.5, 0.5], color=C["orange"], alpha=0.10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xticks(np.arange(0, 1.01, 0.1)); ax.set_yticks(np.arange(0, 1.01, 0.1))
ax.set_xlabel("预测下行概率（分箱均值）"); ax.set_ylabel("实际上行→下行经验频率")
ax.set_title("(b) 分箱校准曲线：概率质量 vs 实际发生频率")
ax.annotate("中低概率区低估下行风险\n(系统性高估下行概率, th=0.39下调线补偿)",
            xy=(0.35, 0.35), fontsize=10, color="#8B4513")
ax.legend(loc="upper left", fontsize=10)
fig.suptitle("图5  预测 vs 真实：散点拟合与概率校准（10天版 5-seed）", fontsize=16,
             fontweight="bold", y=1.00)
fig.tight_layout(rect=[0, 0, 1, 0.96])
save("fig05_散点拟合分布", fig)

# ============================================================
# fig06 误差热力图（混淆矩阵 + 年份×方向准确率）
# ============================================================
print("[图6] 误差热力图 ...", flush=True)
pc = pred[pred["pred_fix"] >= 0]
cm = np.zeros((2, 2), dtype=int)
for t in (0, 1):
    for pr in (0, 1):
        cm[t, pr] = int(((pc["y_true"] == t) & (pc["pred_fix"] == pr)).sum())
fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.2))
ax = axes[0]
im = ax.imshow(cm, cmap=CMAP_BLUE)
for t in (0, 1):
    for pr in (0, 1):
        v = cm[t, pr]
        ax.text(pr, t, f"{v}\n({v/cm.sum():.1%})", ha="center", va="center",
                fontsize=14, color="#222222" if v > cm.max() * 0.4 else "white",
                fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["预测=上行(0)", "预测=下行(1)"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["真实=上行(0)", "真实=下行(1)"])
ax.set_title(f"(a) 混淆矩阵（固定阈值 th={FIX_TH}, 全局 Acc={calib_acc:.4f}）")
ax.set_xlabel("模型预测"); ax.set_ylabel("真实观测")
fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
ax.annotate("错误单元", xy=(1, 0), xytext=(0.62, -0.24), fontsize=11, color=C["red"],
            arrowprops=dict(arrowstyle="->", color=C["red"]))
ax.annotate("", xy=(0, 1), xytext=(0.38, -0.24))

ax = axes[1]
years = sorted(pc["year"].unique())
rows = []
for y in years:
    g = pc[pc["year"] == y]
    rows.append([accuracy_score(g[g["pred_fix"] == 1]["y_true"], g[g["pred_fix"] == 1]["pred_fix"]) if (g["pred_fix"] == 1).sum() else np.nan,
                 accuracy_score(g[g["pred_fix"] == 0]["y_true"], g[g["pred_fix"] == 0]["pred_fix"]) if (g["pred_fix"] == 0).sum() else np.nan,
                 accuracy_score(g["y_true"], g["pred_fix"])])
mat = np.array(rows, dtype=float)
im = ax.imshow(mat, cmap=CMAP_BIN, vmin=0.4, vmax=0.95, aspect="auto")
for i in range(mat.shape[0]):
    for j in range(3):
        if not np.isnan(mat[i, j]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=11,
                    color="#222222" if mat[i, j] < 0.8 else "white", fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(["预测下行时\n准确率", "预测上行时\n准确率", "全年\n准确率"])
ax.set_yticks(range(len(years))); ax.set_yticklabels(years)
ax.set_title("(b) 分年 × 预测方向 准确率热力图")
ax.set_xlabel("预测方向 / 汇总"); ax.set_ylabel("年份")
fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
fig.suptitle(f"图6  误差热力图：混淆矩阵与分年方向准确率（固定阈值 th={FIX_TH}）",
             fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.95])
save("fig06_误差热力图", fig)

# ============================================================
# fig07 评估指标雷达图
# ============================================================
print("[图7] 指标雷达图 ...", flush=True)
labels = ["全局Acc", "全局MacroF1", "全局AUC", "2025年Acc", "2025年AUC"]
series = {
    "XGB基线(5天)":            [0.5635, 0.5627, 0.5822, 0.4444, 0.4809],
    "5天版最优 SEQ80_ACC3":    [0.5720, 0.5715, 0.5875, 0.6008, 0.6137],
    "10天版 FINAL5(5seed)":    [0.5913, 0.5762, 0.6705, 0.5144, 0.6826],
    "FINAL5+固定阈值0.39":      [calib_acc, calib_f1, 0.6705, calib_2025, 0.6826],
}
colors = [C["gray"], C["blue"], C["orange"], C["red"]]
ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
ang += ang[:1]
fig, ax = plt.subplots(figsize=(12.6, 9.4), subplot_kw=dict(polar=True))
for (name, vals), col in zip(series.items(), colors):
    v = vals + vals[:1]
    ax.plot(ang, v, "o-", lw=2.2, ms=5.5, color=col, label=name)
    ax.fill(ang, v, color=col, alpha=0.07)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=12)
ax.set_ylim(0.40, 0.85); ax.set_yticks([0.45, 0.55, 0.65, 0.75, 0.85])
ax.set_yticklabels(["0.45", "0.55", "0.65", "0.75", "0.85"], fontsize=9.5, color="#666666")
ax.grid(color="#CCCCCC", alpha=0.8)
ax.set_title("图7  模型评估指标雷达图（五维对比，28-fold样本外）",
             fontsize=15, fontweight="bold", pad=24)
ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.10), fontsize=10.5)
save("fig07_指标雷达图", fig)

# ============================================================
# fig08 参数性能对比柱状图（数字取自 out/*_nn_metrics.txt，注释标来源）
# ============================================================
print("[图8] 参数对比柱状图 ...", flush=True)
grp = {  # 名称: (Acc, MacroF1, AUC)
    "XGB基线\n(5天)":                    (0.5635, 0.5627, 0.5822),   # xgb基线
    "5天\nSEQ80_ACC3":                   (0.5720, 0.5715, 0.5875),   # SEQ80_ACC3_nn_metrics
    "10天\nH10_SEQ40":                   (0.6338, 0.6224, 0.6534),   # H10_SEQ40_nn_metrics
    "10天\nFINAL5(5seed)":               (0.5913, 0.5762, 0.6705),   # H10_S40_FINAL5_nn_metrics
}
seq = {  # seq_len 扫描（10天版）
    "DRY\nseq60·1seed": (0.5556, 0.4323, 0.5581),
    "ACC2\nseq60":      (0.5992, 0.5625, 0.6266),
    "SEQ80\nseq80":     (0.5626, 0.5400, 0.5447),
    "SEQ40\nseq40":     (0.6338, 0.6224, 0.6534),
    "FINAL5\nseq40·5s": (0.5913, 0.5762, 0.6705),
}
win = {"120": 0.5006, "180": 0.5249, "270": 0.5295, "1260": 0.5578,
       "1512": 0.5697, "1800": 0.5686, "EXP": 0.5368}              # WIN*_ACC2_nn_metrics

fig, axes = plt.subplots(1, 3, figsize=(15.0, 7.4))
def grouped_bar(ax, data: dict, title, note):
    names = list(data); n = len(names); w = 0.26
    xs = np.arange(n)
    for k, (key, col) in enumerate(zip(["Acc", "MacroF1", "AUC"],
                                       [C["blue"], C["orange"], C["green"]])):
        vals = [data[nm][k] for nm in names]
        b = ax.bar(xs + (k - 1) * w, vals, w, color=col, label=key, alpha=0.92)
        ax.bar_label(b, fmt="%.3f", fontsize=7.6, padding=1.5, rotation=90)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0.40, 0.72); ax.axhline(0.5635, color=C["gray"], ls="--", lw=1.1)
    ax.text(len(names) - 0.45, 0.5660, "XGB基线Acc", fontsize=8.5, color=C["gray"])
    ax.set_title(title, fontsize=12.5); ax.legend(fontsize=9.5, ncols=3, loc="lower left")
    ax.set_xlabel(note, fontsize=10)

grouped_bar(axes[0], grp, "(a) 预测周期对比：5天 → 10天", "配置（ horizon / 模型 ）")
grouped_bar(axes[1], seq, "(b) 输入序列长度扫描（10天版）", "seq_len（交易日）")
ax = axes[2]
ks = list(win); vals = list(win.values())
cols = [C["red"] if k == "1512" else C["blue"] for k in ks]
b = ax.bar(ks, vals, color=cols, alpha=0.9)
ax.bar_label(b, fmt="%.3f", fontsize=9, padding=2)
ax.axhline(0.5635, color=C["gray"], ls="--", lw=1.1)
ax.set_ylim(0.45, 0.62); ax.set_title("(c) 训练窗口扫描（5天版, 2seed）", fontsize=12.5)
ax.set_xlabel("训练窗口（交易日, EXP=全量扩展）"); ax.set_ylabel("全局 Acc")
ax.annotate("最优 1512", xy=(4, 0.5697), xytext=(2.6, 0.595), fontsize=10, color=C["red"],
            arrowprops=dict(arrowstyle="->", color=C["red"]))
fig.suptitle("图8  不同参数下的性能对比（28-fold样本外全局指标）",
             fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.94])
save("fig08_参数性能对比柱状图", fig)

# ============================================================
# fig09 误差分布箱线图
# ============================================================
print("[图9] 误差分布箱线图 ...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.0))
ax = axes[0]
cfg_files = {  # 各配置逐fold准确率分布（读fold明细csv）
    "XGB基线(5天)": DATA / "xgb_fold_metrics_bin.csv",
    "5天 FINAL7": OUT / "FINAL7_nn_fold_metrics.csv",
    "5天 SEQ80_ACC3": OUT / "SEQ80_ACC3_nn_fold_metrics.csv",
    "10天 SEQ40": OUT / "H10_SEQ40_nn_fold_metrics.csv",
    "10天 FINAL5": OUT / "H10_S40_FINAL5_nn_fold_metrics.csv",
}
data_box, names_box = [], []
for nm, f in cfg_files.items():
    d = pd.read_csv(f)
    acc_col = "accuracy" if "accuracy" in d.columns else "acc"
    data_box.append(d[acc_col].dropna().to_numpy()); names_box.append(nm)
bp = ax.boxplot(data_box, patch_artist=True, widths=0.55, medianprops=dict(color=C["dark"], lw=2))
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor([C["gray"], C["blue"], C["blue"], C["orange"], C["red"]][i] + "66")
    patch.set_edgecolor("#555555")
for i, d in enumerate(data_box, 1):
    ax.scatter(rng.uniform(i - 0.13, i + 0.13, len(d)), d, s=11, alpha=0.55,
               color=[C["gray"], C["blue"], C["blue"], C["orange"], C["red"]][i - 1], zorder=3)
ax.axhline(0.5, color=C["gray"], ls=":", lw=1.2)
ax.set_xticklabels(names_box, fontsize=9.5)
ax.set_ylabel("逐 fold 样本外 Acc")
ax.set_title("(a) 各配置逐fold准确率分布（箱=P25~P75, 线=中位数, 点=各fold）")

ax = axes[1]
pred["err"] = (pred["proba_down"] - pred["y_true"]).abs()
years9 = [y for y in sorted(pred["year"].unique()) if y != 2026]
data9 = [pred.loc[pred["year"] == y, "err"].to_numpy() for y in years9]
bp = ax.boxplot(data9, patch_artist=True, widths=0.55,
                medianprops=dict(color=C["dark"], lw=2))
for patch in bp["boxes"]:
    patch.set_facecolor(C["blue"] + "55"); patch.set_edgecolor("#555555")
med = [np.median(d) for d in data9]
ax.plot(range(1, len(years9) + 1), med, "o-", color=C["red"], lw=1.8, ms=6,
        label="中位误差趋势")
ax.set_xticklabels(years9)
ax.set_ylabel("|预测概率 − 真实标签|（Brier单样本误差）")
ax.set_title("(b) 10天版FINAL5 概率误差分年分布")
ax.legend(loc="upper right", fontsize=10)
fig.suptitle("图9  误差分布箱线图：fold级稳定性与分年概率误差",
             fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout(rect=[0, 0, 1, 0.95])
save("fig09_误差分布箱线图", fig)

print(f"\n全部完成 → {FIGS}", flush=True)
