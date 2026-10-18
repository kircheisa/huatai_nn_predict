# -*- coding: utf-8 -*-
"""
13_完整汇报PPT生成.py
======================================================================
18页全景汇报：特征筛选 → XGB → 神经网络 → 训练优化 → 融合冠军方案
→ 效果归因 → 不足 → 后续规划。全部数据来自已有实验产物，不重训练。
运行: & "D:\\soft\\Anaconda3\\python.exe" -u -B "13_完整汇报PPT生成.py"
"""
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
DEST = ROOT / "训练效果总结"
PPT = DEST / "国债收益率方向预测_特征模型优化全景汇报.pptx"

FONT = "Microsoft YaHei"
NAVY, TEAL, GOLD, RED = "10243A", "087F8C", "C38B22", "C9474D"
INK, MUTED, LIGHT, WHITE = "26374A", "64758B", "F3F6FA", "FFFFFF"
plt.rcParams.update({
    "font.sans-serif": [FONT], "axes.unicode_minus": False,
    "font.size": 12, "axes.labelcolor": "#" + INK,
    "text.color": "#" + INK, "xtick.color": "#" + MUTED,
    "ytick.color": "#" + MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#D5DDE7",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
C = {"navy": "#" + NAVY, "teal": "#" + TEAL, "gold": "#" + GOLD,
     "red": "#" + RED, "muted": "#" + MUTED, "ink": "#" + INK}


# ================= pptx 基础 =================
def fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(color)
    if hasattr(shape, "line"):
        shape.line.fill.background()


def box(slide, x, y, w, h, color=WHITE, rounded=True):
    s = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    fill(s, color)
    if rounded:
        s.adjustments[0] = .08
    return s


def text(slide, value, x, y, w, h, size=18, color=INK, bold=False,
         align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(.02)
    tf.margin_top = tf.margin_bottom = Inches(.015)
    for i, line in enumerate(str(value).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = line
        r.font.name, r.font.size, r.font.bold = FONT, Pt(size), bold
        r.font.color.rgb = RGBColor.from_string(color)
        ea = OxmlElement("a:ea")
        ea.set("typeface", FONT)
        r._r.get_or_add_rPr().append(ea)
    return shape


class Deck:
    def __init__(self, total=18):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.333), Inches(7.5)
        self.total = total

    def slide(self, title, subtitle, source=""):
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        fill(slide.background, LIGHT)
        box(slide, 0, 0, 13.333, .12, TEAL, False)
        text(slide, "国债收益率方向预测 / 全景汇报", .55, .27, 10, .3, 10, MUTED)
        text(slide, title, .55, .66, 12.1, .62, 28, NAVY, True)
        text(slide, subtitle, .57, 1.36, 12.0, .48, 12, MUTED)
        if source:
            text(slide, source, .57, 7.08, 11.7, .24, 8.5, MUTED)
        n = len(self.prs.slides)
        text(slide, f"{n:02d} / {self.total}", 12.1, 7.04, .65, .3,
             10, MUTED, align=PP_ALIGN.RIGHT)
        return slide

    def chart(self, slide, fig, x, y, w, h=None):
        image = BytesIO()
        fig.savefig(image, format="png", dpi=190, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        image.seek(0)
        pic = slide.shapes.add_picture(image, Inches(x), Inches(y), width=Inches(w))
        if h is not None and abs(pic.height / 914400 - h) > .05:
            pic.height = Inches(h)
            pic.width = Inches(w)
        return pic


def card(slide, x, y, w, h, label, value, detail, accent=TEAL, vsize=30):
    box(slide, x, y, w, h)
    box(slide, x, y, .055, h, accent, False)
    text(slide, label, x + .2, y + .16, w - .4, .36, 13.5, MUTED)
    text(slide, value, x + .2, y + .62, w - .4, .62, vsize, accent, True)
    text(slide, detail, x + .2, y + 1.34, w - .4, h - 1.42, 13)


def bullet_panel(slide, x, y, w, h, heading, items, accent=TEAL, size=15):
    box(slide, x, y, w, h)
    compact = h < 2.4
    start = .52 if compact else .78
    text(slide, heading, x + .22, y + .1, w - .44, .36 if compact else .5,
         16 if compact else 18, accent, True)
    step = (h - start - .08) / max(len(items), 1)
    for i, item in enumerate(items):
        box(slide, x + .23, y + start + .1 + i * step, .055, .055, accent, False)
        text(slide, item, x + .43, y + start + i * step, w - .67, step - .02, size)


def table(slide, columns, rows, x, y, widths, row_h=.42, size=13.5,
          highlights=(), header_size=None):
    total = sum(widths)
    offset = .09 if row_h >= .3 else .04
    text_h = row_h - .1 if row_h >= .3 else row_h - .055
    box(slide, x, y, total, row_h, NAVY, False)
    for j, (name, width) in enumerate(zip(columns, widths)):
        text(slide, name, x + sum(widths[:j]) + .12, y + offset,
             width - .18, text_h, header_size or size, WHITE, True)
    for i, row in enumerate(rows):
        color = "E6F3F1" if i in highlights else (WHITE if i % 2 == 0 else "EAF0F6")
        yy = y + (i + 1) * row_h
        box(slide, x, yy, total, row_h, color, False)
        for j, (value, width) in enumerate(zip(row, widths)):
            text(slide, value, x + sum(widths[:j]) + .12, yy + offset,
                 width - .18, text_h, size, INK, i in highlights)


# ================= 数据加载 =================
def load_all():
    d = {}
    d["rate"] = pd.read_csv(ROOT / "data_ori" / "2026_origin_rate.csv").iloc[:, :2]
    r = d["rate"]
    r.columns = ["date", "rate"]
    r["date"] = pd.to_datetime(r["date"])
    r = r.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
    r["delta"] = r["rate"].shift(-10) - r["rate"]
    d["rate"] = r

    d["fuse"] = pd.read_csv(OUT / "OPTB_融合对比.csv")
    d["wcrit"] = pd.read_csv(OUT / "OPTB_权重准则.csv")
    d["final"] = pd.read_csv(OUT / "OPTB_final_predictions.csv", parse_dates=["date"])
    d["opt9"] = pd.read_csv(OUT / "OPT9_对比汇总.csv")
    d["amp"] = pd.read_csv(OUT / "DIAG_幅度分层.csv")
    d["vol"] = pd.read_csv(OUT / "DIAG_波动分层.csv")
    d["yearly"] = pd.read_csv(OUT / "DIAG_分年指标.csv")
    d["fi_xgb"] = pd.read_csv(ROOT / "model_new" / "data" / "xgb_feature_importance_bin.csv")
    d["stab"] = pd.read_csv(ROOT / "data" / "43_xgb_feature_stability_bin.csv")
    d["x10"] = pd.read_csv(ROOT / "model_new" / "data" / "xgb_test_predictions_10d.csv",
                           parse_dates=["date"])
    return d


# ================= 图表 =================
def fig_rate(D):
    r = D["rate"]
    fig, ax = plt.subplots(figsize=(11.6, 2.9))
    ax.plot(r["date"], r["rate"], lw=1.1, color=C["navy"])
    ax.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"),
               color=C["gold"], alpha=.18)
    ax.text(pd.Timestamp("2025-06-15"), r["rate"].max() * .96,
            "2025 验证年", color=C["gold"] if False else "#8a6414", fontsize=11,
            ha="center", fontweight="bold")
    ax.set_ylabel("10年国债收益率 (%)")
    ax.set_title("10年期国债收益率（2010–2026.1）与2025验证区间", loc="left",
                 fontsize=13, fontweight="bold")
    return fig


def fig_feat_evol(D):
    stages = ["XGB缓存\n(工程特征)", "初筛后", "因果D\n(基础对齐)", "因果E\n(+动量/回撤)",
              "因果F\n(+资金/机构)", "因果G\n(+慢漂移)"]
    vals = [280, 231, 100, 120, 144, 136]
    colors = [C["muted"], C["muted"], C["teal"], C["teal"], C["teal"], "#b0bec5"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    b = ax.bar(stages, vals, color=colors, width=.62)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 6, str(v),
                ha="center", fontsize=11, fontweight="bold", color=C["ink"])
    ax.set_ylim(0, 320)
    ax.set_title("特征体系演进：从缓存工程特征到因果特征", loc="left",
                 fontsize=12.5, fontweight="bold")
    ax.set_ylabel("特征数")
    return fig


def fig_feat_stab(D):
    fi = D["fi_xgb"].sort_values("importance", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh(fi["feature"], fi["importance"], color=C["teal"], height=.62)
    ax.set_title("XGB 最终模型 Top12 特征重要性", loc="left",
                 fontsize=12.5, fontweight="bold")
    ax.set_xlabel("重要性")
    return fig


def fig_xgb_years(D):
    y = D["yearly"].copy()
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.bar(y["年份"].astype(str), y["majority"], color="#cfd8e3", label="多数类基线", width=.6)
    ax.bar(y["年份"].astype(str), y["acc"], color=C["teal"], label="模型Acc", width=.6)
    ax.axhline(.5, color=C["muted"], lw=.8, ls=":")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title("E40基线模型 分年Acc vs 多数类基线", loc="left",
                 fontsize=12.5, fontweight="bold")
    return fig


def fig_amp(D):
    a = D["amp"]
    a25 = a[a["区间"] == "2025"].copy()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(a25))
    ax.bar(x - .2, a25["acc"], .38, label="Acc", color=C["teal"])
    ax2 = ax.twinx()
    ax2.bar(x + .2, a25["auc"], .38, label="AUC", color=C["gold"])
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(a25["幅度"])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10)
    ax.set_title("2025年按未来10日实际涨跌幅分层的判别力", loc="left",
                 fontsize=12.5, fontweight="bold")
    ax.set_xlabel("涨跌幅区间 (bp)")
    return fig


def fig_monthly(D):
    f = D["final"].copy()
    f["month"] = f["date"].dt.month
    m25 = f[f["date"].dt.year == 2025]
    g = m25.groupby("month").agg(
        champ=("p_CHAMPION_devw_t001", lambda s: np.nan),
        n=("date", "size"))
    y = m25["y_dir"].to_numpy()
    months = sorted(m25["month"].unique())
    ok_c = ((m25["p_CHAMPION_devw_t001"] >= .5).astype(int) == y).groupby(m25["month"]).mean()
    ok_e = ((m25["p_e40"] >= .5).astype(int) == y).groupby(m25["month"]).mean() \
        if "p_e40" in m25 else None
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = np.arange(len(months))
    ax.bar(x - .21, ok_e.values if ok_e is not None else ok_c.values * np.nan, .4,
           label="现产E40基线", color="#b6c4d4")
    ax.bar(x + .21, ok_c.values, .4, label="冠军融合", color=C["teal"])
    ax.axhline(1 - y.mean(), color=C["red"], ls="--", lw=1,
               label=f"2025多数类基线 {1 - y.mean():.0%}")
    ax.set_xticks(x); ax.set_xticklabels([f"{m}月" for m in months], fontsize=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9.5, ncol=3)
    ax.set_title("2025年逐月方向准确率：冠军融合 vs 现产E40", loc="left",
                 fontsize=12.5, fontweight="bold")
    return fig


def fig_wcrit(D):
    w = D["wcrit"]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.plot(w["w_nn"], w["dev_auc"], "-o", ms=3.5, color=C["navy"],
            label="开发期AUC (选择准则)")
    ax.plot(w["w_nn"], w["dev2324_auc"], "-s", ms=3.5, color=C["gold"],
            label="开发期2023-24 AUC")
    ax.plot(w["w_nn"], w["2025_auc"], "-^", ms=3.5, color=C["red"],
            label="2025 AUC (仅报告)")
    ax.axvline(.4, color=C["teal"], ls=":", lw=1.2)
    ax.annotate("dev准确率准则峰值 w=0.4", (.42, .40), fontsize=9.5, color=C["teal"])
    ax.set_xlabel("NN权重 w (0=纯XGB10D, 1=纯featF)")
    ax.set_ylabel("AUC (native)")
    ax.legend(loc="center left", fontsize=9.5)
    ax.set_title("融合权重敏感性：开发期准则与2025的错位", loc="left",
                 fontsize=12.5, fontweight="bold")
    return fig


def fig_model_auc(D):
    names = ["XGB-5d\n(二分类)", "XGB-H10\n缓存", "NN E40\n(现产)", "NN featF\n(最优单模)",
             "冠军融合\n(dev选权)"]
    auc_dir = [0.535, 0.634, 0.595, 0.645, 0.641]
    auc_nat = [np.nan, 0.667, 0.717, 0.758, 0.721]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    ax.bar(x - .2, auc_dir, .38, label="2025方向AUC", color=C["navy"])
    ax.bar(x + .2, [0 if np.isnan(v) else v for v in auc_nat], .38,
           label="2025原生AUC", color=C["teal"])
    for i, v in enumerate(auc_dir):
        ax.text(i - .2, v + .012, f"{v:.3f}", ha="center", fontsize=9.5)
    for i, v in enumerate(auc_nat):
        if not np.isnan(v):
            ax.text(i + .2, v + .012, f"{v:.3f}", ha="center", fontsize=9.5)
    ax.axhline(.5, color=C["muted"], lw=.8, ls=":")
    ax.text(4.45, .505, "随机=0.5", fontsize=9, color=C["muted"])
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylim(.4, .85)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("2025年各代模型 AUC 对比", loc="left", fontsize=12.5, fontweight="bold")
    return fig


def fig_complement(D):
    f = D["final"]
    m = f[f["date"].dt.year == 2025]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    axes[0].hist(m["p_single_XGB10D"], bins=24, color="#9fb3c8",
                 label="XGB10D", range=(0, 1))
    axes[0].axvline(.5, color=C["red"], lw=1, ls="--")
    axes[0].set_title("XGB10D 概率分布（长期偏空）", loc="left", fontsize=11.5)
    axes[1].hist(m["p_single_featF"], bins=24, color=C["teal"], label="featF NN")
    axes[1].axvline(.5, color=C["red"], lw=1, ls="--")
    axes[1].set_title("featF NN 概率分布（标度均衡）", loc="left", fontsize=11.5)
    for a in axes:
        a.set_xlabel("下行概率"); a.set_ylabel("天数")
    fig.suptitle("互补性：XGB排序强但概率失准，NN标度准但排序弱", x=.02,
                 ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .93])
    return fig


def fig_evo(D):
    stages = ["三分类XGB\n2024", "二分类XGB-5d\n2025.06", "XGB-H10缓存", "NN E40\n2025.09",
              "NN featF\n2025.10", "冠军融合\n2025.10"]
    auc = [0.385, 0.535, 0.634, 0.595, 0.758, 0.774]
    dev = [0.385, 0.535, 0.622, 0.533, 0.564, 0.640]
    fig, ax = plt.subplots(figsize=(11.6, 3.1))
    ax.plot(range(len(stages)), auc, "-o", ms=6, color=C["teal"], lw=2,
            label="2025原生AUC（或当年最好口径）")
    ax.plot(range(len(stages)), dev, "-s", ms=5, color=C["navy"], lw=1.6,
            label="开发期AUC（诚实对照）")
    for i, v in enumerate(auc):
        ax.text(i, v + .018, f"{v:.3f}", ha="center", fontsize=10,
                color=C["teal"], fontweight="bold")
    for i, v in enumerate(dev):
        ax.text(i, v - .038, f"{v:.3f}", ha="center", fontsize=9.5, color=C["navy"])
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, fontsize=10)
    ax.set_xlim(-.35, len(stages) - .65)
    ax.set_ylim(.3, .88)
    ax.axhline(.5, color=C["muted"], lw=.8, ls=":")
    ax.legend(loc="upper left", fontsize=10.5)
    ax.set_title("模型能力演进全景：从 Macro-F1 0.39 到 2025 原生AUC 0.77（对照0.72）",
                 loc="left", fontsize=13, fontweight="bold")
    return fig


# ================= 主流程 =================
def main():
    D = load_all()
    deck = Deck(total=18)
    src_opt9 = "来源：model_nn/out/OPT9_对比汇总.csv、OPT9_validation.json"
    src_optb = "来源：model_nn/out/OPTB_融合对比.csv、OPTB_final_predictions.csv、OPTB_权重准则.csv"

    # ---- 01 封面 ----
    s = deck.slide("国债收益率方向预测：特征、模型与优化全景",
                   "10年期国债收益率 · 未来10日方向 · 从特征筛选到XGB与神经网络，再到融合冠军方案",
                   "数据：data_ori/2026_origin_rate.csv（2010–2026.1）；全部指标为样本外(OOS)滚动折结果")
    box(s, .65, 2.0, 6.0, 1.5)
    text(s, "任务", .9, 2.14, 5.5, .4, 15, MUTED)
    text(s, "预测未来10日利率方向（下行/上行）", .9, 2.55, 5.6, .5, 21, NAVY, True)
    text(s, "10日持有期 · 二分类 · 28折滚动验证 · 2018-2024开发期 / 2025验证年",
         .9, 3.05, 5.6, .4, 13)
    box(s, 6.85, 2.0, 5.8, 1.5)
    text(s, "当前最优（开发期选权）", 7.1, 2.14, 5.3, .4, 15, MUTED)
    text(s, "冠军融合  2025方向AUC 0.641", 7.1, 2.55, 5.4, .5, 21, TEAL, True)
    text(s, "方向Acc 62.1%（基线60.1%）· 原生AUC 0.721 · 开发期Acc 62.8%全场第一",
         7.1, 3.05, 5.4, .4, 13)
    deck.chart(s, fig_rate(D), .65, 3.8, 11.5)
    text(s, "汇报脉络：特征筛选 → XGB时代 → 神经网络 → 训练优化 → 融合冠军 → 归因 → 不足 → 规划",
         .65, 6.68, 12.0, .35, 13, MUTED)

    # ---- 02 目录 ----
    s = deck.slide("汇报框架", "八个部分，覆盖从数据到部署前的完整证据链")
    items = [
        ("01 任务与数据", "利率序列、标签定义、滚动折与开发期/2025划分"),
        ("02 特征工程与筛选", "280缓存特征 → 49个无效 → 因果特征体系D/E/F/G"),
        ("03 XGBoost时代", "三分类瓶颈、二分类改造、窗口与超参优化、H10缓存版"),
        ("04 神经网络架构与训练", "BiGRU+Inception+Attention、训练协议、多seed"),
        ("05 优化实验全景", "09实验矩阵：标签重定义/两阶段/特征/窗口/加权"),
        ("06 2025诊断与融合突破", "误差结构定位 → 同质融合失败 → 异质融合冠军方案"),
        ("07 为什么效果好 / 不足", "排序×标度互补归因；三类风险与未解决问题"),
        ("08 后续规划", "文本/事件特征、2026盲测、概率校准、影子运行"),
    ]
    for i, (t, d) in enumerate(items):
        col, row = i % 2, i // 2
        x = .65 + col * 6.35
        y = 1.85 + row * 1.24
        box(s, x, y, 6.1, 1.08)
        box(s, x, y, .055, 1.08, TEAL, False)
        text(s, t, x + .22, y + .12, 5.7, .4, 16.5, NAVY, True)
        text(s, d, x + .22, y + .56, 5.7, .45, 12.5, MUTED)

    # ---- 03 任务与数据 ----
    s = deck.slide("任务定义与评估口径", "所有选择只在开发期做，2025仅作验证报告——这是全部结论可信的前提")
    card(s, .65, 1.8, 3.0, 1.75, "预测对象", "10日方向", "10年期国债收益率\n未来10个交易日涨跌\n（H10口径）")
    card(s, 3.8, 1.8, 3.0, 1.75, "标签体系", "双口径", "原始方向 Δ<0\n原生 down_ex2 Δ<-2bp\n死区2bp并入非下行", GOLD)
    card(s, 6.95, 1.8, 3.0, 1.75, "验证框架", "28折滚动", "2018-09起每季度一折\n开发期1509样本\n2025年243样本")
    card(s, 10.1, 1.8, 2.55, 1.75, "防泄漏", "Purge+成熟", "训练标签必须先于\n测试首日成熟\ngap=max(40,10)", RED)
    rows = [
        ["原始方向 Δ<0", "39.9% / 60.1%", "60.1%", "方向观点是否有信息"],
        ["原生 down_ex2 Δ<-2bp", "24.3% / 75.7%", "75.7%", "模型实际训练任务；交易相关口径"],
        ["AUC（两口径均报）", "—", "0.5=随机", "排序能力，不受预测占比影响，主选型指标"],
    ]
    table(s, ["标签口径", "2025下行/上行占比", "多数类基线", "含义"], rows,
          .65, 3.95, [2.6, 2.2, 1.7, 5.4], row_h=.52, size=13,
          highlights=(2,))
    bullet_panel(s, .65, 5.75, 12.0, 1.1, "关键设计", [
        "开发期(≤2024)选一切超参/特征/权重；2025年243天从不参与任何选择，仅输出报告数字",
        "2025年两种标签的多数类方向相反（方向上行60% vs 原生非深跌76%）——单看Acc极易误判，必须配AUC",
    ], size=13.5)

    # ---- 04 特征演进 ----
    s = deck.slide("特征工程：从缓存工程特征到可追溯因果特征",
                   "旧体系量大但含未审计特征；新体系小而因果可辩护，为上线做准备",
                   "来源：data/43_xgb_feature_stability_bin.csv、model_new/data/xgb_feature_importance_10d.csv")
    deck.chart(s, fig_feat_evol(D), .65, 1.85, 6.3)
    bullet_panel(s, 7.15, 1.85, 5.5, 2.5, "两代特征体系", [
        "缓存体系(280维)：STL分解、遗传规划、滚动工程特征——信息强但可得性未审计",
        "因果体系(100–144维)：只用当时可知数据，发布时点对齐(align_published)",
        "D基础对齐 → E加动量/回撤 → F加资金面/机构行为/供给 → G加慢漂移",
    ], size=13)
    rows = [
        ["D 因果基础", "100", "日线/周线/政策/月度宏观按发布时点对齐；利差、汇率、股债衍生", "E40现产基线所用"],
        ["E +动量水平", "120", "60/120日动量、z-score、252日回撤、长周期波动", "dev候选"],
        ["F +资金机构", "144", "政策锚利差、互换利差、机构净买入MA、国债期货持仓、供给", "2025最强单模型featF"],
        ["G +慢漂移", "136", "20/60日趋势斜率、下行占比、252日 stochastic、120日跨资产动量", "实验证明无增益，弃用"],
    ]
    table(s, ["变体", "维度", "新增内容", "定位"], rows, .65, 4.55,
          [1.5, .9, 6.6, 3.0], row_h=.55, size=12.5, highlights=(2,))

    # ---- 05 特征筛选 ----
    s = deck.slide("特征筛选：稳定性比单次重要性更可信",
                   "49个特征跨23折重要性恒为零；出现率>30%的核心特征仅7个",
                   "来源：data/43_xgb_feature_stability_bin.csv（176特征×23折）、xgb_never_nonzero_bin.csv")
    deck.chart(s, fig_feat_stab(D), .65, 1.9, 6.3)
    st = D["stab"].sort_values("appear_ratio", ascending=False).head(7)
    rows = [[r["feature"], f"{r['appear_ratio']:.0%}", f"{r['avg_importance']:.4f}"]
            for _, r in st.iterrows()]
    table(s, ["Top稳定特征(按出现率)", "出现率", "平均重要性"], rows,
          7.15, 1.9, [3.1, 1.0, 1.4], row_h=.44, size=12)
    bullet_panel(s, 7.15, 5.15, 5.5, 1.75, "筛选结论", [
        "完全无效49个：慢变量(CPI/PPI/M2)、哑变量、低频托管数据",
        "真正稳定核心：美元/汇率类、波动率类、曲线形态类、信用利差变化",
        "高重要性≠稳定：rsi_24单次第1但出现率仅13%，易过拟合",
    ], size=12.5, accent=GOLD)

    # ---- 06 XGB 时代 ----
    s = deck.slide("XGBoost时代：从三分类瓶颈到二分类上限",
                   "三分类被±3bp噪音标签拖垮；二分类消除标签噪音后，AUC停在0.53–0.63",
                   "来源：data/43_xgb_metrics_bin.txt（5日版）、model_new/data/xgb_metrics_10d.txt（10日版）")
    card(s, .65, 1.8, 3.9, 1.6, "三分类（弃）", "MacroF1 0.385", "±3bp把同方向运动切成\n偏多/震荡/偏空，震荡类58%\n是小幅下行→噪音标签", RED)
    card(s, 4.7, 1.8, 3.9, 1.6, "二分类 5日", "AUC 0.535", "W=180/step=63最优\nAcc 56.0% vs 基线56.7%\n8/23折best_iter≤1", GOLD)
    card(s, 8.75, 1.8, 3.9, 1.6, "二分类 10日(H10)", "AUC 0.634", "W=120、116特征、29折\nAcc 57.7% vs 基线50.3%\n唯一可复用异质组件", TEAL)
    rows = [
        ["训练窗口 W", "90/120/150/180/270/360", "W=180（5d）/ W=120（10d）", "短窗训练不足；长窗被2022后结构断裂污染"],
        ["树结构", "depth 2–5, mcw 1–10", "depth=3, mcw=5–8", "小训练集+高维→强正则更优"],
        ["正则", "alpha 0–5, lambda 1–2", "alpha=1.0, lambda=2.0", "同上；放宽即单特征占比飙升"],
        ["样本时间衰减", "decay 0.998/0.999", "移除", "与滚动窗口叠加→近期过拟合，Acc 0.49→0.44"],
    ]
    table(s, ["优化维度", "搜索范围", "最优", "解释"], rows, .65, 3.75,
          [1.7, 3.2, 2.6, 4.8], row_h=.5, size=12.5)
    bullet_panel(s, .65, 5.95, 12.0, .95, "XGB时代核心结论", [
        "5日信号信噪比不足是硬伤；10日版把AUC抬高10pt，但其缓存特征可得性未审计，只能作融合组件而非独立交付",
    ], size=13)

    # ---- 07 NN 架构 ----
    s = deck.slide("神经网络：BiGRU + Inception + Attention",
                   "以40日原始序列直接捕捉时序依赖，替代手工lag/rolling特征",
                   "来源：model_nn/01_BiGRU训练_二分类_10天.py、05_基础特征对照训练.py")
    box(s, .65, 1.85, 12.0, 1.35)
    text(s, "输入 (B,40,F)  →  BiGRU 双向编码  →  Inception多尺度卷积  →  Self-Attention加权  →  全连接 → 下行概率",
         .95, 2.15, 11.5, .5, 16, NAVY, True)
    text(s, "F=100–144因果特征；hidden=32；dropout=0.3；lr=5e-4；早停patience=8；每折3–6 seed取均值概率",
         .95, 2.72, 11.5, .4, 12.5, MUTED)
    card(s, .65, 3.45, 3.9, 1.5, "序列输入", "40个交易日", "直接消费原始特征矩阵\n无需手工构造lag特征\n模型自学时序模式")
    card(s, 4.7, 3.45, 3.9, 1.5, "训练协议", "28折滚动", "标准化/填补/筛列\n只用梯度训练段\n固定阈值0.5", GOLD)
    card(s, 8.75, 3.45, 3.9, 1.5, "方差控制", "3–6 seed", "单seed方差大\n多seed取均值概率\nAUC稳定+1–2pt", TEAL)
    rows = [
        ["RAW_B(E40现产)", "B缓存特征", "52.2%", "0.522", "同源对照：NN并非自动优于XGB"],
        ["因果D(基础)", "D×100", "53.1%", "0.533", "因果体系起点"],
        ["因果E(+动量)", "E×120", "53.1%", "0.516", "特征更多≠更强"],
        ["因果F(+资金机构)", "F×144", "55.2%", "0.564", "开发期最优NN，2025原生AUC 0.758"],
    ]
    table(s, ["NN变体", "特征", "开发期Acc", "开发期AUC", "结论"], rows, .65, 5.15,
          [2.3, 1.3, 1.5, 1.6, 5.6], row_h=.42, size=12.5, highlights=(3,))

    # ---- 08 NN 训练协议 ----
    s = deck.slide("训练协议：因果性是全部实验的地基",
                   "每个折内的样本切分都保证：训练标签成熟日 < 验证首日 < 测试首日",
                   src_opt9)
    steps = [
        ("1 折边界", "沿用XGB时代28折，每折约63个交易日测试段"),
        ("2 训练集", "测试首日前的全部历史（标签已成熟），可选窗口截断"),
        ("3 验证集", "训练尾部20%，与训练之间隔 purge gap = max(seq_len,horizon)"),
        ("4 标准化", "均值/方差/填补只用梯度训练段计算，防止未来信息渗入"),
        ("5 多seed", "每折3–6个随机种子独立训练，概率取均值降低方差"),
        ("6 断言校验", "代码内assert逐折校验标签成熟时序，违反即报错"),
    ]
    for i, (t, d) in enumerate(steps):
        col, row = i % 3, i // 3
        x = .65 + col * 4.1
        y = 1.9 + row * 1.32
        box(s, x, y, 3.85, 1.16)
        box(s, x, y, .055, 1.16, TEAL, False)
        text(s, t, x + .2, y + .1, 3.5, .38, 15, NAVY, True)
        text(s, d, x + .2, y + .52, 3.55, .6, 12, MUTED)
    bullet_panel(s, .65, 4.75, 12.0, 2.05, "为什么这么讲究", [
        "10日持有期标签在date+10才落地：任何\"用到未来\"的捷径都会让回测虚高",
        "历史上踩过的坑：全局阈值校准在2025反向（57.2%→51.0%）、W=250滚动校准误算78.6%实为48.2%",
        "因此本轮所有方案（含冠军融合）的权重/超参/特征选择全部锁定在开发期，2025只读不写",
    ], size=13.5, accent=RED)

    # ---- 09 优化实验矩阵 ----
    s = deck.slide("优化实验矩阵：25+个受控对照",
                   "四个维度系统验证：标签、两阶段、特征、训练侧适配——结论大多为负， equally informative",
                   src_opt9)
    rows = [
        ["relabel_D 重定义标签", "down→down_ex2(死区2bp)", "dev +1.2pt / 2025原生+18.5pt", "采纳：全线默认标签"],
        ["twostage 两阶段死区", "Stage1筛大变动再判向", "gate从未弃权，与单阶段同", "弃用"],
        ["features_E/F 特征升级", "E动量 / F资金机构", "F最优：dev +2.6pt，2025原生AUC 0.758", "采纳F"],
        ["W504/756/1260 缩窗", "只用近2–5年训练", "2025崩至32–47%", "弃用：反转年放大近期偏差"],
        ["R252 近期加权", "半衰期252天样本权重", "2025原生81.5%但dev −2.5pt", "未采纳：单年证据不足"],
        ["S20 状态相似加权 / S6 加seed / G慢漂移 / pos_weight", "训练侧适配4项", "均无稳定增益", "全部弃用"],
    ]
    table(s, ["实验", "内容", "结果", "决策"], rows, .65, 1.9,
          [3.4, 3.0, 3.6, 2.3], row_h=.62, size=12.5, highlights=(0, 2))
    bullet_panel(s, .65, 5.85, 12.0, 1.0, "方法论收获", [
        "死区标签重定义是NN时代最大单项增益；而所有\"让模型更关注近期\"的训练侧改造（缩窗/加权/相似度）全部失败",
    ], size=13)

    # ---- 10 2025 诊断 ----
    s = deck.slide("2025年误差诊断：定位三大结构",
                   "误差不是均匀噪音：集中在小幅波动日、中波动月、以及10月温和阴跌段",
                   "来源：out/DIAG_幅度分层.csv、DIAG_波动分层.csv、DIAG_分年指标.csv（E40基线）")
    deck.chart(s, fig_amp(D), .65, 1.9, 6.3)
    deck.chart(s, fig_xgb_years(D), 7.15, 1.9, 5.6)
    rows = [
        ["小幅0–2bp (n=86)", "43.0%", "0.315", "本质随机，35%的天数怎么调都救不了"],
        ["大幅5–10bp (n=49)", "75.5%", "0.901", "大行情判别力极强，AUC近0.9"],
        ["10月温和阴跌 (n=17)", "11.8%", "—", "88%天下行，模型只给38%下行概率——系统性漏判"],
        ["中波动月(如2/5/9月)", "19–57%", "0.15–0.60", "信号近似反转，最大失血区"],
    ]
    table(s, ["误差集中区", "Acc", "AUC", "性质"], rows, .65, 5.35,
          [2.6, 1.2, 1.2, 7.4], row_h=.4, size=12, highlights=(2,))

    # ---- 11 融合探索 ----
    s = deck.slide("融合探索：三步排除法走到冠军方案",
                   "同质融合到顶 → 因果XGB无信号 → 缓存XGB10D是唯一有效异质组件",
                   src_optb)
    cards = [
        ("P0 同质融合", "失败", "62个NN模型(含逐seed)等权/秩/贪心/softmax\n全部推不破featF单模型2025 AUC 0.758\n结论：需要异质模型", RED),
        ("P1 因果XGB重训", "无信号", "同折同标签喂XGB，dev AUC仅0.44–0.47\n4种配置（含关闭早停）均低于随机\n结论：因果特征喂树模型不行", RED),
        ("P1' 缓存XGB10D", "有效", "dev AUC 0.622超全部NN\n2025原生0.667\n代价：特征可得性未审计", TEAL),
        ("P2 全池softmax", "冠军", "83模型按dev AUC softmax(T=0.01)加权\n≈80%XGB10D+20%featF系\ndev与2025同时受益", GOLD),
    ]
    for i, (t, v, d, acc) in enumerate(cards):
        x = .65 + i * 3.13
        box(s, x, 1.85, 2.95, 2.6)
        box(s, x, 1.85, 2.95, .5, NAVY, False)
        text(s, t, x + .15, 1.92, 2.7, .36, 14, WHITE, True)
        text(s, v, x + .18, 2.5, 2.6, .45, 20, acc, True)
        text(s, d, x + .18, 3.05, 2.65, 1.35, 11, INK)
    rows = [
        ["单模型 featF", "0.552", "0.564", "0.770", "0.758", "NN最优单模"],
        ["单模型 XGB10D", "0.563", "0.622", "0.576", "0.667", "排序强/概率失准"],
        ["冠军 devw_t001", "0.628", "0.640", "0.737", "0.721", "两段同时第一"],
        ["0.7融合(2025选权)", "0.575", "0.590", "0.765", "0.774", "仅作上限对照"],
    ]
    table(s, ["方案", "dev Acc", "dev AUC", "2025 Acc", "2025原生AUC", "说明"], rows,
          .65, 4.75, [2.5, 1.3, 1.3, 1.3, 1.9, 3.7], row_h=.46, size=12.5,
          highlights=(2,))

    # ---- 12 权重敏感性 ----
    s = deck.slide("核心洞察：权重准则的错位与破局",
                   "开发期AUC准则永远选w≈0.15，而2025最优在w≈0.8——直接调权必踩泄漏",
                   src_optb)
    deck.chart(s, fig_wcrit(D), .65, 1.85, 6.6)
    bullet_panel(s, 7.5, 1.85, 5.15, 2.6, "三条诚实路线", [
        "准则1 dev AUC→w=0.15：2025原生0.688，安全但浪费NN",
        "准则2 dev准确率→w=0.40：dev Acc 60.0%全场峰值，2025原生0.718",
        "准则3 全池softmax(T=0.01)：dev Acc/AUC双第一，2025原生0.721",
    ], size=12.5)
    rows = [
        ["w=0.15 (devAUC准则)", "0.626", "0.569", "0.576", "0.688", "偏XGB"],
        ["w=0.40 (devAcc准则)", "0.618", "0.600", "0.733", "0.718", "平衡点"],
        ["w=0.70 (2025最优)", "0.590", "0.575", "0.765", "0.774", "需2025信息，不可选"],
        ["全池softmax", "0.640", "0.628", "0.737", "0.721", "冠军，自动实现≈2:8混合"],
    ]
    table(s, ["方案", "dev AUC", "dev Acc", "2025 Acc", "2025原生AUC", "性质"], rows,
          .65, 5.15, [2.6, 1.2, 1.2, 1.3, 1.8, 3.9], row_h=.38, size=12,
          highlights=(3,))
    text(s, "权重构成按dev AUC softmax(T=0.01)计算：XGB10D得99.2%→与各NN seed求和后≈80/20",
         .65, 6.95, 12.0, .3, 12, MUTED)

    # ---- 13 冠军方案 ----
    s = deck.slide("冠军方案：CHAMPION_devw_t001",
                   "83个OOS模型池 × 开发期AUC softmax加权融合——全部选择在开发期完成",
                   src_optb)
    box(s, .65, 1.8, 5.9, 2.15)
    text(s, "构成", .9, 1.94, 5.4, .38, 15, MUTED)
    text(s, "≈ 80% XGB10D + 20% featF系NN", .9, 2.32, 5.5, .5, 20, NAVY, True)
    text(s, "wᵢ ∝ exp(devAUCᵢ / 0.01)\nXGB10D dev AUC 0.622 → 权重0.992\nfeatF各seed 0.56–0.59 → 合计≈0.008×N",
         .9, 2.88, 5.5, .95, 13)
    box(s, 6.75, 1.8, 5.9, 2.15)
    text(s, "2025成绩（243天，未参与任何选择）", 7.0, 1.94, 5.4, .38, 15, MUTED)
    text(s, "方向AUC 0.641 / Acc 62.1%", 7.0, 2.32, 5.5, .5, 20, TEAL, True)
    text(s, "原生AUC 0.721 / Acc 73.7%\nvs 现产E40：方向AUC +4.6pt、方向Acc +4.9pt\n开发期Acc 62.8%同样全场第一",
         7.0, 2.88, 5.5, .95, 13)
    deck.chart(s, fig_monthly(D), .65, 4.25, 6.6)
    bullet_panel(s, 7.5, 4.25, 5.15, 2.5, "逐月修复与残余", [
        "8月 0.57→1.00、9月 0.23→0.59、1月 +28pt",
        "5/6月小幅回撤（-32/-15pt）",
        "10月阴跌漏判仍在（0.12→0.18）",
        "下行召回0.30（E40为0.41）：发信号更保守",
    ], size=12.5, accent=GOLD)

    # ---- 14 为什么效果好 ----
    s = deck.slide("为什么效果好：排序 × 标度的互补",
                   "XGB10D提供排序能力，NN修正概率标度——融合后两段同时受益",
                   src_optb)
    deck.chart(s, fig_complement(D), .65, 1.85, 7.4)
    bullet_panel(s, 8.3, 1.85, 4.35, 2.9, "机制拆解", [
        "XGB10D：dev AUC 0.622全场最高，但历史上94%天预测下行，Acc被概率失准拖累",
        "featF NN：概率标度均衡（2025预测下行21.8% vs 真实24.3%），排序偏弱",
        "80/20混合：排序基本保住XGB，标度被NN拉回→dev Acc +6.6pt、2025 AUC +5.5pt",
    ], size=12)
    rows = [
        ["排序能力(dev AUC)", "0.622", "0.564", "0.640", "融合后≈保住较强者"],
        ["概率标度(dev 预测下行率)", "≈0.94", "0.55*", "0.28–0.62", "NN把XGB拉回均衡区"],
        ["dev Acc / 2025方向Acc", "0.563/0.535", "0.552/0.613", "0.628/0.621", "两段同时超越两单模"],
    ]
    table(s, ["维度", "XGB10D", "featF NN", "冠军融合", "效果"], rows, .65, 5.05,
          [3.0, 1.6, 1.6, 1.6, 4.2], row_h=.5, size=12.5, highlights=(2,))
    text(s, "*featF开发期预测下行率为近似值；冠军融合逐段预测率随市况浮动",
         .65, 6.7, 12.0, .3, 11.5, MUTED)

    # ---- 15 演进全景 ----
    s = deck.slide("模型能力演进全景", "六个里程碑，两条线：2025表现（吸引力）与开发期诚实对照（可信度）")
    deck.chart(s, fig_evo(D), .65, 1.82, 11.3)
    rows = [
        ["三分类XGB (2024)", "MacroF1 0.385", "±3bp标签噪音，全面失败", "触发二分类改造"],
        ["二分类XGB-5d (2025.06)", "AUC 0.535", "消除标签噪音，5日信噪比不足", "转10日"],
        ["XGB-H10缓存 (2025.07)", "AUC 0.634", "10日版最优，特征未审计", "留作融合组件"],
        ["NN E40 (2025.09)", "方向AUC 0.595", "因果特征起点，2025失效", "触发诊断"],
        ["NN featF (2025.10)", "原生AUC 0.758", "死区标签+资金特征，最优单模", "融合基座"],
        ["冠军融合 (2025.10)", "原生0.721/方向0.641", "dev锁定选权，两段第一", "冻结待盲测"],
    ]
    table(s, ["里程碑", "关键指标", "得失", "后续动作"], rows, .65, 4.9,
          [2.7, 2.0, 4.6, 2.3], row_h=.33, size=11)

    # ---- 16 不足 ----
    s = deck.slide("不足与风险：三条硬约束", "冠军方案不完美，边界必须讲清楚",
                   src_optb)
    cards = [
        ("R1 组件依赖", "高", ["冠军80%权重依赖XGB10D", "其缓存特征(STL/GP/宏观修订)可得性未审计", "上线前必须完成审计或同折重训替换"], RED),
        ("R2 已知盲区", "高", ["10月型温和阴跌系统性漏判(0.18)", "下行召回仅0.30，漏判79日", "需要事件/资金流实时特征，非融合可解"], RED),
        ("R3 验证边界", "中", ["2025已被多轮查看，单年AUC波动±数pt", "0.79级上限数字含2025选权信息", "2026年仅2样本，盲测未开始"], GOLD),
    ]
    for i, (t, lvl, items, acc) in enumerate(cards):
        x = .65 + i * 4.13
        box(s, x, 1.85, 3.95, 2.9)
        box(s, x, 1.85, .055, 2.9, acc, False)
        text(s, t, x + .22, 1.98, 3.0, .4, 16.5, NAVY, True)
        text(s, f"风险 {lvl}", x + 2.9, 2.02, 1.0, .32, 12, acc, True)
        for j, it in enumerate(items):
            box(s, x + .24, 2.55 + j * .72, .05, .05, acc, False)
            text(s, it, x + .42, 2.45 + j * .72, 3.4, .68, 11.5)
    bullet_panel(s, .65, 5.0, 12.0, 1.85, "已证伪路线（不再投入）", [
        "全局/滚动阈值校准救2025：2025涨跌占比与历史翻转，历史选出的阈值必然反向",
        "三分类、两阶段gate、样本时间衰减、缩窗(W504-1260)、R252/S20加权、pos_weight、慢漂移特征G：全部有对照数据否定",
    ], size=13, accent=RED)

    # ---- 17 规划 ----
    s = deck.slide("后续规划：先补证据，再谈部署", "四个阶段，每阶段有明确交付与通过门槛")
    phases = [
        ("P0 数据与口径 (当前)", "XGB10D特征可得性审计→同折重训替换；事件/文本来源版本冻结",
         "交付：审计报告+可复现组件；门槛：同日历同标签对账一致"),
        ("P1 文本/事件特征", "政策事件日历升级为实时新闻源；LDA/情绪/事件类型嵌入",
         "重点攻击10月型阴跌盲区；门槛：开发期超E基线+配对区间为正"),
        ("P2 概率校准与融合扩展", "Platt/温度校准；加入更多异质组件(轻量时序模型/规则)",
         "门槛：开发期校准曲线改善；融合权重仍只在开发期选"),
        ("P3 影子运行与盲测", "冻结冠军方案，2026年逐日出信号；成本/换手/回撤监控",
         "门槛：2026盲测AUC≥0.55且成本后收益为正——这是唯一真正的验收"),
    ]
    for i, (t, d, g) in enumerate(phases):
        y = 1.85 + i * 1.28
        box(s, .65, y, 12.0, 1.12)
        box(s, .65, y, .055, 1.12, TEAL if i < 3 else GOLD, False)
        text(s, t, .9, y + .1, 3.4, .4, 15, NAVY, True)
        text(s, d, 4.4, y + .1, 8.0, .42, 12.5, INK)
        text(s, g, .9, y + .58, 11.5, .5, 12, MUTED)

    # ---- 18 总结 ----
    s = deck.slide("总结", "一次从特征到融合的完整证据链")
    card(s, .65, 1.9, 3.9, 1.7, "任务", "10日方向", "10年国债收益率\n二分类双口径\n28折滚动+2025验证")
    card(s, 4.7, 1.9, 3.9, 1.7, "最优方案", "冠军融合", "83模型dev选权softmax\n≈80%XGB10D+20%featF\n2025方向AUC 0.641", TEAL)
    card(s, 8.75, 1.9, 3.9, 1.7, "诚实上限", "0.72/0.77", "dev锁定0.721(可交付)\n2025选权0.774(对照)\n差距=选择信息含量", GOLD)
    bullet_panel(s, .65, 3.9, 12.0, 2.9, "三句话结论", [
        "1 特征上：因果可辩护的小特征集(F,144维)胜过280维缓存大集——上线视角必须可追溯",
        "2 模型上：没有单一最强模型，只有互补组合——XGB10D出排序、NN出标度、融合出冠军",
        "3 流程上：所有选择锁开发期让2025只读不写，这是0.72比0.77更值钱的原因，也是下一步盲测的前提",
    ], size=15)
    text(s, "产出：模型演进PPT(10_汇报PPT生成.py) · 本全景PPT(13) · 训练效果总结/1015_2025AUC优化定位.md · OPTB系列CSV",
         .65, 6.85, 12.0, .3, 11.5, MUTED)

    assert len(deck.prs.slides) == 18, len(deck.prs.slides)
    DEST.mkdir(exist_ok=True)
    deck.prs.save(PPT)
    print(f"已生成 {len(deck.prs.slides)} 页: {PPT}")


if __name__ == "__main__":
    main()
