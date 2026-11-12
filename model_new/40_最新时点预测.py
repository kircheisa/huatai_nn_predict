# -*- coding: utf-8 -*-
"""
最新时点预测：自动判数据新鲜度 -> 选口径 -> 四周期XGB+NN融合预测 -> 结果落盘 -> 曲线HTML

流程:
  1. 读 data_ori/2026_ori_data_new.xlsx:
     - 滚动稳健标准化日频 sheet: 国债10年期最新完整(非0)日期 = 发出日 asof
     - 滚动稳健标准化日频+买卖行为 sheet: asof 当日净买入5列是否有数据
       有 -> 用含净买入特征 43v4_features.csv；无 -> 用无净买入 43v4nb_features.csv
  2. 若 asof 晚于特征网格末日（原始数据已更新但特征未重建），自动重跑
     30(四个horizon) -> 32 -> 38 重建特征链
  3. 四周期(h5/10/15/20)：
     XGB 用截至asof全部标签样本(756窗口, 21协议超参)现场训练预测；
     NN 复用 17 脚本 BiGRU(NN_B协议, 3-seed, 训练窗内Top30特征)现场训练预测；
     融合 = 两者概率平均
  4. 结果:
     data_new/预测记录.csv          追加式台账（同发出日覆盖旧记录）
     data_new/out/时点预测曲线.html  自包含SVG曲线（收益率+XGB/融合隐含点位射线
                                    +各周期概率史+最新信号星标）
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "data_ori" / "2026_ori_data_new.xlsx"
DATA = BASE / "data_new"
OUT = DATA / "out"
LEDGER = DATA / "预测记录.csv"
HTML = OUT / "时点预测曲线.html"

SHEET_D = "滚动稳健标准化日频"
SHEET_B = "滚动稳健标准化日频+买卖行为"
YCOL = "中国:国债到期收益率:10年"
NETBUY = ["净买入：券商自营", "净买入：保险", "净买入：基金",
          "净买入：理财", "净买入：货币基金"]

PARAMS = dict(max_depth=3, n_estimators=600, learning_rate=0.05,
              min_child_weight=8.0, reg_alpha=0.5, reg_lambda=2.0,
              subsample=0.8, colsample_bytree=0.8, tree_method="hist",
              random_state=42, eval_metric="logloss", n_jobs=4)
HORIZONS = [(5, "43v3_labels_bin.csv"), (10, "43v3_labels_bin_h10.csv"),
            (15, "43v3_labels_bin_h15.csv"), (20, "43v3_labels_bin_h20.csv")]
COLORS = {"h5": "#1f77b4", "h10": "#2ca02c", "h15": "#d62728", "h20": "#9467bd"}


def load_sheet_dates(sheet: str) -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_excel(SRC, sheet_name=sheet, header=None)
    hi = raw.index[raw[0] == "指标名称"][0]
    names = list(raw.iloc[hi, 1:])
    df = raw.iloc[hi + 1:].copy()
    df.columns = ["date"] + names
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index().apply(pd.to_numeric, errors="coerce")
    y = df[YCOL].replace(0, pd.NA).dropna() if YCOL in df.columns else pd.Series(dtype=float)
    return df, y


def rebuild_features() -> None:
    """原始数据比特征新时，自动重建 43v3 -> 43v4 -> 43v4nb"""
    py = sys.executable
    for cmd in [["-X", "utf8", "-W", "ignore", str(BASE / "model_new" / "30_新版原始数据_特征构建.py")],
                ["-X", "utf8", "-W", "ignore", str(BASE / "model_new" / "30_新版原始数据_特征构建.py"), "--horizon", "10"],
                ["-X", "utf8", "-W", "ignore", str(BASE / "model_new" / "30_新版原始数据_特征构建.py"), "--horizon", "15"],
                ["-X", "utf8", "-W", "ignore", str(BASE / "model_new" / "30_新版原始数据_特征构建.py"), "--horizon", "20"],
                ["-X", "utf8", str(BASE / "model_new" / "32_派生特征增强.py")],
                ["-X", "utf8", str(BASE / "model_new" / "38_无买卖行为_特征构建.py")]]:
        print("  重建:", Path(cmd[-1]).name, " ".join(cmd[-2:]))
        r = subprocess.run([py] + cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout[-1500:], r.stderr[-1500:])
            raise RuntimeError(f"特征重建失败: {cmd[-1]}")


def _load_nn17():
    spec = importlib.util.spec_from_file_location(
        "nn17", BASE / "model_nn" / "17_BiGRU训练_新基础数据_二分类.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nn_predict(nn17, feats: pd.DataFrame, lab: pd.DataFrame,
               asof: pd.Timestamp, h: int) -> float:
    """NN支路：训练窗内XGB重要性Top30 + 17脚本BiGRU(NN_B, 3-seed)预测asof"""
    df = feats.join(lab, how="inner")
    train = df[(df.index <= asof) & df["label"].notna()
               & df["sample_weight"].notna()].iloc[-756:]
    mi = xgb.XGBClassifier(**{**PARAMS, "n_estimators": 300})
    mi.fit(train[feats.columns], train["label"].astype(int),
           sample_weight=train["sample_weight"])
    top30 = pd.Series(mi.feature_importances_, index=feats.columns)\
        .sort_values(ascending=False).head(30).index.tolist()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Namespace(seq_len=60, purge_gap=-1, val_rate=0.2, epochs=80,
                     lr=5e-4, weight_decay=1e-4, patience=12, min_epochs=10,
                     label_smoothing=0.05, class_weight=True, batch_size=32,
                     noise_std=0.0, grad_clip=1.0, min_lr=1e-5, n_blocks=1,
                     hidden_size=64, gru_layers=2, dropout=0.5)
    full = feats[top30]
    y_all = lab["label"].reindex(feats.index)
    w_all = lab["sample_weight"].reindex(feats.index)
    asof_pos, tr_hi = len(feats) - 1, len(feats) - 1 - h
    lo = max(0, asof_pos - 1260)
    mu = np.nanmean(full.values[lo:tr_hi + 1], axis=0)
    sd = np.nanstd(full.values[lo:tr_hi + 1], axis=0) + 1e-8
    Xs = np.clip(np.nan_to_num((full.values - mu) / sd), -8, 8).astype(np.float32)
    X_t = torch.from_numpy(Xs)
    y_arr = np.nan_to_num(y_all.values, nan=0).astype(int)
    w_arr = np.nan_to_num(w_all.values, nan=1.0).astype(float)
    probas = []
    for k in range(3):
        model, info = nn17.train_one_fold(
            args, X_t, y_arr, w_arr, 0, tr_hi, device,
            input_size=len(top30), seed=42 + 1000 * k)
        if model is None:
            break
        probas.append(nn17.predict_proba(model, X_t, np.array([asof_pos]),
                                         args, device)[0])
    return float(np.mean(probas)) if probas else float("nan")


def predict_asof(feats: pd.DataFrame, y: pd.Series, asof: pd.Timestamp,
                 with_nn: bool = True):
    nn17 = _load_nn17() if with_nn else None
    bd = pd.bdate_range(asof, asof + pd.Timedelta(days=45))
    rows = []
    for h, lab_file in HORIZONS:
        lab = pd.read_csv(DATA / lab_file, index_col=0, parse_dates=True)
        df = feats.join(lab, how="inner")
        train = df[(df.index <= asof) & df["label"].notna()
                   & df["sample_weight"].notna()].iloc[-756:]
        m = xgb.XGBClassifier(**PARAMS)
        m.fit(train[feats.columns], train["label"].astype(int),
              sample_weight=train["sample_weight"])
        p = float(m.predict_proba(feats.loc[[asof]])[:, 1][0])
        p_nn = nn_predict(nn17, feats, lab, asof, h) if with_nn else float("nan")
        p_fu = (p + p_nn) / 2
        med_bp = ((y.shift(-h) - y).abs().dropna()
                  [lambda s: s.index >= (asof - pd.Timedelta(days=730))].median() * 100)
        rows.append({"周期": f"h{h}", "到期日": bd[h].date().isoformat(),
                     "下行概率": round(p, 3),
                     "信号": "下行(债涨)" if p >= 0.5 else "上行(债跌)",
                     "隐含点位估算(%)": round(y[asof] + (1 - 2 * p) * med_bp / 100, 3),
                     "NN概率": round(p_nn, 3),
                     "融合概率": round(p_fu, 3),
                     "融合信号": "下行(债涨)" if p_fu >= 0.5 else "上行(债跌)",
                     "融合隐含点位(%)": round(y[asof] + (1 - 2 * p_fu) * med_bp / 100, 3),
                     "训练样本截至": train.index.max().date().isoformat()})
    return pd.DataFrame(rows)


def make_html(asof, ymode: str, sig: pd.DataFrame, y: pd.Series,
              hist: dict[str, pd.Series], llm: dict | None = None) -> None:
    """自包含SVG：上=收益率，下=各周期下行概率史+最新信号点
    sig 可含 LLM 修正列（修正概率/修正信号/修正隐含点位(%)），llm 为
    41 脚本的分析结果（情绪/要点/理由/新闻）时追加解读卡片。"""

    def xmap(d):  # 时间->像素（横轴延伸到最远到期日，给隐含点位留空间）
        mat_max = max(pd.Timestamp(r["到期日"]) for _, r in sig.iterrows())
        t0 = pd.Timestamp("2026-01-01")
        t1 = max(asof, mat_max + pd.Timedelta(days=6))
        return 70 + (pd.Timestamp(d) - t0).total_seconds() / (t1 - t0).total_seconds() * 1030

    def line(pts, color, w=1.6, dash=""):
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="{w}" {f"stroke-dasharray={dash}" if dash else ""}/>')

    y26 = y[y.index >= "2026-01-01"]
    W, H1, H2 = 1130, 300, 300
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H1 + H2 + 70}" '
           f'font-family="Microsoft YaHei,sans-serif" font-size="12">']
    # 网格与月刻度
    for mo in pd.date_range("2026-01-01", "2026-12-31", freq="MS"):
        x = xmap(mo)
        for top, hh in [(20, H1), (H1 + 50, H2)]:
            svg.append(f'<line x1="{x:.0f}" y1="{top}" x2="{x:.0f}" '
                       f'y2="{top + hh - 20}" stroke="#eee"/>')
        svg.append(f'<text x="{x:.0f}" y="{H1 + H2 + 58}" text-anchor="middle" '
                   f'fill="#666">{mo.month}月</text>')
    # 面板1: 收益率 + 前向隐含点位（XGB实心圆 / 融合空心菱形 / 修正×）
    imp = [r["隐含点位估算(%)"] for _, r in sig.iterrows()]
    imp += [r["融合隐含点位(%)"] for _, r in sig.iterrows()
            if pd.notna(r.get("融合隐含点位(%)"))] or imp
    imp += [r["修正隐含点位(%)"] for _, r in sig.iterrows()
            if pd.notna(r.get("修正隐含点位(%)"))] or imp
    lo, hi_ = min(y26.min(), min(imp)), y26.max()
    pad = (hi_ - lo) * 0.08
    lo, hi_ = lo - pad, hi_ + pad

    def y1(v):
        return 20 + (hi_ - v) / (hi_ - lo) * (H1 - 40)

    svg.append(line([(xmap(d), y1(v)) for d, v in y26.items()], "black", 2.2))
    # 前向隐含点位：发出日->各到期日的虚线射线 + 端点标注
    y_now = y[asof]
    for _, r in sig.iterrows():
        c = COLORS[r["周期"]]
        xe, ye = xmap(pd.Timestamp(r["到期日"])), y1(r["隐含点位估算(%)"])
        svg.append(line([(xmap(asof), y1(y_now)), (xe, ye)], c, 1.4, "6,4"))
        svg.append(f'<circle cx="{xe:.0f}" cy="{ye:.0f}" r="4.5" fill="{c}" '
                   f'stroke="white" stroke-width="1.2"/>')
        svg.append(f'<text x="{xe - 4:.0f}" y="{ye + 18:.0f}" fill="{c}" '
                   f'font-size="11" font-weight="bold">'
                   f'{r["周期"]}→{r["隐含点位估算(%)"]:.3f}</text>')
        # 融合隐含点位：空心菱形 + 细虚线
        if pd.notna(r.get("融合隐含点位(%)")):
            yf = y1(r["融合隐含点位(%)"])
            svg.append(line([(xmap(asof), y1(y_now)), (xe, yf)], c, 1.0, "2,4"))
            svg.append(f'<path d="M{xe:.0f},{yf - 6:.0f} l5,6 -5,6 -5,-6 z" '
                       f'fill="white" stroke="{c}" stroke-width="1.6"/>')
            svg.append(f'<text x="{xe + 8:.0f}" y="{yf + 4:.0f}" fill="{c}" '
                       f'font-size="10">融{r["融合隐含点位(%)"]:.3f}</text>')
        # LLM修正隐含点位：×标记
        if pd.notna(r.get("修正隐含点位(%)")):
            yc = y1(r["修正隐含点位(%)"])
            svg.append(f'<path d="M{xe - 6:.0f},{yc - 6:.0f} l12,12 '
                       f'm0,-12 l-12,12" stroke="{c}" stroke-width="2.4" '
                       f'fill="none"/>')
            svg.append(f'<text x="{xe - 4:.0f}" y="{yc - 12:.0f}" fill="{c}" '
                       f'font-size="10" font-weight="bold">修'
                       f'{r["修正隐含点位(%)"]:.3f}</text>')
    svg.append(f'<circle cx="{xmap(asof):.0f}" cy="{y1(y_now):.0f}" r="4" '
               f'fill="black" stroke="white" stroke-width="1"/>')
    svg.append(f'<text x="{xmap(asof) - 4:.0f}" y="{y1(y_now) - 10:.0f}" '
               f'fill="black" font-size="11" font-weight="bold">'
               f'发出日 {y_now:.3f}</text>')
    for v in np.linspace(lo, hi_, 5):
        svg.append(f'<line x1="70" y1="{y1(v):.0f}" x2="1100" y2="{y1(v):.0f}" '
                   f'stroke="#f2f2f2"/><text x="62" y="{y1(v) + 4:.0f}" '
                   f'text-anchor="end" fill="#666">{v:.2f}</text>')
    svg.append(f'<text x="76" y="36" fill="black" font-weight="bold">'
               f'10Y国债收益率(%)  2026</text>')
    # 面板2: 概率曲线
    def y2(v):
        return H1 + 50 + (1 - v) * (H2 - 70)

    svg.append(f'<line x1="70" y1="{y2(0.5):.0f}" x2="1100" y2="{y2(0.5):.0f}" '
               f'stroke="#999" stroke-dasharray="6,4"/>')
    svg.append(f'<text x="1104" y="{y2(0.5) + 4:.0f}" fill="#999">0.5</text>')
    lg_x = 76
    for name, s in hist.items():
        h = name.split("h")[-1]
        svg.append(line([(xmap(d), y2(v)) for d, v in s.dropna().items()],
                        COLORS.get(f"h{h}", "gray"), 1.3))
        svg.append(f'<rect x="{lg_x}" y="{H1 + 8}" width="26" height="4" '
                   f'fill="{COLORS.get(f"h{h}", "gray")}"/><text x="{lg_x + 32}" '
                   f'y="{H1 + 14}" fill="#333">{name}</text>')
        lg_x += 110
    # 最新信号点（右侧星标：实心=XGB，空心=融合）
    for _, r in sig.iterrows():
        c = COLORS[r["周期"]]
        xx, yy = xmap(asof), y2(r["下行概率"])
        svg.append(f'<path d="M{xx:.0f},{yy:.0f} m0,-9 l2.4,5.4 6,0.6 -4.5,4 '
                   f'1.3,5.9 -5.2,-3.1 -5.2,3.1 1.3,-5.9 -4.5,-4 6,-0.6 z" '
                   f'fill="{c}" stroke="black" stroke-width="0.6"/>')
        svg.append(f'<text x="{xx - 6:.0f}" y="{yy - 14:.0f}" fill="{c}" '
                   f'font-weight="bold">{r["下行概率"]:.2f}</text>')
        if pd.notna(r.get("融合概率")):
            yf2 = y2(min(1.0, r["融合概率"] + 0.0))
            svg.append(f'<path d="M{xx - 18:.0f},{yf2:.0f} m0,-7 l1.9,4.2 '
                       f'4.7,0.5 -3.5,3.1 1,4.6 -4.1,-2.4 -4.1,2.4 1,-4.6 '
                       f'-3.5,-3.1 4.7,-0.5 z" fill="white" stroke="{c}" '
                       f'stroke-width="1.4"/>')
            svg.append(f'<text x="{xx - 44:.0f}" y="{yf2 + 4:.0f}" fill="{c}" '
                       f'font-size="10">融{r["融合概率"]:.2f}</text>')
    svg.append(f'<text x="76" y="{H1 + 44}" fill="black" font-weight="bold">'
               f'各周期下行概率（样本外预测史 + ★{asof.date()}最新信号）</text>')
    svg.append("</svg>")

    def fmt(v, spec=".3f"):
        if pd.isna(v):
            return "-"
        return v if isinstance(v, str) else f"{v:{spec}}"

    has_fix = any(pd.notna(r.get("修正概率")) for _, r in sig.iterrows())

    def fix_cells(r):
        if not has_fix:
            return ""
        p = r.get("修正概率")
        col = "green" if (p or 0) >= 0.5 else "red"
        return (f"<td style='color:{col};font-weight:bold'>{fmt(p)}</td>"
                f"<td>{fmt(r.get('修正信号'))}</td>"
                f"<td>{fmt(r.get('修正隐含点位(%)'))}</td>")

    tbl = "".join(
        f"<tr><td>{r['周期']}</td><td>{r['到期日']}</td>"
        f"<td style='color:{'green' if r['下行概率'] >= 0.5 else 'red'};font-weight:bold'>"
        f"{r['下行概率']:.3f}</td><td>{r['信号']}</td><td>{r['隐含点位估算(%)']}</td>"
        f"<td>{fmt(r.get('NN概率'))}</td>"
        f"<td style='color:{'green' if (r.get('融合概率') or 0) >= 0.5 else 'red'};font-weight:bold'>"
        f"{fmt(r.get('融合概率'))}</td><td>{fmt(r.get('融合信号'))}</td>"
        f"<td>{fmt(r.get('融合隐含点位(%)'))}</td>"
        f"{fix_cells(r)}"
        f"<td>{r['训练样本截至']}</td></tr>" for _, r in sig.iterrows())
    fix_head = ("<th>LLM修正概率</th><th>修正信号</th><th>修正隐含点位</th>"
                if has_fix else "")
    llm_card = ""
    if llm:
        pts = "".join(f"<li>{p}</li>" for p in llm.get("信息要点", []))
        news = llm.get("新闻样本") or []
        news_html = "\n".join(news[:30]) if news else "（本次未命中相关快讯）"
        senti = str(llm.get("债市情绪", "—"))
        scolor = "green" if "偏多" in senti else ("red" if "偏空" in senti else "gray")
        llm_card = f"""
<div class="card"><h3>大模型信息修正（{llm.get('model','GLM')} · 有界±0.08）</h3>
<p><b>债市情绪：</b><span style="color:{scolor};font-weight:bold">{senti}</span>
&nbsp;|&nbsp;<b>信息质量：</b>{llm.get('信息质量','—')}
&nbsp;|&nbsp;<b>命中快讯：</b>{len(news)} 条</p>
<p><b>信息要点：</b></p><ul>{pts}</ul>
<p><b>修正理由：</b>{llm.get('理由','—')}</p>
<details><summary>查看搜集到的快讯原文</summary>
<pre style="white-space:pre-wrap;font-size:12px;color:#555">{news_html}</pre>
</details>
<p style="color:#888;font-size:12px">LLM修正基于公开快讯的定性判断，
未经历史回测验证，仅作模型信号的小幅倾斜参考。</p></div>"""

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>时点预测曲线 {asof.date()}</title>
<style>body{{font-family:'Microsoft YaHei';margin:24px;background:#fafafa}}
.card{{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:18px;
margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;
padding:6px 12px;text-align:center}}th{{background:#f5f7fa}}</style></head><body>
<h2>国债10Y方向 时点预测 · {asof.date()}</h2>
<div class="card"><b>发出日：</b>{asof.date()}（日频sheet国债10Y最新完整日）
&nbsp;|&nbsp;<b>特征口径：</b>{ymode}
&nbsp;|&nbsp;<b>当日收盘：</b>{y[asof]:.4f}%
&nbsp;|&nbsp;<b>协议：</b>滚动756窗口/21脚本超参/四周期XGB+NN融合</div>
<div class="card"><table><tr><th>周期</th><th>到期日</th><th>XGB下行概率</th>
<th>XGB信号</th><th>XGB隐含点位</th><th>NN概率</th><th>融合概率</th>
<th>融合信号</th><th>融合隐含点位</th>{fix_head}<th>训练样本截至</th></tr>{tbl}</table>
<p style="color:#888;font-size:12px">隐含点位=当前值±概率加权的历史该周期变动中位数，
非模型原生输出；融合=(XGB+NN概率平均)；下行=收益率下降(债券价格上涨)。
图中：实心圆/实心星=XGB，空心菱形/空心星=融合，×=LLM修正。</p></div>
{llm_card}
<div class="card">{''.join(svg)}</div>
</body></html>"""
    HTML.write_text(html, encoding="utf-8")


def main() -> None:
    print("[1/4] 读取原始数据，判定发出日与净买入可用性 ...")
    df_d, y_d = load_sheet_dates(SHEET_D)
    df_b, y_b = load_sheet_dates(SHEET_B)
    asof = y_d.index.max()
    netbuy_ok = (asof in df_b.index
                 and df_b.loc[asof, [c for c in NETBUY if c in df_b.columns]]
                 .notna().sum() >= 3)
    ymode = "含净买入(43v4)" if netbuy_ok else "无净买入(43v4nb，日频sheet口径)"
    print(f"  日频10Y最新完整日: {asof.date()}  |  买卖行为表10Y最新完整日: "
          f"{y_b.index.max().date() if len(y_b) else '无'}  |  asof净买入可用: {netbuy_ok}")
    print(f"  -> 特征口径: {ymode}")
    y = y_d

    feats_file = "43v4_features.csv" if netbuy_ok else "43v4nb_features.csv"
    feats = pd.read_csv(DATA / feats_file, index_col=0, parse_dates=True)
    if feats.index.max() < asof:
        print("[2/4] 原始数据比特征新，自动重建特征链(30/32/38) ...")
        rebuild_features()
        feats = pd.read_csv(DATA / feats_file, index_col=0, parse_dates=True)
    else:
        print("[2/4] 特征已覆盖发出日，无需重建")
    assert asof in feats.index, f"发出日{asof.date()}不在特征网格"

    print("[3/4] 四周期预测 ...")
    sig = predict_asof(feats, y, asof)
    sig.insert(0, "发出日", asof.date().isoformat())
    sig.insert(1, "特征口径", ymode)
    sig.insert(2, "当日收益率", round(y[asof], 4))
    print(sig.to_string(index=False))

    if LEDGER.exists():
        old = pd.read_csv(LEDGER, dtype={"发出日": str})
        old = old[old["发出日"] != asof.date().isoformat()]
    else:
        old = pd.DataFrame()
    pd.concat([old, sig], ignore_index=True).to_csv(
        LEDGER, index=False, encoding="utf-8-sig")
    print(f"  台账: {LEDGER}（累计{len(old) + len(sig)}条记录）")

    print("[4/4] 生成曲线HTML ...")
    pref = "xgbv4nb" if not netbuy_ok else "xgbv4"
    hist = {}
    for h, f in [(10, f"{pref}h10"), (15, "xgbv4h15"), (20, f"{pref}h20")]:
        p = OUT / f"{f}_test_predictions.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["date"] = pd.to_datetime(d["date"])
            hist[f"XGB h{h}"] = d.set_index("date")["proba_down"]
    make_html(asof, ymode, sig, y, hist)
    print(f"  输出: {HTML}")


if __name__ == "__main__":
    main()
