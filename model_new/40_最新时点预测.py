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

import argparse
import importlib.util
import json
import shutil
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
MODELS_DIR = DATA / "models"          # 当日模型缓存（按发出日分目录）


def grade(p) -> str:
    """信号分级：概率分档代替二值方向（观望带 0.40-0.60）"""
    if p is None or pd.isna(p):
        return "-"
    if p >= 0.75:
        return "强烈看下行"
    if p >= 0.60:
        return "看下行"
    if p >= 0.40:
        return "观望"
    if p >= 0.25:
        return "看上行"
    return "强烈看上行"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true",
                    help="忽略当日模型缓存，强制重训")
    ap.add_argument("--no-nn", action="store_true",
                    help="跳过NN支路（仅XGB+融合退化）")
    return ap.parse_args()


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


def _model_dir(asof: pd.Timestamp) -> Path:
    """当日模型缓存目录；清掉其它发出日的旧缓存"""
    mdir = MODELS_DIR / asof.strftime("%Y%m%d")
    if MODELS_DIR.exists():
        for d in MODELS_DIR.iterdir():
            if d.is_dir() and d.name != mdir.name:
                shutil.rmtree(d, ignore_errors=True)
    mdir.mkdir(parents=True, exist_ok=True)
    return mdir


def get_xgb_model(feats: pd.DataFrame, train: pd.DataFrame,
                  asof: pd.Timestamp, h: int, retrain: bool = False):
    """XGB：当日缓存命中则加载（特征列一致才复用），否则训练并保存"""
    mdir = _model_dir(asof)
    mfile, ffile = mdir / f"h{h}_xgb.ubj", mdir / f"h{h}_xgb_features.json"
    if not retrain and mfile.exists():
        try:
            if json.loads(ffile.read_text(encoding="utf-8")) == list(feats.columns):
                m = xgb.XGBClassifier(**PARAMS)
                m.load_model(mfile)
                return m, "缓存"
        except Exception:
            pass
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(train[feats.columns], train["label"].astype(int),
          sample_weight=train["sample_weight"])
    m.save_model(mfile)
    ffile.write_text(json.dumps(list(feats.columns), ensure_ascii=False),
                     encoding="utf-8")
    return m, "新训"


def get_nn_probas(nn17, feats: pd.DataFrame, lab: pd.DataFrame,
                  asof: pd.Timestamp, h: int, retrain: bool = False):
    """NN：按当日缓存加载3-seed权重，否则训练并保存；返回(概率, 来源)"""
    import torch
    df = feats.join(lab, how="inner")
    train = df[(df.index <= asof) & df["label"].notna()
               & df["sample_weight"].notna()].iloc[-756:]
    mi = xgb.XGBClassifier(**{**PARAMS, "n_estimators": 300})
    mi.fit(train[feats.columns], train["label"].astype(int),
           sample_weight=train["sample_weight"])
    top30 = pd.Series(mi.feature_importances_, index=feats.columns)        .sort_values(ascending=False).head(30).index.tolist()

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

    mdir = _model_dir(asof)
    nfile = mdir / f"h{h}_nn.pt"
    if not retrain and nfile.exists():
        try:
            bundle = torch.load(nfile, map_location=device, weights_only=True)
            if bundle.get("features") == top30:
                probas = []
                for k in range(3):
                    model = nn17.BiGRUInceptionAttentionModel(
                        input_size=len(top30), hidden_size=args.hidden_size,
                        num_layers=args.gru_layers, dropout=args.dropout,
                        n_blocks=args.n_blocks).to(device)
                    model.load_state_dict(bundle[f"seed{k}"])
                    model.eval()
                    probas.append(nn17.predict_proba(
                        model, X_t, np.array([asof_pos]), args, device)[0])
                return float(np.mean(probas)), "缓存"
        except Exception as exc:  # noqa: BLE001
            print(f"    [NN缓存失效h{h}]", exc)

    probas, bundle = [], {"features": top30}
    for k in range(3):
        model, info = nn17.train_one_fold(
            args, X_t, y_arr, w_arr, 0, tr_hi, device,
            input_size=len(top30), seed=42 + 1000 * k)
        if model is None:
            break
        probas.append(nn17.predict_proba(model, X_t, np.array([asof_pos]),
                                         args, device)[0])
        bundle[f"seed{k}"] = {key: v.cpu() for key, v in
                              model.state_dict().items()}
    torch.save(bundle, nfile)
    return (float(np.mean(probas)) if probas else float("nan")), "新训"


def predict_asof(feats: pd.DataFrame, y: pd.Series, asof: pd.Timestamp,
                 with_nn: bool = True, retrain: bool = False):
    nn17 = _load_nn17() if with_nn else None
    bd = pd.bdate_range(asof, asof + pd.Timedelta(days=45))
    rows = []
    for h, lab_file in HORIZONS:
        lab = pd.read_csv(DATA / lab_file, index_col=0, parse_dates=True)
        df = feats.join(lab, how="inner")
        train = df[(df.index <= asof) & df["label"].notna()
                   & df["sample_weight"].notna()].iloc[-756:]
        m, xgb_src = get_xgb_model(feats, train, asof, h, retrain)
        p = float(m.predict_proba(feats.loc[[asof]])[:, 1][0])
        p_nn, nn_src = (get_nn_probas(nn17, feats, lab, asof, h, retrain)
                        if with_nn else (float("nan"), "-"))
        p_fu = (p + p_nn) / 2
        med_bp = ((y.shift(-h) - y).abs().dropna()
                  [lambda s: s.index >= (asof - pd.Timedelta(days=730))].median() * 100)
        rows.append({"周期": f"h{h}", "到期日": bd[h].date().isoformat(),
                     "下行概率": round(p, 3),
                     "信号": "下行(债涨)" if p >= 0.5 else "上行(债跌)",
                     "隐含点位估算(%)": round(y[asof] + (1 - 2 * p) * med_bp / 100, 3),
                     "NN概率": round(p_nn, 3),
                     "融合概率": round(p_fu, 3),
                     "融合分级": grade(p_fu),
                     "融合信号": "下行(债涨)" if p_fu >= 0.5 else "上行(债跌)",
                     "融合隐含点位(%)": round(y[asof] + (1 - 2 * p_fu) * med_bp / 100, 3),
                     "训练样本截至": train.index.max().date().isoformat(),
                     "模型来源": f"XGB:{xgb_src}/NN:{nn_src}"})
    return pd.DataFrame(rows)


def make_html(asof, ymode: str, sig: pd.DataFrame, y: pd.Series,
              hist: dict[str, pd.Series], llm: dict | None = None) -> None:
    """自包含SVG：收益率历史 + 融合模型的前向隐含点位射线（单面板简洁版）。
    sig 可含 LLM 修正列；llm 为 41 脚本分析结果时追加解读卡片。"""

    def xmap(d):  # 时间->像素（横轴延伸到最远到期日，给隐含点位留空间）
        mat_max = max(pd.Timestamp(r["到期日"]) for _, r in sig.iterrows())
        t0 = pd.Timestamp("2026-01-01")
        t1 = max(asof, mat_max + pd.Timedelta(days=8))
        return 70 + (pd.Timestamp(d) - t0).total_seconds() / (t1 - t0).total_seconds() * 1030

    def line(pts, color, w=1.6, dash=""):
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        dashattr = f'stroke-dasharray="' + dash + '"' if dash else ""
        return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="{w}" {dashattr}/>')

    y_hist = y[y.index >= asof - pd.Timedelta(days=150)]   # 近150日历日
    W, H, T, B, L, R = 1130, 420, 26, 40, 56, 16
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'font-family="Microsoft YaHei,sans-serif" font-size="12">']
    for mo in pd.date_range("2026-01-01", "2026-12-31", freq="MS"):
        x = xmap(mo)
        svg.append(f'<line x1="{x:.0f}" y1="{T}" x2="{x:.0f}" y2="{H - B}" stroke="#eee"/>')
        svg.append(f'<text x="{x:.0f}" y="{H - 12}" text-anchor="middle" '
                   f'fill="#666">{mo.month}月</text>')

    sig = sig.sort_values("到期日")          # 射线与表格按到期日递增
    levels = [float(r["融合隐含点位(%)"]) for _, r in sig.iterrows()
              if pd.notna(r.get("融合隐含点位(%)"))]
    lo = min(float(y_hist.min()), *levels) if levels else float(y_hist.min())
    hi_ = max(float(y_hist.max()), *levels) if levels else float(y_hist.max())
    pad = (hi_ - lo) * 0.10
    lo, hi_ = lo - pad, hi_ + pad

    def Y(v):
        return T + (hi_ - v) / (hi_ - lo) * (H - T - B)

    for v in np.linspace(lo, hi_, 5):
        svg.append(f'<line x1="{L}" y1="{Y(v):.0f}" x2="{W - R}" y2="{Y(v):.0f}" '
                   f'stroke="#f2f2f2"/>')
        svg.append(f'<text x="{L - 8}" y="{Y(v) + 4:.0f}" text-anchor="end" '
                   f'fill="#666">{v:.2f}</text>')
    svg.append(line([(xmap(d), Y(v)) for d, v in y_hist.items()], "black", 2.2))

    y_now = y[asof]
    svg.append(f'<circle cx="{xmap(asof):.0f}" cy="{Y(y_now):.0f}" r="4.5" '
               f'fill="black" stroke="white" stroke-width="1"/>')
    svg.append(f'<text x="{xmap(asof) - 6:.0f}" y="{Y(y_now) - 12:.0f}" '
               f'fill="black" font-size="12" font-weight="bold">'
               f'发出日 {y_now:.3f}</text>')
    for gi, (_, r) in enumerate(sig.iterrows()):
        c = COLORS[r["周期"]]
        lv = r.get("融合隐含点位(%)")
        if pd.isna(lv):
            continue
        xe, ye = xmap(pd.Timestamp(r["到期日"])), Y(float(lv))
        dy = gi * 18                                   # 标签按周期错行
        svg.append(line([(xmap(asof), Y(y_now)), (xe, ye)], c, 1.8, "7,5"))
        svg.append(f'<path d="M{xe:.0f},{ye - 7:.0f} l6,7 -6,7 -6,-7 z" '
                   f'fill="white" stroke="{c}" stroke-width="2"/>')
        lx = xe + 9 if xe + 130 <= W - R else xe - 9
        anch = "start" if lx > xe else "end"
        svg.append(f'<text x="{lx:.0f}" y="{ye + 5 + dy:.0f}" fill="{c}" '
                   f'font-size="12" font-weight="bold" text-anchor="{anch}">'
                   f'{r["周期"]}融合 {float(lv):.3f}</text>')
    svg.append(f'<text x="{L}" y="{T - 8}" fill="black" font-weight="bold">'
               f'10Y国债收益率(%) 与融合模型隐含点位预测路径  2026</text>')
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
                f"<td style='color:#6b7280'>{fmt(r.get('修正分级'))}</td>"
                f"<td>{fmt(r.get('修正信号'))}</td>"
                f"<td>{fmt(r.get('修正隐含点位(%)'))}</td>")

    tbl = "".join(
        f"<tr><td>{r['周期']}</td><td>{r['到期日']}</td>"
        f"<td style='color:{'green' if r['下行概率'] >= 0.5 else 'red'};font-weight:bold'>"
        f"{r['下行概率']:.3f}</td><td>{r['信号']}</td><td>{r['隐含点位估算(%)']}</td>"
        f"<td>{fmt(r.get('NN概率'))}</td>"
        f"<td style='color:{'green' if (r.get('融合概率') or 0) >= 0.5 else 'red'};font-weight:bold'>"
        f"{fmt(r.get('融合概率'))}</td><td style='color:#6b7280'>{fmt(r.get('融合分级'))}</td>"
        f"<td>{fmt(r.get('融合信号'))}</td>"
        f"<td>{fmt(r.get('融合隐含点位(%)'))}</td>"
        f"{fix_cells(r)}"
        f"<td>{r['训练样本截至']}</td></tr>" for _, r in sig.iterrows())
    fix_head = ("<th>LLM修正概率</th><th>修正分级</th><th>修正信号</th>"
                "<th>修正隐含点位</th>" if has_fix else "")
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
<th>融合分级</th><th>融合信号</th><th>融合隐含点位</th>{fix_head}<th>训练样本截至</th></tr>{tbl}</table>
<p style="color:#888;font-size:12px">隐含点位=当前值±概率加权的历史该周期变动中位数，
非模型原生输出；融合=(XGB+NN概率平均)；下行=收益率下降(债券价格上涨)。
图中彩色虚线=融合模型的隐含点位预测路径（菱形=到期日预测值）；XGB与LLM修正数值见表格。</p></div>
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
    try:
        feats = pd.read_csv(DATA / feats_file, index_col=0, parse_dates=True)
        need_rebuild = feats.index.max() < asof
    except FileNotFoundError:
        need_rebuild = True
        print(f"  特征文件 {feats_file} 不存在，将自动重建")
    if need_rebuild:
        print("[2/4] 原始数据比特征新（或缺失），自动重建特征链(30/32/38) ...")
        rebuild_features()
        feats = pd.read_csv(DATA / feats_file, index_col=0, parse_dates=True)
    else:
        print("[2/4] 特征已覆盖发出日，无需重建")
    assert asof in feats.index, f"发出日{asof.date()}不在特征网格"

    print("[3/4] 四周期预测 ...")
    args = parse_args()
    if args.retrain:
        print("  --retrain: 忽略当日缓存，强制重训")
    sig = predict_asof(feats, y, asof,
                       with_nn=not args.no_nn, retrain=args.retrain)
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
