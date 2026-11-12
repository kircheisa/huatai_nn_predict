# -*- coding: utf-8 -*-
"""
最新时点预测：自动判数据新鲜度 -> 选口径 -> 四周期XGB预测 -> 结果落盘 -> 曲线HTML

流程:
  1. 读 data_ori/2026_ori_data_new.xlsx:
     - 滚动稳健标准化日频 sheet: 国债10年期最新完整(非0)日期 = 发出日 asof
     - 滚动稳健标准化日频+买卖行为 sheet: asof 当日净买入5列是否有数据
       有 -> 用含净买入特征 43v4_features.csv；无 -> 用无净买入 43v4nb_features.csv
  2. 若 asof 晚于特征网格末日（原始数据已更新但特征未重建），自动重跑
     30(四个horizon) -> 32 -> 38 重建特征链
  3. 四周期(h5/10/15/20) XGB 用截至asof全部标签样本(756窗口, 21协议超参)
     现场训练后预测 asof 当日；输出概率/信号/到期日/隐含点位
  4. 结果:
     data_new/预测记录.csv          追加式台账（同发出日覆盖旧记录）
     data_new/out/时点预测曲线.html  自包含SVG曲线（收益率+各周期概率史+最新信号）
"""

from __future__ import annotations

import subprocess
import sys
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


def predict_asof(feats: pd.DataFrame, y: pd.Series, asof: pd.Timestamp):
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
        med_bp = ((y.shift(-h) - y).abs().dropna()
                  [lambda s: s.index >= (asof - pd.Timedelta(days=730))].median() * 100)
        rows.append({"周期": f"h{h}", "到期日": bd[h].date().isoformat(),
                     "下行概率": round(p, 3),
                     "信号": "下行(债涨)" if p >= 0.5 else "上行(债跌)",
                     "隐含点位估算(%)": round(y[asof] + (1 - 2 * p) * med_bp / 100, 3),
                     "训练样本截至": train.index.max().date().isoformat()})
    return pd.DataFrame(rows)


def make_html(asof, ymode: str, sig: pd.DataFrame, y: pd.Series,
              hist: dict[str, pd.Series]) -> None:
    """自包含SVG：上=收益率，下=各周期下行概率史+最新信号点"""

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
    # 面板1: 收益率 + 前向隐含点位
    implied_min = min(r["隐含点位估算(%)"] for _, r in sig.iterrows())
    lo, hi_ = min(y26.min(), implied_min), y26.max()
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
    # 最新信号点（右侧星标）
    for _, r in sig.iterrows():
        c = COLORS[r["周期"]]
        xx, yy = xmap(asof), y2(r["下行概率"])
        svg.append(f'<path d="M{xx:.0f},{yy:.0f} m0,-9 l2.4,5.4 6,0.6 -4.5,4 '
                   f'1.3,5.9 -5.2,-3.1 -5.2,3.1 1.3,-5.9 -4.5,-4 6,-0.6 z" '
                   f'fill="{c}" stroke="black" stroke-width="0.6"/>')
        svg.append(f'<text x="{xx - 6:.0f}" y="{yy - 14:.0f}" fill="{c}" '
                   f'font-weight="bold">{r["下行概率"]:.2f}</text>')
    svg.append(f'<text x="76" y="{H1 + 44}" fill="black" font-weight="bold">'
               f'各周期下行概率（样本外预测史 + ★{asof.date()}最新信号）</text>')
    svg.append("</svg>")

    tbl = "".join(
        f"<tr><td>{r['周期']}</td><td>{r['到期日']}</td>"
        f"<td style='color:{'green' if r['下行概率'] >= 0.5 else 'red'};font-weight:bold'>"
        f"{r['下行概率']:.3f}</td><td>{r['信号']}</td><td>{r['隐含点位估算(%)']}</td>"
        f"<td>{r['训练样本截至']}</td></tr>" for _, r in sig.iterrows())
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
&nbsp;|&nbsp;<b>协议：</b>滚动756窗口/21脚本超参/四周期独立XGB</div>
<div class="card"><table><tr><th>周期</th><th>到期日</th><th>下行概率</th>
<th>信号</th><th>隐含点位(启发式)</th><th>训练样本截至</th></tr>{tbl}</table>
<p style="color:#888;font-size:12px">隐含点位=当前值±概率加权的历史该周期变动中位数，
非模型原生输出；下行=收益率下降(债券价格上涨)。</p></div>
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
