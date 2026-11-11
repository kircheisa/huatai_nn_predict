# -*- coding: utf-8 -*-
"""
新基础数据（data_new/国债10年基础数据.xlsx，3 sheet：日/周/月频）特征构建

与旧管线（01_数据预处理.py 的 Wind 全量导出）不同，本数据为精简基础库：
  - 缺失全部以 0 占位（无 NaN），0 -> NaN 后使用
  - 含 10 年期国债收益率本身（中债国债到期收益率:10年），标签就地构造
  - 周频/月频无发布日期列，用保守发布滞后对齐（防前视）：
      周频: 发布日 = 统计日 + PUBLISH_LAG_WEEKLY 天（默认1，即次日才可用）
      月频: 发布日 = 统计日 + PUBLISH_LAG_MONTHLY 天（默认15，约次月中旬）
    对齐用 merge_asof(direction='backward', allow_exact_matches=False)，
    只允许"严格早于当前交易日"的最近一次发布。

特征族（全部因果滚动，只用量化当日及更早数据）：
  F1 利差/价差: 国债期限利差、国开-国债(税收)、中短票-国债(信用)、AAA-AA+(等级)、
               中美利差、GC-DR(非银融资溢价)、存单-国开
  F2 逐指标三件套: robust z-score(252d) + 20日变化 + 60日变化（利率bp差/其余pct）
  F3 10Y收益率技术面: 动量/均线偏离/60日z/52周分位/RSI/波动/布林位置/连涨跌
  F4 事件: 降准公告哑变量 + 方向

输出（data_new/）:
  43v2_features.csv       特征矩阵（index=交易日）
  43v2_labels_bin.csv     label(delta<0=1) + sample_weight(死区0.3) + rate_delta_5d
  43v2_build_meta.txt     构建摘要（特征数/NaN画像/标签分布）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "data_new" / "国债10年基础数据.xlsx"
OUT_DIR = BASE / "data_new"

YIELD_COL = "中债国债到期收益率:10年"          # 标签源（另一列相关0.9996，弃用）
DEADZONE_BP = 2.0                              # |Δ|<2bp 死区样本权重0.3
PUBLISH_LAG_WEEKLY = 1                         # 周频发布滞后（自然日）
PUBLISH_LAG_MONTHLY = 15                       # 月频发布滞后（自然日）
Z_WIN = 252                                    # robust z 窗口（交易日）
Z_MIN = 120                                    # robust z 最小观察数

# 0 是真实信息（无公告=0）的事件列，不做 0->NaN
EVENT_COLS = ["中国:人民币存款准备金率(变动公告日期):大型存款类金融机构"]

# 直接弃用的列：阶梯常数（IQR=0）与稀疏事件日期（信息已由公告哑变量承载）
DROP_COLS = {
    "中国:法定存款准备金率:大型银行",
    "中国:人民币存款准备金率:大型存款类金融机构:变动日期",
}


def load_sheet(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """定位'指标名称'行，去重列名，返回数值矩阵（index=日期升序）"""
    raw = xl.parse(sheet, header=None)
    hi = raw.index[raw[0] == "指标名称"][0]
    names = list(raw.iloc[hi, 1:])
    for i, n in enumerate(names):              # 重复指标名加编号
        if names[: i + 1].count(n) > 1:
            names[i] = f"{n}#{i}"
    df = raw.iloc[hi + 1:].copy()
    df.columns = ["date"] + names
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df.apply(pd.to_numeric, errors="coerce")


def zero_to_nan(d: pd.DataFrame) -> pd.DataFrame:
    """0 -> NaN（事件列除外）；随后删除全空列与完全重复列"""
    d = d.copy()
    d = d.drop(columns=[c for c in DROP_COLS if c in d.columns])
    for c in EVENT_COLS:
        if c in d.columns:
            d[c] = d[c].fillna(0.0)
    d = d.mask(d == 0)
    # 全空列（如 法定存款准备金率:大型银行 全为0）
    dead = [c for c in d.columns if d[c].notna().sum() == 0]
    d = d.drop(columns=dead)
    # 值完全相同的重复列，仅保留首列
    keep, seen = [], {}
    for c in d.columns:
        key = tuple(d[c].fillna(-9e18).values[:2000])
        if key in seen:
            continue
        seen[key] = c
        keep.append(c)
    d = d[keep]
    return d, dead


def merge_lagged(grid_idx: pd.DatetimeIndex, low: pd.DataFrame,
                 lag_days: int) -> pd.DataFrame:
    """低频表按'统计日+lag'发布，merge_asof 向后对齐到交易日（严禁当日可用）"""
    # 周频sheet是"各指标发布日交错"的稀疏表：必须先列级ffill，
    # 否则行级asof对齐时未到发布日的列永远是NaN
    pub = low.ffill().dropna(how="all").copy()
    pub.index = pub.index + pd.Timedelta(days=lag_days)
    pub = pub[~pub.index.duplicated(keep="last")].sort_index()
    pub.index.name = "pub_date"
    left = pd.DataFrame({"date": grid_idx.values}, index=grid_idx)
    left.index.name = None
    right = pub.reset_index()
    merged = pd.merge_asof(left, right, left_on="date", right_on="pub_date",
                           direction="backward", allow_exact_matches=False)
    return merged.drop(columns=["date", "pub_date"]).set_index(grid_idx)


def robust_z(s: pd.Series, win: int = Z_WIN, minp: int = Z_MIN) -> pd.Series:
    """滚动稳健标准化: (x - median) / IQR，只含历史窗口"""
    med = s.rolling(win, min_periods=minp).median()
    q1 = s.rolling(win, min_periods=minp).quantile(0.25)
    q3 = s.rolling(win, min_periods=minp).quantile(0.75)
    iqr = (q3 - q1).clip(lower=1e-9)
    return (s - med) / iqr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=5,
                   help="预测周期（交易日）；h5输出43v2_*，其余加_h{H}后缀")
    return p.parse_args()


def build(horizon: int) -> None:
    sfx = "" if horizon == 5 else f"_h{horizon}"
    xl = pd.ExcelFile(SRC)
    daily, weekly, monthly = (load_sheet(xl, s) for s in xl.sheet_names)
    daily, dead_cols = zero_to_nan(daily)
    weekly = weekly.mask(weekly == 0)
    monthly = monthly.mask(monthly == 0)

    # ---- 交易日网格 & 标签源（收益率非零的日期）----
    y = daily[YIELD_COL].dropna()
    grid = y.index
    # 稀疏发布的价格类日频列（玻璃/存单等）在交易日网格上ffill（限20日防空段僵尸值）
    daily_g = daily.reindex(grid).ffill(limit=20)
    y.name = "yield10y"

    # ---- F1 利差/价差（日频水平就地相减）----
    C = daily_g
    spreads: dict[str, pd.Series] = {}
    pairs = [
        ("期限利差_10Y-1Y", "中债国债到期收益率:10年", "中债国债到期收益率:1年"),
        ("期限利差_10Y-3Y", "中债国债到期收益率:10年", "中债国债到期收益率:3年"),
        ("期限利差_10Y-5Y", "中债国债到期收益率:10年", "中债国债到期收益率:5年"),
        ("期限利差_5Y-1Y", "中债国债到期收益率:5年", "中债国债到期收益率:1年"),
        ("税收利差_国开5Y-国债5Y", "中债国开债到期收益率:5年", "中债国债到期收益率:5年"),
        ("税收利差_国开3Y-国债3Y", "中债国开债到期收益率:3年", "中债国债到期收益率:3年"),
        ("信用利差_AA+_3Y-国债3Y", "中债中短期票据到期收益率(AA+):3年", "中债国债到期收益率:3年"),
        ("信用利差_AA+_1Y-国债1Y", "中债中短期票据到期收益率(AA+):1年", "中债国债到期收益率:1年"),
        ("信用利差_AAA_3Y-国债3Y", "中债中短期票据到期收益率(AAA):3年", "中债国债到期收益率:3年"),
        ("等级利差_AA+-AAA_3Y", "中债中短期票据到期收益率(AA+):3年", "中债中短期票据到期收益率(AAA):3年"),
        ("等级利差_AA+-AAA_1Y", "中债中短期票据到期收益率(AA+):1年", "中债中短期票据到期收益率(AAA):1年"),
        ("中美利差_10Y", "中债国债到期收益率:10年", "美国:国债收益率:10年"),
        ("中美利差_短端", "中债国债到期收益率:1年", "美国:国债收益率:1年"),
        ("非银溢价_GC007-DR007", "GC007:收盘价", "DR007"),
        ("存单-国开_1Y", "中国:发行利率(国有银行):同业存单:1年", "中债国开债到期收益率:1年"),
    ]
    for name, a, b in pairs:
        if a in C.columns and b in C.columns:
            spreads[name] = C[a] - C[b]

    # ---- F2 逐指标三件套：z252 + chg20 + chg60 ----
    # 利率/收益率/汇率/利差类用差值(bp)，指数/量/占比类用百分比变化
    RATE_PAT = ("收益率", "利率", "中间价", "汇率", "利差", "溢价", "互换",
                "存款准备金率", "PMI", "市盈率", "市净率")
    feat = pd.DataFrame(index=grid)

    def is_rate(name: str) -> bool:
        return any(p in name for p in RATE_PAT)

    def add_indicator(name: str, s: pd.Series) -> None:
        s = s.astype(float)
        feat[f"{name}__z"] = robust_z(s)
        if is_rate(name):
            feat[f"{name}__chg20"] = s.diff(20)
            feat[f"{name}__chg60"] = s.diff(60)
        else:
            feat[f"{name}__pct20"] = s.pct_change(
                20, fill_method=None).replace(
                [np.inf, -np.inf], np.nan).clip(-5, 5)
            feat[f"{name}__pct60"] = s.pct_change(
                60, fill_method=None).replace(
                [np.inf, -np.inf], np.nan).clip(-5, 5)

    for c in C.columns:
        if c == YIELD_COL or c in EVENT_COLS:
            continue
        add_indicator(c, C[c])
    for name, s in spreads.items():
        add_indicator(name, s)

    # 周频/月频合并（保守发布滞后）后同样三件套
    wk = merge_lagged(grid, weekly, PUBLISH_LAG_WEEKLY)
    mo = merge_lagged(grid, monthly, PUBLISH_LAG_MONTHLY)
    for c in wk.columns:
        add_indicator(f"周频.{c}", wk[c])
    for c in mo.columns:
        add_indicator(f"月频.{c}", mo[c])

    # ---- F3 10Y收益率技术面 ----
    t: dict[str, pd.Series] = {}
    t["y__mom5"] = y.diff(5)
    t["y__mom20"] = y.diff(20)
    t["y__mom60"] = y.diff(60)
    t["y__z60"] = (y - y.rolling(60, 20).mean()) / y.rolling(60, 20).std().clip(lower=1e-9)
    t["y__ma20_dev"] = y / y.rolling(20, 10).mean() - 1
    t["y__ma60_dev"] = y / y.rolling(60, 20).mean() - 1
    t["y__pctile252"] = y.rolling(252, 120).rank(pct=True)
    t["y__vol20"] = y.diff().rolling(20, 10).std()
    roll = y.rolling(20, 20)
    sd = roll.std().clip(lower=1e-9)
    t["y__bbpos20"] = (y - roll.mean()) / (2 * sd)
    d = y.diff()
    up = (d < 0).astype(int)                    # 下行=1 口径下的"连跌"
    t["y__streak_down"] = up.groupby((up != up.shift()).cumsum()).cumsum() * up
    dn = (d > 0).astype(int)
    t["y__streak_up"] = dn.groupby((dn != dn.shift()).cumsum()).cumsum() * dn
    t["y__dist_252high"] = y / y.rolling(252, 120).max() - 1
    t["y__dist_252low"] = y / y.rolling(252, 120).min() - 1
    gain = d.clip(lower=0).rolling(14, 10).mean()
    loss = (-d.clip(upper=0)).rolling(14, 10).mean()
    t["y__rsi14"] = 100 - 100 / (1 + gain / loss.clip(lower=1e-9))
    for k, s in t.items():
        feat[k] = s

    # ---- F4 降准事件 ----
    ev = daily_g[EVENT_COLS[0]] if EVENT_COLS[0] in daily_g.columns else None
    if ev is not None:
        is_ev = ev.notna().astype(int)
        feat["事件__降准公告"] = is_ev
        feat["事件__降准公告_5日内"] = is_ev.rolling(5, 1).max()
        feat["事件__准备金率_z"] = robust_z(ev.ffill())

    # ---- 标签（未来H个交易日方向，下行=1）----
    delta = y.shift(-horizon) - y
    # 注意：delta<0 对 NaN 为 False，必须显式置 NaN，否则尾部H行被错标为0
    label = (delta < 0).astype(float).mask(delta.isna())
    weight = np.where(delta.abs() * 100 < DEADZONE_BP, 0.3, 1.0)
    weight = np.where(delta.isna(), np.nan, weight)
    lab = pd.DataFrame({"label": label, "rate_delta_5d": delta,
                        "sample_weight": weight}, index=grid)

    # 去掉标签不可定义的尾部行（特征保留全量，训练时按标签过滤）
    feat.to_csv(OUT_DIR / f"43v2_features{sfx}.csv", encoding="utf-8-sig")
    lab.to_csv(OUT_DIR / f"43v2_labels_bin{sfx}.csv", encoding="utf-8-sig")

    n_valid = lab["label"].notna().sum()
    nan_row = feat.loc[lab["label"].notna().index].isna().mean().sort_values()
    lines = [
        "=" * 60,
        f"样本网格: {grid.min().date()} ~ {grid.max().date()}  共{len(grid)}个交易日",
        f"特征数: {feat.shape[1]}  (日频{daily.shape[1]}列中弃用死列{dead_cols})",
        f"标签: 下行占比 {label.sum() / n_valid:.4f}  死区样本 "
        f"{int((weight == 0.3).sum())}/{n_valid}",
        f"逐特征NaN率: min={nan_row.iloc[0]:.3f} 中位={nan_row.median():.3f} "
        f"max={nan_row.iloc[-1]:.3f}",
        f"NaN率>0.5的特征数: {(feat.isna().mean() > 0.5).sum()}",
        f"口径: 周频T+1、月频+15天发布滞后；z窗口252/min120；horizon={horizon}",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print(report)
    (OUT_DIR / f"43v2_build_meta{sfx}.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    build(parse_args().horizon)
