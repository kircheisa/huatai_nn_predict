# -*- coding: utf-8 -*-
"""
新版原始数据（data_ori/2026_ori_data_new.xlsx）特征构建

数据源 4 sheet（Wind 导出，缺失以 0 占位，买卖行为列为空白）：
  滚动稳健标准化日频+买卖行为  79列，至2026-11-06（主表：72基础列+5机构净买入）
  滚动稳健标准化日频           72列，至2026-11-10（用于延长尾部交易日网格）
  滚动稳健标准化周频            6列，至2026-11-08
  滚动稳健标准化月频           14列，至2026-10-31（含2010前历史，作热身段）

与 20_新基础数据_特征构建.py（43v2，源=国债10年基础数据.xlsx）的关系：
  - 特征族 F1~F4 口径一致（robust z 252/min120、chg20/60、发布滞后、
    死区2bp权重0.3、下行=1），仅原始列名随新版数据变化：
      中债国债到期收益率:*年  ->  中国:国债到期收益率:*年（标签源同步更换，
      旧库中两列相关0.9996）
      准备金率事件列(变动公告日期)缺失 -> 用 :变动日期 列代替（36次变动，
      有效日≥公告日，保守无前视）
  - 新版新增原始列全部纳入：企业债AAA 1/3/5Y、阴极铜期货、静态市盈率、
    标普500、城投债/地方债净融资
  - 新增 F5 机构行为/资金流族：5类现券净买入（券商自营/保险/基金/理财/货基，
    2022-01起）+ 2类债券净融资，构造 z / 5日累计z / 20日累计z / 5日累计变化
  - 周频/月频完全重复列在合并前去重（43v2 中以 #编号 保留的重复特征不再生成）

输出（data_new/，43v3 前缀，schema 与 43v2 完全兼容）：
  43v3_features{,_h10}.csv      特征矩阵（index=交易日）
  43v3_labels_bin{,_h10}.csv    label(下行=1) + rate_delta_5d + sample_weight
  43v3_build_meta{,_h10}.txt    构建摘要

用法：
  python 30_新版原始数据_特征构建.py            # h5
  python 30_新版原始数据_特征构建.py --horizon 10

下游（无需改代码/最小改动）：
  XGB: 21_新基础数据_XGB训练.py --features-file 43v3_features.csv \
       --labels-file 43v3_labels_bin.csv --out-prefix xgbv3
  NN : 17_BiGRU训练_新基础数据_二分类.py --features-file 43v3_features.csv \
       --labels-file 43v3_labels_bin.csv --folds-file xgbv3_fold_metrics.csv \
       --importance-file xgbv3_feature_importance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "data_ori" / "2026_ori_data_new.xlsx"
OUT_DIR = BASE / "data_new"
PREFIX = "43v3"

SHEET_DAILY_B = "滚动稳健标准化日频+买卖行为"   # 主表（含机构净买入）
SHEET_DAILY = "滚动稳健标准化日频"              # 尾部延长用
SHEET_WEEKLY = "滚动稳健标准化周频"
SHEET_MONTHLY = "滚动稳健标准化月频"

YIELD_COL = "中国:国债到期收益率:10年"          # 新版列名（旧版为中债国债到期收益率:10年）
DEADZONE_BP = 2.0                              # |Δ|<2bp 死区样本权重0.3
PUBLISH_LAG_WEEKLY = 1                         # 周频发布滞后（自然日）
PUBLISH_LAG_MONTHLY = 15                       # 月频发布滞后（自然日）
Z_WIN = 252                                    # robust z 窗口（交易日）
Z_MIN = 120                                    # robust z 最小观察数

# 事件列（0 是真实信息=无变动，不做 0->NaN）；新版仅剩":变动日期"列
EVENT_COLS = ["中国:人民币存款准备金率:大型存款类金融机构:变动日期"]

# 直接弃用的列：阶梯常数（IQR=0）
DROP_COLS = {"中国:法定存款准备金率:大型银行"}

# 机构买卖行为 + 债券净融资（流量类，亿元，可正可负）：
# 不走 F2 三件套（pct 在正负切换时失真），统一进 F5 流量族
NETBUY_COLS = ["净买入：券商自营", "净买入：保险", "净买入：基金",
               "净买入：理财", "净买入：货币基金"]
FINANCE_COLS = ["中国:债券净融资额:城投债", "中国:净融资:地方政府债"]
FLOW_COLS = NETBUY_COLS + FINANCE_COLS


def load_sheet(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """定位'指标名称'行，返回数值矩阵（index=日期升序）"""
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


def drop_exact_dups(d: pd.DataFrame) -> pd.DataFrame:
    """值完全相同的重复列，仅保留首列（比 #编号 保留重复更干净）"""
    return d.loc[:, ~d.T.duplicated()]


def combine_daily(behavior: pd.DataFrame, plain: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """主表=日频+买卖行为（至11-06）；用日频表补齐更晚交易日的基础列。

    两表72个基础列同名同源，重叠日期数值应一致（此处校验并报告），
    净买入列仅主表有，尾部自动为 NaN。
    """
    shared = [c for c in behavior.columns if c in plain.columns]
    both = behavior.index.intersection(plain.index)
    diff_n, diff_max = 0, 0.0
    for c in shared:
        a, b = behavior.loc[both, c], plain.loc[both, c]
        m = a.notna() & b.notna() & ((a - b).abs() > 1e-9)
        diff_n += int(m.sum())
        if m.any():
            diff_max = max(diff_max, float((a - b).abs().max()))
    combined = behavior.combine_first(plain[shared])
    info = {"shared_cols": len(shared), "overlap_rows": len(both),
            "value_mismatch_cells": diff_n, "max_abs_diff": diff_max,
            "behavior_last": behavior.index.max().date().isoformat(),
            "plain_last": plain.index.max().date().isoformat(),
            "combined_last": combined.index.max().date().isoformat()}
    return combined, info


def zero_to_nan(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """0 -> NaN（事件列除外）；随后删除全空列与完全重复列"""
    d = d.copy()
    d = d.drop(columns=[c for c in DROP_COLS if c in d.columns])
    for c in EVENT_COLS:
        if c in d.columns:
            d[c] = d[c].fillna(0.0)
    d = d.mask(d == 0)
    dead = [c for c in d.columns if d[c].notna().sum() == 0]
    d = d.drop(columns=dead)
    d = drop_exact_dups(d)
    return d, dead


def merge_lagged(grid_idx: pd.DatetimeIndex, low: pd.DataFrame,
                 lag_days: int) -> pd.DataFrame:
    """低频表按'统计日+lag'发布，merge_asof 向后对齐到交易日（严禁当日可用）"""
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
                   help="预测周期（交易日）；h5输出43v3_*，其余加_h{H}后缀")
    return p.parse_args()


def build(horizon: int) -> None:
    sfx = "" if horizon == 5 else f"_h{horizon}"
    xl = pd.ExcelFile(SRC)

    # ---- 日频：主表(买卖行为) + 日频表延长尾部 ----
    # 各自先去重（城投债/地方债/市净率在主表中重复出现），否则尾部拼接后
    # 重复列与基础列在尾部数值不同，精确去重失效
    behavior = drop_exact_dups(load_sheet(xl, SHEET_DAILY_B))
    plain = drop_exact_dups(load_sheet(xl, SHEET_DAILY))
    combined, comb_info = combine_daily(behavior, plain)
    daily, dead_cols = zero_to_nan(combined)
    weekly = drop_exact_dups(load_sheet(xl, SHEET_WEEKLY).mask(lambda s: s == 0))
    monthly = drop_exact_dups(load_sheet(xl, SHEET_MONTHLY).mask(lambda s: s == 0))
    flow_cols_have = [c for c in FLOW_COLS if c in daily.columns]

    # ---- 交易日网格 & 标签源（收益率非零的日期）----
    y = daily[YIELD_COL].dropna()
    grid = y.index
    # 稀疏发布的价格类日频列（玻璃/存单等）在交易日网格上ffill（限20日防空段僵尸值）
    daily_g = daily.reindex(grid).ffill(limit=20)
    y.name = "yield10y"

    feat = pd.DataFrame(index=grid)

    # ---- F1 利差/价差（日频水平就地相减；曲线列用新版列名）----
    C = daily_g
    spreads: dict[str, pd.Series] = {}
    pairs = [
        ("期限利差_10Y-1Y", YIELD_COL, "中国:国债到期收益率:1年"),
        ("期限利差_10Y-3Y", YIELD_COL, "中国:国债到期收益率:3年"),
        ("期限利差_10Y-5Y", YIELD_COL, "中国:国债到期收益率:5年"),
        ("期限利差_5Y-1Y", "中国:国债到期收益率:5年", "中国:国债到期收益率:1年"),
        ("期限利差_3Y-1Y", "中国:国债到期收益率:3年", "中国:国债到期收益率:1年"),
        ("税收利差_国开5Y-国债5Y", "中债国开债到期收益率:5年", "中国:国债到期收益率:5年"),
        ("税收利差_国开3Y-国债3Y", "中债国开债到期收益率:3年", "中国:国债到期收益率:3年"),
        ("信用利差_AA+_3Y-国债3Y", "中债中短期票据到期收益率(AA+):3年", "中国:国债到期收益率:3年"),
        ("信用利差_AA+_1Y-国债1Y", "中债中短期票据到期收益率(AA+):1年", "中国:国债到期收益率:1年"),
        ("信用利差_AAA_3Y-国债3Y", "中债中短期票据到期收益率(AAA):3年", "中国:国债到期收益率:3年"),
        ("等级利差_AA+-AAA_3Y", "中债中短期票据到期收益率(AA+):3年", "中债中短期票据到期收益率(AAA):3年"),
        ("等级利差_AA+-AAA_1Y", "中债中短期票据到期收益率(AA+):1年", "中债中短期票据到期收益率(AAA):1年"),
        ("信用利差_企业AAA_3Y-国债3Y", "中债企业债到期收益率(AAA):3年", "中国:国债到期收益率:3年"),  # 新版列
        ("信用利差_企业AAA_5Y-国债5Y", "中债企业债到期收益率(AAA):5年", "中国:国债到期收益率:5年"),  # 新版列
        ("中美利差_10Y", YIELD_COL, "美国:国债收益率:10年"),
        ("中美利差_短端", "中国:国债到期收益率:1年", "美国:国债收益率:1年"),
        ("非银溢价_GC007-DR007", "GC007:收盘价", "DR007"),
        ("存单-国开_1Y", "中国:发行利率(国有银行):同业存单:1年", "中债国开债到期收益率:1年"),
    ]
    spreads_dropped = []
    for name, a, b in pairs:
        if a in C.columns and b in C.columns:
            spreads[name] = C[a] - C[b]
        else:
            spreads_dropped.append(name)

    # ---- F2 逐指标三件套：z252 + chg20 + chg60 ----
    # 利率/收益率/汇率/利差类用差值(bp)，指数/量/占比类用百分比变化
    RATE_PAT = ("收益率", "利率", "中间价", "汇率", "利差", "溢价", "互换",
                "存款准备金率", "PMI", "市盈率", "市净率")

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
        if c == YIELD_COL or c in EVENT_COLS or c in flow_cols_have:
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

    # ---- F4 准备金率事件（新版仅":变动日期"列，有效日即事件日）----
    # 注意: 事件列取未ffill的原始列（daily_g 的 ffill(limit=20) 会把事件
    # 涂抹成约20个交易日——43v2 的已知缺陷，此处修复）；事件落在非交易日
    # 时顺延到下一交易日（周末公告周一才可知，无前视）
    ev_col = EVENT_COLS[0] if EVENT_COLS[0] in daily.columns else None
    if ev_col is not None:
        ev_dates = daily.index[daily[ev_col].notna()]
        is_ev = pd.Series(0, index=grid, dtype=int)
        # 事件日 -> 其后首个交易日
        ev_on_grid = pd.Index(
            [grid[grid.searchsorted(d)] for d in ev_dates
             if d <= grid[-1]])
        is_ev.loc[is_ev.index.intersection(ev_on_grid)] = 1
        feat["事件__准备金变动"] = is_ev
        feat["事件__准备金变动_5日内"] = is_ev.rolling(5, 1).max()
        feat["事件__准备金率_z"] = robust_z(
            daily[ev_col].reindex(grid).ffill())

    # ---- F5 机构买卖行为/资金流（流量族，可正可负，禁用pct）----
    flow_note = {}
    for c in flow_cols_have:
        s = C[c].astype(float)
        sum5 = s.rolling(5, min_periods=1).sum()
        sum20 = s.rolling(20, min_periods=1).sum()
        feat[f"{c}__z"] = robust_z(s)
        feat[f"{c}__sum5_z"] = robust_z(sum5)
        feat[f"{c}__sum20_z"] = robust_z(sum20)
        feat[f"{c}__sum5_chg5"] = sum5.diff(5)
        first = s.dropna().index.min()
        if first is not pd.NaT:
            flow_note[c] = f"{first.date()} 起 {int(s.notna().sum())} 个观测"

    # ---- 标签（未来H个交易日方向，下行=1）----
    delta = y.shift(-horizon) - y
    # 注意：delta<0 对 NaN 为 False，必须显式置 NaN，否则尾部H行被错标为0
    label = (delta < 0).astype(float).mask(delta.isna())
    weight = np.where(delta.abs() * 100 < DEADZONE_BP, 0.3, 1.0)
    weight = np.where(delta.isna(), np.nan, weight)
    lab = pd.DataFrame({"label": label, "rate_delta_5d": delta,
                        "sample_weight": weight}, index=grid)

    feat.to_csv(OUT_DIR / f"{PREFIX}_features{sfx}.csv", encoding="utf-8-sig")
    lab.to_csv(OUT_DIR / f"{PREFIX}_labels_bin{sfx}.csv", encoding="utf-8-sig")

    # ---- 构建摘要 ----
    n_valid = lab["label"].notna().sum()
    nan_row = feat.loc[lab["label"].notna().index].isna().mean().sort_values()
    fam_cnt = {
        "F1利差": len(spreads) * 3,
        "F2三件套(日频)": sum(3 for c in C.columns
                           if c not in [YIELD_COL] and c not in EVENT_COLS
                           and c not in flow_cols_have),
        "F2三件套(周频)": len(wk.columns) * 3,
        "F2三件套(月频)": len(mo.columns) * 3,
        "F3技术面": len(t),
        "F4事件": 3 if ev_col is not None else 0,
        "F5流量族": len(flow_cols_have) * 4,
    }
    lines = [
        "=" * 60,
        f"新版原始数据特征构建 [{PREFIX}{sfx}]  horizon={horizon}",
        f"源: data_ori/2026_ori_data_new.xlsx",
        f"  日频主表(买卖行为) {behavior.shape[1]}列 至{comb_info['behavior_last']}"
        f" | 日频表 {plain.shape[1]}列 至{comb_info['plain_last']}",
        f"  两表重叠: {comb_info['overlap_rows']}行×{comb_info['shared_cols']}列"
        f" 不一致单元格 {comb_info['value_mismatch_cells']}个"
        f" (max|Δ|={comb_info['max_abs_diff']:.4g})",
        f"样本网格: {grid.min().date()} ~ {grid.max().date()}  共{len(grid)}个交易日",
        f"标签源: {YIELD_COL}（旧版为中债列，相关0.9996）",
        f"特征数: {feat.shape[1]}  按族: {fam_cnt}",
        f"  (日频{daily.shape[1]}列中弃用死列{dead_cols}; "
        f"利差缺源丢弃{spreads_dropped})",
        f"标签: 下行占比 {label.sum() / n_valid:.4f}  死区样本 "
        f"{int((weight == 0.3).sum())}/{n_valid}",
        f"逐特征NaN率: min={nan_row.iloc[0]:.3f} 中位={nan_row.median():.3f} "
        f"max={nan_row.iloc[-1]:.3f}",
        f"NaN率>0.5的特征数: {(feat.isna().mean() > 0.5).sum()}",
        "F5流量族覆盖: " + "; ".join(f"{k}: {v}" for k, v in flow_note.items()),
        "口径: 周频T+1、月频+15天发布滞后；z窗口252/min120；"
        "事件列用':变动日期'（旧版为'(变动公告日期)'，缺失）；"
        "事件哑变量修复43v2的ffill涂抹缺陷（仅事件日及其后5日内标记）",
        "对比43v2: 曲线/标签列改名、新增企业债AAA/阴极铜期货/静态市盈率/标普500/"
        "城投债+地方债净融资、新增5类机构净买入(F5)、月频周频重复列去重",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print(report)
    (OUT_DIR / f"{PREFIX}_build_meta{sfx}.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    build(parse_args().horizon)
