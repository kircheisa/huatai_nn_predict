# -*- coding: utf-8 -*-
"""
旧库风格特征族（全部因果，源自新基础数据）——复刻旧库被验证有用的特征思路，
但用滚动因果实现替代其全样本 STL 等前视实现。

七个特征族（前缀标识）：
  lr__    因果趋势回归族：trailing OLS 斜率/R²/偏离/偏离z（60/120/252窗）
          ——替代旧库泄漏的 STL 趋势族（lr_* 在旧库中因被 STL 压制重要性为0）
  corr__  滚动相关族：利率-股/美元/商品 相关及其突变（股债跷跷板等）
  vol__   波动率族：rate_std / vol_of_vol
  fund__  资金面族：隔夜回购占比、回购总量、IRS-DR007预期溢价、非银溢价5日变
  fut__   期货族：成交持仓比（投机度）、活跃-连续价差
  curve__ 曲线形态族：曲线均线差、凸性、税收/信用利差5日变
  fx__    美元/中美族：美债期限利差、利率-美元分化、即期-中间价贬值预期、中美利差5日变

输入: data_new/43v2cal_features.csv（291列）+ 国债10年基础数据.xlsx（原始列）
输出: data_new/43v2ext_features.csv（约330列）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "data_new" / "国债10年基础数据.xlsx"
NEW_DIR = BASE / "data_new"

YIELD_COL = "中债国债到期收益率:10年"


def load_daily() -> pd.DataFrame:
    raw = pd.ExcelFile(SRC).parse("滚动稳健标准化日频", header=None)
    hi = raw.index[raw[0] == "指标名称"][0]
    names = list(raw.iloc[hi, 1:])
    for i, n in enumerate(names):
        if names[: i + 1].count(n) > 1:
            names[i] = f"{n}#{i}"
    df = raw.iloc[hi + 1:].copy()
    df.columns = ["date"] + names
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df.apply(pd.to_numeric, errors="coerce").mask(lambda s: s == 0)


def robust_z(s: pd.Series, win: int = 252, minp: int = 120) -> pd.Series:
    med = s.rolling(win, min_periods=minp).median()
    q1 = s.rolling(win, min_periods=minp).quantile(0.25)
    q3 = s.rolling(win, min_periods=minp).quantile(0.75)
    return (s - med) / (q3 - q1).clip(lower=1e-9)


def lr_family(y: pd.Series, wins=(60, 120, 252)) -> dict[str, pd.Series]:
    """因果 trailing 回归闭式解：slope / r2 / dev / dev_z（仅用 t 及更早数据）"""
    out: dict[str, pd.Series] = {}
    v = y.values.astype(float)
    n = len(v)
    for w in wins:
        x = np.arange(w, dtype=float)
        wts = x - x.mean()
        sxx = float((wts ** 2).sum())
        slope = np.full(n, np.nan)
        r2 = np.full(n, np.nan)
        dev = np.full(n, np.nan)
        for t in range(w - 1, n):
            seg = v[t - w + 1: t + 1]
            if np.isnan(seg).any():
                continue
            sl = float(wts @ seg) / sxx
            ybar = seg.mean()
            sst = float(((seg - ybar) ** 2).sum())
            slope[t] = sl
            if sst > 1e-18:
                r2[t] = sl * sl * sxx / sst
            dev[t] = seg[-1] - (ybar + sl * (w - 1 - x.mean()))
        out[f"lr__slope_{w}"] = pd.Series(slope, index=y.index) * 252  # 年化
        out[f"lr__r2_{w}"] = pd.Series(r2, index=y.index)
        dev_s = pd.Series(dev, index=y.index)
        out[f"lr__dev_{w}"] = dev_s
        sd = dev_s.rolling(252, min_periods=120).std().clip(lower=1e-9)
        out[f"lr__dev_z_{w}"] = dev_s / sd
    return out


def main() -> None:
    d = load_daily()
    y = d[YIELD_COL].dropna()
    grid = y.index
    C = d.reindex(grid).ffill(limit=20)
    ext = pd.DataFrame(index=grid)

    # ---- lr__ 因果趋势族 ----
    for k, s in lr_family(y).items():
        ext[k] = s

    # ---- corr__ 滚动相关族 ----
    dy = y.diff()
    hs_ret = C["沪深300指数:涨跌幅"] / 100.0        # % -> 小数
    hs_ret = hs_ret.fillna(0.0)
    d_hs = (1 + hs_ret).cumprod().pct_change()
    d_usd = C["美元指数"].diff()
    d_nh = C["南华工业品指数"].pct_change(fill_method=None)
    ext["corr__stockbond_20"] = dy.rolling(20, 10).corr(d_hs)
    ext["corr__stockbond_60"] = dy.rolling(60, 20).corr(d_hs)
    ext["corr__dollar_20"] = dy.rolling(20, 10).corr(d_usd)
    ext["corr__dollar_60"] = dy.rolling(60, 20).corr(d_usd)
    ext["corr__commodity_60"] = dy.rolling(60, 20).corr(d_nh)
    ext["corr__dollar_break"] = ext["corr__dollar_60"] - ext["corr__dollar_20"]
    ext["corr__stockbond_break"] = (ext["corr__stockbond_20"] < 0).astype(int)

    # ---- vol__ 波动率族 ----
    rstd20 = dy.rolling(20, 10).std()
    ext["vol__rate_std_20"] = rstd20
    ext["vol__rate_std_60"] = dy.rolling(60, 20).std()
    ext["vol__vol_of_vol_60"] = rstd20.rolling(60, 20).std()
    ext["vol__vol_of_vol_120"] = rstd20.rolling(120, 60).std()

    # ---- fund__ 资金面族 ----
    r1, r7 = C["成交量:R001"], C["成交量:R007"]
    tot = r1 + r7
    ext["fund__overnight_ratio"] = r1 / tot.clip(lower=1e-9)
    ext["fund__overnight_ratio_chg5"] = ext["fund__overnight_ratio"].diff(5)
    ext["fund__repo_total_z"] = robust_z(tot)
    ext["fund__repo_total_pct20"] = tot.pct_change(20, fill_method=None).clip(-5, 5)
    irs_dr = C["利率互换:FR007:1年"] - C["DR007"]
    ext["fund__irs_minus_dr_z"] = robust_z(irs_dr)
    ext["fund__irs_minus_dr_chg20"] = irs_dr.diff(20)
    gc_dr = C["GC007:收盘价"] - C["DR007"]
    ext["fund__gc_dr_chg5"] = gc_dr.diff(5)

    # ---- fut__ 期货族 ----
    vol_f = C["期货成交量:国债期货:10年期"]
    oi_f = C["期货持仓量:国债期货:10年期"]
    ext["fut__vol_oi_ratio_z"] = robust_z(vol_f / oi_f.clip(lower=1e-9))
    cont, act = C["期货收盘价(连续):国债期货:10年期"], C["期货收盘价(活跃合约):国债期货:10年期"]
    ext["fut__active_minus_cont_z"] = robust_z(act - cont)

    # ---- curve__ 曲线形态族 ----
    s10y1y = C["中债国债到期收益率:10年"] - C["中债国债到期收益率:1年"]
    ext["curve__ma_gap"] = s10y1y.rolling(20, 10).mean() - s10y1y.rolling(60, 20).mean()
    curv = (C["中债国债到期收益率:10年"] - C["中债国债到期收益率:5年"]) - \
           (C["中债国债到期收益率:5年"] - C["中债国债到期收益率:1年"])
    ext["curve__curvature_z"] = robust_z(curv)
    tax5 = C["中债国开债到期收益率:5年"] - C["中债国债到期收益率:5年"]
    ext["curve__tax5_chg5"] = tax5.diff(5)
    cred1 = C["中债中短期票据到期收益率(AA+):1年"] - C["中债国债到期收益率:1年"]
    ext["curve__credit1_chg5"] = cred1.diff(5)

    # ---- fx__ 美元/中美族 ----
    us_term = C["美国:国债收益率:10年"] - C["美国:国债收益率:1年"]
    ext["fx__us_term_z"] = robust_z(us_term)
    ext["fx__rate_dollar_gap"] = robust_z(y) - robust_z(C["美元指数"])
    cfets_gap = (C["CFETS:即期汇率:美元兑人民币"] - C["中间价:美元兑人民币"]) * 1000
    ext["fx__cfets_mid_gap_z"] = robust_z(cfets_gap)
    cnus = C["中债国债到期收益率:10年"] - C["美国:国债收益率:10年"]
    ext["fx__cnus10_chg5"] = cnus.diff(5)

    # ---- 合并输出 ----
    cal = pd.read_csv(NEW_DIR / "43v2cal_features.csv", index_col=0,
                      parse_dates=True)
    out = cal.join(ext)
    nan_rate = ext.isna().mean().sort_values(ascending=False)
    print(f"新增 {ext.shape[1]} 个旧库风格特征 -> 合计 {out.shape[1]} 列")
    print(f"新增特征NaN率: max={nan_rate.iloc[0]:.2f}({nan_rate.index[0][:24]}) "
          f"中位={nan_rate.median():.3f}")
    out.to_csv(NEW_DIR / "43v2ext_features.csv", encoding="utf-8-sig")
    print("输出:", NEW_DIR / "43v2ext_features.csv")


if __name__ == "__main__":
    main()
