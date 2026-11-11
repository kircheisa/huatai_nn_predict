# -*- coding: utf-8 -*-
"""
华泰证券债券预测 —— 数据集预处理流水线（2026_ori_data_new 版）

================================================================================
基于 01_数据预处理.py（适配 2026_origin_data.xlsx 的 10-sheet 版本）改造，
适配最新的 4-sheet 原始数据 2026_ori_data_new.xlsx。

新数据结构与旧版的核心区别：
  1. Excel 从 10 sheet → 4 sheet（日频 / 日频+买卖行为 / 周频 / 月频）
     - 无季度 sheet（GDP、产能利用率不再有）
     - 无月度发布日期 sheet（需按各指标标准发布滞后做因果对齐）
     - 无特殊频率 sheet（逆回购利率/存准率并入日频，且为稀疏事件序列）
     - 无时代变量 sheet（领导人/总统时期独热编码不再有）
  2. 列名为 Wind 原始长名 → 需映射回旧管线（03_XGB训练.build_features /
     NN Top-30 特征）依赖的策展短名
  3. 日期倒序排列（最新在前）；缺失值以 0 填充（非交易日全为 0）
  4. 含 pandas 重复列后缀 ".1"（市净率/基建投资/社融/城投债/地方债）
  5. 10 年期国债收益率并入日频 sheet → rate.csv 直接从中提取（不再依赖
     单独的 rate CSV）

下游兼容性：
  输出列名与旧 model_new/data/data.csv、rate.csv 完全同构，可直接用于
  02_训练数据构建_二分类.py（--data-file/--rate-file/--output-dir 指向新目录）
  与 03_XGB训练_二分类.py（build_features）。

================================================================================
输入：
  data_ori/2026_ori_data_new.xlsx（4 sheet）

输出：
  model_new/data_new/data.csv  — 合并后的日频宽表（滚动鲁棒标准化后）
  model_new/data_new/rate.csv  — 10Y 利率 + 多窗口动量特征
================================================================================
"""

import os
import logging
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# =====================================================================
# 工具函数
# =====================================================================

def setup_logging():
    if LOGGER.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )


def rolling_robust_standardize(arr, window):
    """
    滚动鲁棒标准化（与旧版 01_数据预处理.py 完全一致）：
      标准化值 = (当前值 - 滚动窗口中位数) / 滚动窗口 IQR
    使用中位数/IQR 抵抗极端值；滚动窗口避免未来信息泄露。
    前 window-1 个位置为 NaN。
    """
    if window <= 0:
        raise ValueError("window 必须为正整数")
    series = pd.to_numeric(pd.Series(arr), errors="coerce")
    arr = series.to_numpy(dtype=np.float64)
    if arr.size < window:
        return np.full(arr.shape, np.nan, dtype=np.float64)

    window_view = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rolling_median = np.nanmedian(window_view, axis=1)
        rolling_q1 = np.nanpercentile(window_view, 25, axis=1)
        rolling_q3 = np.nanpercentile(window_view, 75, axis=1)
    rolling_iqr = rolling_q3 - rolling_q1
    rolling_iqr[rolling_iqr == 0] = np.nan
    standardized = (arr[window - 1:] - rolling_median) / rolling_iqr
    result = np.full(arr.shape, np.nan, dtype=np.float64)
    result[window - 1:] = standardized
    return result


def apply_rolling_robust_std(df_origin, window, date_col="指标名称"):
    """
    对 DataFrame 的每个数值列做滚动鲁棒标准化（按日期升序）。
    用 pandas rolling 实现（与旧版 sliding_window_view + np.nanpercentile
    结果一致：median=quantile(0.5)，插值方式均为 linear），内存与速度更优。
    """
    df = df_origin.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise ValueError("日期列无法解析")
    df = df.sort_values(by=date_col, ascending=True).set_index(date_col)
    result = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        med = s.rolling(window, min_periods=window).median()
        q1 = s.rolling(window, min_periods=window).quantile(0.25)
        q3 = s.rolling(window, min_periods=window).quantile(0.75)
        iqr = (q3 - q1).replace(0, np.nan)
        result[col] = (s - med) / iqr
    return result


# =====================================================================
# 列名映射：Wind 原始长名 → 旧管线策展短名
# （03_XGB训练_二分类.py 的 build_features / NN Top-30 特征依赖这些短名）
# =====================================================================

DAILY_RENAME_MAP = {
    # --- 利率期限结构 ---
    "中国:国债到期收益率:1年": "1年国债",
    "中国:国债到期收益率:3年": "3年国债",
    "中国:国债到期收益率:5年": "5年国债",
    "中国:国债到期收益率:10年": "10年国债",
    "中债国开债到期收益率:1年": "国开债1年",
    "中债国开债到期收益率:3年": "国开债3年",
    "中债国开债到期收益率:5年": "国开债5年",
    # --- 资金面 ---
    "DR001": "DR001",
    "DR007": "DR007",
    "GC001:收盘价": "GC001",
    "GC007:收盘价": "GC007",
    "中国:逆回购利率:7天": "逆回购利率:7天",
    "中国:法定存款准备金率:大型银行": "存款准备金率:大型银行",
    "成交量:R001": "成交量:R001",
    "成交量:R007": "成交量:R007",
    "中国:成交量:银行间质押式回购": "银行间质押式回购成交量",
    "利率互换:FR007:1年": "FR007:1年",
    "中国:发行利率(国有银行):同业存单:1年": "1Y存单发行利率",
    # --- 汇率/海外 ---
    "中间价:美元兑人民币": "中间价",
    "CFETS:即期汇率:美元兑人民币": "即期汇率",
    "美元指数": "美元指数",
    "美国:国债收益率:1年": "1年美债",
    "美国:国债收益率:2年": "2年美债",
    "美国:国债收益率:10年": "10年美债",
    # --- 权益市场 ---
    "沪深300:成交量": "沪深300成交量",
    "沪深300:成交金额": "沪深300成交金额",
    "沪深300指数:涨跌点数": "沪深300涨跌点数",
    "沪深300指数:涨跌幅": "沪深300涨跌幅",
    "市盈率:沪深300指数": "沪深300市盈率",
    "市净率:沪深300指数": "沪深300市净率",
    "静态市盈率:沪深300指数": "沪深300静态市盈率",
    "上证科创板50成份指数": "科创50",
    "恒生科技指数": "恒生科技",
    # --- 商品/高频产业 ---
    "期货收盘价(连续):COMEX黄金": "COMEX金价",
    "南华工业品指数": "南华工业品指数",
    "期货结算价(活跃合约):焦煤": "期货结算价(活跃合约):焦煤",
    "期货结算价(活跃合约):螺纹钢": "期货结算价(活跃合约):螺纹钢",
    "期货结算价(活跃合约):动力煤": "期货结算价(活跃合约):动力煤",
    "中国:水泥价格指数": "水泥价格指数",
    # 新版数据用期货结算价（2010 年起全历史）替换了原"现货领先价格:阴极铜"（仅 2023+），
    # 直接映射为下游 corr_rate_copper_* 特征依赖的列名
    "期货结算价(活跃合约):阴极铜": "阴极铜价格",
    "波罗的海干散货指数(BDI)": "波罗的海",
    "中国进口干散货运价指数:综合指数": "进口干散货运价指数",
    "全球:现货价:原油(英国布伦特Dtd)": "布伦特现货Dtd",
    "欧洲:现货离岸价:布伦特原油": "ICE布油",
    "中国:开工率:精对苯二甲酸": "精对苯二甲酸开工率",
    "中国:市场价:浮法平板玻璃(4.8/5mm)": "玻璃市场价",
    "中国:30大中城市:成交面积:商品房": "30城商品房成交面积",
    "中国:秦皇岛港:港口吞吐量:煤炭": "秦皇岛港口吞吐量",
    "中国:北京:地铁客运量": "北京地铁客运量",
    "中国:上海:地铁客运量": "上海地铁客运量",
    "中国:广州:地铁客运量": "广州地铁客运量",
    "中国:深圳:地铁客运量": "深圳地铁客运量",
    "中国:票房收入:电影": "票房收入",
    # --- 国债期货（沿用旧 data.csv 列名，注意"国债期货成交量：10年"为全角冒号）---
    "期货收盘价(连续):国债期货:10年期": "国债期货收盘价(连续):10年",
    "期货收盘价(活跃合约):国债期货:10年期": "国债期货收盘价(活跃合约):10年",
    "期货成交量:国债期货:10年期": "国债期货成交量：10年",
    "期货持仓量:国债期货:10年期": "国债期货持仓量:10年",
    "期货持仓量(连续):国债期货:10年期": "国债期货持仓量(连续):10年",
    # --- 债券发行/机构行为 ---
    "中国:债券净融资额:城投债": "城投债净融资",
    "中国:净融资:地方政府债": "地方政府债净融资",
    "净买入：券商自营": "净买入：券商自营",
    "净买入：保险": "净买入：保险",
    "净买入：基金": "净买入：基金",
    "净买入：理财": "净买入：理财",
    "净买入：货币基金": "净买入：货币基金",
}

WEEKLY_RENAME_MAP = {
    "中国:开工率:石油沥青装置": "石油沥青开工率",
    "中国:日均产量:粗钢": "粗钢日均产量",
    "中国:唐山:高炉开工率": "唐山高炉开工率",
    "中国:开工率:汽车轮胎(半钢胎)": "半钢胎开工率",
    "中国:100大中城市:成交土地占地面积": "100城成交土地占地面积",
    "中国:上海出口集装箱运价指数:综合指数": "上海出口集装箱运价指数",
}

MONTHLY_RENAME_MAP = {
    "中国:工业增加值:规模以上工业企业:累计同比": "工业增加值累计同比",
    "中国:社会消费品零售总额:累计同比": "社零累计同比",
    "中国:固定资产投资完成额:基础设施建设(Wind衍生计算):累计同比": "基建投资累计同比",
    "中国:CPI:当月同比": "CPI同比",
    "中国:PPI:当月同比": "PPI同比",
    "中国:社会融资规模存量:同比": "社融同比",
    "中国:M1:同比": "M1同比",
    "中国:M2:同比": "M2同比",
    "中国:制造业PMI": "制造业PMI",
    "中国:进口金额:当月同比": "进口金额同比",
    "中国:商品房销售面积:累计同比": "商品房销售面积累计同比",
    "中国:房屋销售价格指数:新建商品住宅:70个大中城市:当月同比": "70城房价同比",
}

# 月度指标 → 发布滞后（天数，自数据所属月末起算）
# 依据公开的统计发布惯例做保守因果对齐（新数据无发布日期表）：
#   PMI：统计局于当月最后一天发布 → 0
#   CPI/PPI：次月 9-10 日 → 10；  M1/M2/社融：次月 10-15 日 → 13
#   进出口：次月 7-13 日 → 10；   工业增加值/社零/固投/商品房销售：次月 15-17 日 → 16
#   70 城房价：次月中下旬 → 20
MONTHLY_PUBLISH_LAG_DAYS = {
    "制造业PMI": 0,
    "CPI同比": 10,
    "PPI同比": 10,
    "进口金额同比": 10,
    "社融同比": 13,
    "M1同比": 13,
    "M2同比": 13,
    "工业增加值累计同比": 16,
    "社零累计同比": 16,
    "基建投资累计同比": 16,
    "商品房销售面积累计同比": 16,
    "70城房价同比": 20,
}

# 交易日内 0 为合法值的列（收益率/涨跌点数），不做 0→NaN 转换
ZERO_IS_VALID_COLS = {"沪深300涨跌幅", "沪深300涨跌点数"}

# 政策利率水平列：稀疏事件序列（仅在调整窗口附近有值），需先 0→NaN→ffill
# 形成连续水平序列后再派生变动标记；水平列本身不进入特征表（与旧版一致）
POLICY_RATE_COLS = ["逆回购利率:7天", "存款准备金率:大型银行"]


# =====================================================================
# 日频数据清洗
# =====================================================================

def clean_daily(daily_raw, netbuy_raw=None):
    """
    清洗日频 sheet（新数据为 sheet0，买卖行为为 sheet1）：
      1. 日期升序；sheet1 的 5 列净买入并入（重叠日期数据已验证一致）
      2. 去掉 pandas 重复列后缀 ".1" 与不可用的乱码列（存准率变动日期）
      3. 列名映射为旧管线短名
      4. 交易日过滤（10Y 收益率非 0），0=缺失 → NaN（涨跌幅等除外）
      5. 政策利率稀疏事件列 → 0→NaN→ffill 形成连续水平序列
    """
    df = daily_raw.copy()
    df["指标名称"] = pd.to_datetime(df["指标名称"], errors="coerce")
    df = df.dropna(subset=["指标名称"]).sort_values("指标名称")

    # 并入 sheet1 的净买入列（sheet0 无这些列）
    if netbuy_raw is not None:
        nb = netbuy_raw.copy()
        nb["指标名称"] = pd.to_datetime(nb["指标名称"], errors="coerce")
        nb = nb.dropna(subset=["指标名称"])
        nb_cols = [c for c in nb.columns if c not in df.columns and c != "指标名称"]
        df = pd.merge(df, nb[["指标名称"] + nb_cols], on="指标名称", how="outer")

    # 去重复列（".1" 后缀经核验与原列完全一致）与乱码列
    drop_cols = [c for c in df.columns if str(c).endswith(".1")]
    drop_cols += [c for c in df.columns if "变动日期" in str(c)]
    df = df.drop(columns=drop_cols)

    # 列名映射
    df = df.rename(columns=DAILY_RENAME_MAP)

    # 交易日过滤：10Y 国债收益率非 0 的行
    if "10年国债" not in df.columns:
        raise ValueError("日频表缺少 10年国债收益率列")
    df = df[df["10年国债"].fillna(0) != 0].reset_index(drop=True)

    # 0=缺失 → NaN（涨跌幅/涨跌点数在交易日内可为 0，保留）
    for col in df.columns:
        if col == "指标名称" or col in ZERO_IS_VALID_COLS:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == 0, col] = np.nan

    # 政策利率稀疏事件列 → 连续水平序列（ffill 因果填充）
    for col in POLICY_RATE_COLS:
        if col in df.columns:
            df[col] = df[col].ffill()
    return df


def derive_raw_features(df):
    """
    在标准化之前、基于原始量纲派生模型需要的特征（旧 data.csv 中
    由外部整理好、而新数据必须自行构建的列）：
      - R001成交量占比
      - 1/3/5年信用利差（中短票AAA − 国债）+ AA+/企业债AAA 版本
      - 国开利差 1/3/5年（国开 − 国债，隐含税率代理）
      - 期限利差 10Y-1Y / 10Y-5Y / 5Y-1Y
      - 股债相对（沪深300盈利收益率 − 10Y 国债收益率，股债性价比）
      - 沪深300 指数水平（由涨跌幅复建，基准 1000，供动量/相关性特征）
      - A_modified / B_modified（逆回购利率、存准率变动事件标记）
    """
    # --- R001 成交量占比 ---
    if "成交量:R001" in df.columns and "银行间质押式回购成交量" in df.columns:
        total = df["银行间质押式回购成交量"].replace(0, np.nan)
        df["R001成交量占比"] = df["成交量:R001"] / total

    # --- 信用利差（中短票/企业债 − 国债，同期限，单位 %）---
    for tenor in ["1年", "3年", "5年"]:
        gov, mtk_aaap, mtk_aap, corp = (
            df.get(f"{tenor}国债"), df.get(f"中债中短期票据到期收益率(AAA):{tenor}"),
            df.get(f"中债中短期票据到期收益率(AA+):{tenor}"), df.get(f"中债企业债到期收益率(AAA):{tenor}"),
        )
        if gov is not None:
            if mtk_aaap is not None:
                df[f"{tenor}信用利差"] = mtk_aaap - gov  # 旧列名：1年/3年/5年信用利差
            if mtk_aap is not None:
                df[f"{tenor}信用利差AA+"] = mtk_aap - gov
            if corp is not None:
                df[f"{tenor}企业债利差"] = corp - gov

    # --- 国开-国债利差（隐含税率代理）---
    for tenor in ["1年", "3年", "5年"]:
        gov, cdb = df.get(f"{tenor}国债"), df.get(f"国开债{tenor}")
        if gov is not None and cdb is not None:
            df[f"国开国债利差{tenor}"] = cdb - gov

    # --- 期限利差 ---
    if "10年国债" in df.columns and "1年国债" in df.columns:
        df["期限利差10Y-1Y"] = df["10年国债"] - df["1年国债"]
    if "10年国债" in df.columns and "5年国债" in df.columns:
        df["期限利差10Y-5Y"] = df["10年国债"] - df["5年国债"]
    if "5年国债" in df.columns and "1年国债" in df.columns:
        df["期限利差5Y-1Y"] = df["5年国债"] - df["1年国债"]

    # --- 股债相对（股债性价比 ERP）：100/PE − 10Y 国债收益率 ---
    if "沪深300市盈率" in df.columns and "10年国债" in df.columns:
        pe = df["沪深300市盈率"].replace(0, np.nan)
        df["股债相对"] = 100.0 / pe - df["10年国债"]

    # --- 沪深300 指数水平复建（新数据无指数点位，仅涨跌幅；基准 1000）---
    if "沪深300涨跌幅" in df.columns:
        ret = df["沪深300涨跌幅"].fillna(0.0) / 100.0
        df["沪深300"] = (1.0 + ret).cumprod() * 1000.0

    # --- 政策利率变动事件标记（与旧版 A_modified/B_modified 语义一致）---
    if "逆回购利率:7天" in df.columns:
        changed = df["逆回购利率:7天"].diff().fillna(0.0) != 0.0
        df["A_modified"] = df["逆回购利率:7天"].where(changed, 0.0)
    else:
        df["A_modified"] = 0.0
    if "存款准备金率:大型银行" in df.columns:
        changed = df["存款准备金率:大型银行"].diff().fillna(0.0) != 0.0
        df["B_modified"] = df["存款准备金率:大型银行"].where(changed, 0.0)
    else:
        df["B_modified"] = 0.0

    # 政策利率水平列不进入特征表（与旧 data.csv 一致）
    df = df.drop(columns=[c for c in POLICY_RATE_COLS if c in df.columns])
    return df


def standardize_irregular(df, window):
    """
    对日期不规则、列内含 NaN 空洞的表（周频/月频）做滚动鲁棒标准化。
    每列仅在其自身有值的日期子序列上取滚动窗口（等价旧版对整周期表的
    window 取窗语义），否则含 NaN 的窗口会导致全 NaN。
    返回与输入同形状的标准化结果。
    """
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float64)
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < window:
            continue
        med = s.rolling(window, min_periods=window).median()
        q1 = s.rolling(window, min_periods=window).quantile(0.25)
        q3 = s.rolling(window, min_periods=window).quantile(0.75)
        iqr = (q3 - q1).replace(0, np.nan)
        out[col] = pd.Series(((s - med) / iqr).to_numpy(), index=s.index)
    return out


def leading_bfill(df):
    """先 ffill（因果）再对列首残留 NaN 做一次 bfill（等价旧版 daily.bfill，
    仅影响数据起始段，不引入区间内未来信息）"""
    return df.ffill().bfill()


# =====================================================================
# 周频 / 月频清洗
# =====================================================================

def clean_weekly(week_raw):
    df = week_raw.copy()
    df["指标名称"] = pd.to_datetime(df["指标名称"], errors="coerce")
    df = df.dropna(subset=["指标名称"]).sort_values("指标名称")
    df = df.drop(columns=[c for c in df.columns if str(c).endswith(".1")])
    df = df.rename(columns=WEEKLY_RENAME_MAP)
    for col in df.columns:
        if col == "指标名称":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == 0, col] = np.nan
    return df  # 不做 bfill：standardize_irregular 按各列自身日期取窗，列首 NaN → 99 哨兵


def clean_monthly(month_raw):
    df = month_raw.copy()
    df["指标名称"] = pd.to_datetime(df["指标名称"], errors="coerce")
    df = df.dropna(subset=["指标名称"]).sort_values("指标名称")
    df = df.drop(columns=[c for c in df.columns if str(c).endswith(".1")])
    df = df.rename(columns=MONTHLY_RENAME_MAP)
    for col in df.columns:
        if col == "指标名称":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == 0, col] = np.nan  # 同比指标精确 0 极罕见，视作缺失
    return df  # 不做 bfill：standardize_irregular 按各列自身日期取窗


def align_monthly_publish(monthly_clean):
    """按发布滞后将月度数据重排到"可得日期"索引上"""
    cols = {}
    for col in monthly_clean.columns:
        if col == "指标名称":
            continue
        lag = MONTHLY_PUBLISH_LAG_DAYS.get(col, 16)
        s = monthly_clean.set_index("指标名称")[col].dropna()
        if s.empty:
            continue
        avail_index = s.index + pd.Timedelta(days=lag)
        cols[col] = pd.Series(s.values, index=avail_index)
    out = pd.DataFrame(cols).sort_index()
    return out


# =====================================================================
# 利率动量特征（与旧版 preprocess_rate_data 一致，数据源改为新 Excel 的 10Y 列）
# =====================================================================

def build_rate_table(daily_clean):
    df = daily_clean[["指标名称", "10年国债"]].copy()
    df.columns = ["time", "rate"]
    df["rate"] = pd.to_numeric(df["rate"].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=["rate"]).sort_values("time").reset_index(drop=True)

    short_windows = [1, 7]
    long_windows = [30, 60, 180]
    short_win, short_min = 252, 66
    long_win, long_min = 756, 188

    def calc_momentum(window, rolling_win, min_periods):
        momentum = df["rate"].diff(window)
        med = momentum.rolling(rolling_win, min_periods=min_periods).median()
        q1 = momentum.rolling(rolling_win, min_periods=min_periods).quantile(0.25)
        q3 = momentum.rolling(rolling_win, min_periods=min_periods).quantile(0.75)
        iqr = q3 - q1
        return (momentum - med) / iqr

    for w in short_windows:
        df[f"momentum_{w}_robust_scaled"] = calc_momentum(w, short_win, short_min)
    for w in long_windows:
        df[f"momentum_{w}_robust_scaled"] = calc_momentum(w, long_win, long_min)
    return df


# =====================================================================
# 主流程
# =====================================================================

def preprocess_dataset(excel_path, output_data_path, output_rate_path):
    LOGGER.info("读取 Excel: %s", excel_path)
    daily_raw = pd.read_excel(excel_path, sheet_name=0, header=1)   # 日频（多 4 天）
    netbuy_raw = pd.read_excel(excel_path, sheet_name=1, header=0)  # 日频+买卖行为
    week_raw = pd.read_excel(excel_path, sheet_name=2, header=1)    # 周频
    month_raw = pd.read_excel(excel_path, sheet_name=3, header=1)   # 月频

    # ===== 步骤1：日频清洗 + 原始量纲派生特征 =====
    LOGGER.info("清洗日频数据...")
    daily = clean_daily(daily_raw, netbuy_raw)
    daily = derive_raw_features(daily)
    daily = daily.ffill()  # 因果填充；列首残留 NaN 走标准化后 99 哨兵（同旧版）
    LOGGER.info("日频: %d 个交易日, %d 列", len(daily), daily.shape[1] - 1)

    # ===== 步骤2：各频率滚动鲁棒标准化（日 252 / 周 52 / 月 12，同旧版）=====
    # A/B_modified 为事件标记（大量 0 值），滚动 IQR 恒为 0，与旧版一致地
    # 不做标准化、以原始值直接并入
    flag_cols = [c for c in ["A_modified", "B_modified"] if c in daily.columns]
    daily_std = apply_rolling_robust_std(daily.drop(columns=flag_cols), 252)
    for c in flag_cols:
        daily_std[c] = daily[c].to_numpy()
    week_std = standardize_irregular(clean_weekly(week_raw).set_index("指标名称"), 52)
    monthly_avail = align_monthly_publish(clean_monthly(month_raw))
    month_std = standardize_irregular(monthly_avail, 12)

    # ===== 步骤3：日频 ← 周频（backward asof：取最近已观测周值，无前视）=====
    merge1 = pd.merge_asof(
        daily_std.sort_index(), week_std.sort_index(),
        left_index=True, right_index=True, direction="backward"
    )
    # ===== 步骤4：日频 ← 月频（backward asof：取最近已发布月值，无前视）=====
    merge2 = pd.merge_asof(
        merge1.sort_index(), month_std.sort_index(),
        left_index=True, right_index=True, direction="backward"
    )

    # ===== 步骤5：缺失处理（同旧版：ffill + 余缺填 99.00）=====
    merged = merge2.ffill().fillna(99.00)
    merged["date_list"] = merged.index.to_series().apply(
        lambda x: [x.year, x.month, x.day]
    )
    merged.index.name = None

    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    merged.to_csv(output_data_path)
    LOGGER.info("输出主数据: %s (%d 行, %d 列)", output_data_path, merged.shape[0], merged.shape[1])

    # ===== 步骤6：10Y 利率 + 动量特征 =====
    rate_df = build_rate_table(daily)
    rate_df.to_csv(output_rate_path, index=False)
    LOGGER.info("输出利率数据: %s (%d 行, %s -> %s)",
                output_rate_path, len(rate_df),
                rate_df["time"].min().date(), rate_df["time"].max().date())
    return merged, rate_df


# =====================================================================
# 下游兼容性自检：03_XGB训练.build_features 依赖的列名
# =====================================================================

REQUIRED_BY_BUILD_FEATURES = [
    "1年信用利差", "3年信用利差", "5年信用利差",
    "美元指数", "即期汇率", "中间价", "股债相对",
    "成交量:R001", "R001成交量占比", "R001", "R007",
    "DR001", "DR007", "国债期货持仓量:10年", "国债期货收盘价(活跃合约):10年",
    "10年美债", "2年美债", "标普500", "沪深300", "恒生科技",
    "期货结算价(活跃合约):螺纹钢", "水泥价格指数", "阴极铜价格", "南华工业品指数",
    "净买入：基金", "净买入：券商自营", "净买入：保险", "净买入：理财", "净买入：货币基金",
]


def compat_check(merged):
    missing = [c for c in REQUIRED_BY_BUILD_FEATURES if c not in merged.columns]
    LOGGER.info("=== 下游兼容性自检 ===")
    LOGGER.info("总列数: %d", merged.shape[1])
    if missing:
        LOGGER.warning("缺失列（对应 build_features 分支将跳过）: %s", missing)
    else:
        LOGGER.info("build_features 依赖列全部齐备")


def main():
    setup_logging()
    base_dir = os.path.dirname(os.path.abspath(__file__))          # model_new/
    project_root = os.path.dirname(base_dir)
    excel_path = os.path.join(project_root, "data_ori", "2026_ori_data_new.xlsx")
    out_dir = os.path.join(base_dir, "data_new")
    output_data = os.path.join(out_dir, "data.csv")
    output_rate = os.path.join(out_dir, "rate.csv")

    try:
        merged, rate_df = preprocess_dataset(excel_path, output_data, output_rate)
        compat_check(merged)
        LOGGER.info("预处理完成")
    except Exception:
        LOGGER.exception("预处理失败")
        raise


if __name__ == "__main__":
    main()
