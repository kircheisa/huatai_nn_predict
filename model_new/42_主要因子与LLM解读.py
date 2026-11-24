# -*- coding: utf-8 -*-
"""
主要因子输出与 LLM 解读：对最新发出日的四周期预测做逐因子贡献分解

方法:
  1. 加载当日缓存 XGB 模型（40 脚本产出 models/{发出日}/h*_xgb.ubj）
  2. 对发出日特征行做 pred_contribs（TreeSHAP）：每因子对该次预测
     "下行概率"的边际贡献，正=推向下行，负=推向上行
  3. 每周期取 |贡献| Top10 因子，附当前值与近20交易日变化
  4. GLM-5.3-Flash 依据因子表解释"因子变化与预测结果的关系"

输出: data_new/out/{发出日}主要因子.csv
      data_new/out/主要因子LLM解读.md
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data_new"
OUT = DATA / "out"
API = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "glm-5.3-flash"
TOPN = 10
KEY_FILE = DATA / "llm_key.txt"


def get_key() -> str:
    import os
    key = os.environ.get("ZHIPU_API_KEY", "").strip()
    return key or (KEY_FILE.read_text(encoding="utf-8").strip()
                   if KEY_FILE.exists() else "")


def main() -> None:
    led = pd.read_csv(DATA / "预测记录.csv", dtype={"发出日": str})
    asof = pd.Timestamp(led["发出日"].max())
    mdir = DATA / "models" / asof.strftime("%Y%m%d")
    assert mdir.exists(), f"未找到模型缓存 {mdir}，请先运行 40"

    feats = pd.read_csv(DATA / "43v4nb_features.csv", index_col=0,
                        parse_dates=True)
    x_row = feats.loc[[asof]]
    pos = feats.index.get_loc(asof)
    print(f"发出日 {asof.date()}  因子分解 Top{TOPN}/周期")

    sig = led[led["发出日"] == asof.strftime("%Y-%m-%d")].assign(
        _h=lambda d: d["周期"].str.lstrip("h").astype(int)).sort_values("_h")

    FACTOR_CN = {
        "蝶式1-3-5": "国债曲线弯曲度(1/3/5年组合利差)",
        "蝶式3-5-10": "国债曲线弯曲度(3/5/10年组合利差)",
        "中国:水泥价格指数": "水泥价格(基建地产需求晴雨表)",
        "中国:开工率:精对苯二甲酸": "PTA化纤原料开工率(化工景气度)",
        "期货持仓量:国债期货:10年期": "10年国债期货持仓量(市场参与热度)",
        "期货结算价(活跃合约):焦煤": "焦煤期货",
        "美国:标准普尔500指数": "美股标普500指数",
        "中国:票房收入:电影": "电影票房(居民消费热度)",
        "趋势__dev_z_252": "收益率偏离一年趋势的程度",
        "趋势__slope": "收益率一年趋势线斜率",
        "联动__corr_利率铜": "利率与铜价的联动性",
        "联动__corr_利率美元": "利率与美元指数的联动性",
        "联动__corr_利率标普": "利率与美股的联动性",
        "联动__corr_break_dollar": "美元联动的突然变化",
        "曲线__斜率": "收益率曲线斜率(10年减1年)",
        "曲线__蝶式": "国债收益率曲线弯曲度",
        "中美利差_10Y": "中美10年期利差",
        "中美利差_短端": "中美1年期利差",
        "中债国开债到期收益率:3年": "3年国开债收益率",
        "中债国开债到期收益率:1年": "1年国开债收益率",
        "中债国开债到期收益率:5年": "5年国开债收益率",
        "美国:国债收益率:10年": "美债10年收益率",
        "美国:国债收益率:1年": "美债1年收益率",
        "美国:国债收益率:2年": "美债2年收益率",
        "月频.中国:制造业PMI": "制造业PMI(景气调查)",
        "月频.中国:M1:同比": "M1货币增速(企业活期资金)",
        "月频.中国:PPI:当月同比": "PPI工业品通胀",
        "月频.中国:CPI:当月同比": "CPI消费通胀",
        "月频.中国:房屋销售价格指数:新建商品住宅:70个大中城市:当月同比": "70城房价同比",
        "月频.中国:商品房销售面积:累计同比": "商品房销售面积累计同比",
        "中国进口干散货运价指数:综合指数": "干散货运价指数(大宗贸易热度)",
        "中国:30大中城市:成交面积:商品房": "30城商品房成交面积",
        "周频.中国:上海出口集装箱运价指数:综合指数": "上海出口集装箱运价(出口景气度)",
        "周频.中国:开工率:汽车轮胎(半钢胎)": "汽车轮胎开工率(汽车生产景气)",
        "周频.中国:开工率:石油沥青装置": "石油沥青开工率(道路建设景气)",
        "周频.中国:唐山:高炉开工率": "唐山高炉开工率(钢铁生产景气)",
        "周频.中国:日均产量:粗钢": "粗钢日均产量(工业生产热度)",
        "中国:100大中城市:成交土地占地面积": "百城土地成交面积(地产拿地热度)",
        "全球:现货价:原油(英国布伦特Dtd)": "布伦特原油价格",
        "期限利差_3Y-1Y": "3年-1年期国债利差",
        "期限利差_10Y-1Y": "10年-1年期国债利差",
        "期限利差_10Y-3Y": "10年-3年期国债利差",
        "期限利差_10Y-5Y": "10年-5年期国债利差",
        "期限利差_5Y-1Y": "5年-1年期国债利差",
        "信用利差_AA+_3Y-国债3Y": "3年AA+信用利差",
        "信用利差_AA+_1Y-国债1Y": "1年AA+信用利差",
        "信用利差_AAA_3Y-国债3Y": "3年AAA信用利差",
        "等级利差_AA+-AAA_3Y": "3年AA+与AAA等级利差",
        "等级利差_AA+-AAA_1Y": "1年AA+与AAA等级利差",
        "税收利差_国开5Y-国债5Y": "5年税收利差(国开-国债)",
        "税收利差_国开3Y-国债3Y": "3年税收利差(国开-国债)",
        "非银溢价_GC007-DR007": "交易所与银行间回购利差(非银融资压力)",
        "存单-国开_1Y": "同业存单与国开债1年利差(银行负债压力)",
        "中国:国债到期收益率:10年": "10年国债收益率",
        "中国:国债到期收益率:1年": "1年国债收益率",
        "中国:国债到期收益率:3年": "3年国债收益率",
        "中国:国债到期收益率:5年": "5年国债收益率",
        "中债中短期票据到期收益率(AA+):3年": "3年AA+信用债收益率",
        "中债中短期票据到期收益率(AA+):1年": "1年AA+信用债收益率",
        "中债中短期票据到期收益率(AA+):5年": "5年AA+信用债收益率",
        "中债中短期票据到期收益率(AAA):3年": "3年AAA信用债收益率",
        "中债中短期票据到期收益率(AAA):1年": "1年AAA信用债收益率",
        "中债中短期票据到期收益率(AAA):5年": "5年AAA信用债收益率",
        "中债企业债到期收益率(AAA):3年": "3年AAA企业债收益率",
        "中债企业债到期收益率(AAA):1年": "1年AAA企业债收益率",
        "中债企业债到期收益率(AAA):5年": "5年AAA企业债收益率",
        "中国:逆回购利率:7天": "央行7天逆回购利率",
        "中国:净融资:地方政府债": "地方政府债净融资",
        "中国:债券净融资额:城投债": "城投债净融资(融资供给压力)",
        "中国:发行利率(国有银行):同业存单:1年": "1年同业存单发行利率(银行负债成本)",
        "资金__IRS1Y-DR007": "利率互换与资金利率之差(宽松预期)",
        "资金__DR007-DR001": "7天与隔夜资金利率之差",
        "资金__隔夜占比": "隔夜资金成交占比(加杠杆情绪)",
        "资金__DR001波动20": "隔夜利率波动(20日)",
        "汇率__逆周期强度": "人民币中间价与即期价差(央行引导力度)",
        "股债__swing20": "股债相对回报(20日)",
        "股债__股20日回报": "股市20日回报",
        "股债__股60日回报": "股市60日回报",
        "股债__corr_利率权益": "利率与股市涨跌的联动性",
        "波动__rate_std_60": "利率波动率(60日)",
        "波动__vol_of_vol_60": "利率波动率的变化幅度",
        "波动__rate_std_ratio": "短期与长期波动率之比",
        "机构__交易盘净买入": "交易型机构(基金券商理财)净买入",
        "机构__配置盘净买入": "配置型机构(保险货基)净买入",
        "机构__交易配置差额": "交易盘与配置盘买入之差",
        "机构__分歧度": "各机构买卖分歧度",
        "机构__corr_基金净买入利率": "基金买入力度与利率变动的联动",
        "事件__准备金变动": "存款准备金率调整",
        "日历__距月末天数": "距月末天数(月末资金面效应)",
        "日历__距季末天数": "距季末天数(季末考核效应)",
        "日历__月末3日内": "月末最后3天",
        "日历__季末3日内": "季末最后3天",
        "净买入：券商自营": "券商自营净买入",
        "净买入：保险": "保险净买入",
        "净买入：基金": "基金净买入",
        "净买入：理财": "银行理财净买入",
        "净买入：货币基金": "货币基金净买入",
        # ---- 收益率自身技术因子（y__ 家族）----
        "y__bbpos20": "收益率在20日布林带中的位置(0=中轨,±1=上下轨,极端=超涨超跌)",
        "y__mom5": "收益率5日动量", "y__mom20": "收益率20日动量",
        "y__mom60": "收益率60日动量",
        "y__z60": "收益率偏离60日均值的程度",
        "y__ma20_dev": "收益率偏离20日均线的幅度",
        "y__ma60_dev": "收益率偏离60日均线的幅度",
        "y__pctile252": "收益率在一年中的分位数(0=最低,1=最高)",
        "y__vol20": "收益率波动率(20日)",
        "y__streak_down": "连续下跌天数", "y__streak_up": "连续上涨天数",
        "y__dist_252high": "距一年高点的距离", "y__dist_252low": "距一年低点的距离",
        "y__rsi14": "RSI相对强弱指标(14日,>70超买,<30超卖)",
        # ---- 家族前缀剥离后的别名 ----
        "蝶式1-3-5": "国债曲线弯曲度(1/3/5年组合利差)",
        "蝶式3-5-10": "国债曲线弯曲度(3/5/10年组合利差)",
        "斜率": "收益率曲线斜率(10年减1年)",
        "dev_z_252": "收益率偏离一年趋势的程度",
        "slope_120": "收益率半年趋势线斜率", "slope_252": "收益率一年趋势线斜率",
        "r2_120": "收益率半年趋势的规整度", "r2_252": "收益率一年趋势的规整度",
        "rate_std_60": "利率波动率(60日)", "vol_of_vol_60": "利率波动率的变化幅度",
        "rate_std_ratio": "短期与长期波动率之比",
        "IRS1Y-DR007": "利率互换与资金利率之差(宽松预期)",
        "DR007-DR001": "7天与隔夜资金利率之差",
        "隔夜占比": "隔夜资金成交占比(加杠杆情绪)",
        "DR001波动20": "隔夜利率波动(20日)",
        "swing20": "股债相对回报(20日)",
        "股20日回报": "股市20日回报", "股60日回报": "股市60日回报",
        "corr_利率权益20": "利率与股市涨跌的联动(20日)",
        "corr_利率权益60": "利率与股市涨跌的联动(60日)",
        "corr_利率铜20": "利率与铜价的联动(20日)", "corr_利率铜60": "利率与铜价的联动(60日)",
        "corr_利率美元20": "利率与美元指数的联动(20日)",
        "corr_利率美元60": "利率与美元指数的联动(60日)",
        "corr_利率标普20": "利率与美股的联动(20日)", "corr_利率标普60": "利率与美股的联动(60日)",
        "corr_break_dollar": "美元联动的突然变化",
        "corr_基金净买入利率20": "基金买入力度与利率变动的联动(20日)",
        "交易盘净买入": "交易型机构(基金券商理财)净买入",
        "配置盘净买入": "配置型机构(保险货基)净买入",
        "交易配置差额": "交易盘与配置盘买入之差",
        "分歧度": "各机构买卖分歧度",
        "逆周期强度": "人民币中间价与即期价差(央行引导力度)",
        "准备金变动": "存款准备金率调整",
        "距月末天数": "距月末天数(月末资金面效应)",
        "距季末天数": "距季末天数(季末考核效应)",
        "月末3日内": "月末最后3天", "季末3日内": "季末最后3天",
        "DR001": "银行间隔夜利率DR001", "DR007": "银行间7天利率DR007",
        "利率互换:FR007:1年": "1年利率互换(市场隐含的政策利率预期)",
        "GC001:收盘价": "交易所隔夜利率GC001", "GC007:收盘价": "交易所7天利率GC007",
        "成交量:R001": "隔夜回购成交量", "成交量:R007": "7天回购成交量",
        "中国:成交量:银行间质押式回购": "银行间质押式回购成交量(资金面活跃度)",
        "中间价:美元兑人民币": "人民币中间价",
        "CFETS:即期汇率:美元兑人民币": "人民币即期汇率",
        "波罗的海干散货指数(BDI)": "波罗的海干散货指数(全球贸易热度)",
        "期货收盘价(连续):COMEX黄金": "COMEX黄金期货",
        "南华工业品指数": "南华工业品指数(大宗商品整体)",
        "中国:秦皇岛港:港口吞吐量:煤炭": "秦皇岛港煤炭吞吐量(能源运输)",
        "中国:北京:地铁客运量": "北京地铁客运量(出行热度)",
        "中国:上海:地铁客运量": "上海地铁客运量(出行热度)",
        "中国:广州:地铁客运量": "广州地铁客运量(出行热度)",
        "中国:深圳:地铁客运量": "深圳地铁客运量(出行热度)",
        "上证科创板50成份指数": "科创板50指数",
        "恒生科技指数": "恒生科技指数",
        "沪深300指数:涨跌幅": "沪深300涨跌幅",
        "沪深300指数:涨跌点数": "沪深300涨跌点数",
        "沪深300:成交金额": "沪深300成交金额",
        "沪深300:成交量": "沪深300成交量",
        "市盈率:沪深300指数": "沪深300市盈率",
        "静态市盈率:沪深300指数": "沪深300静态市盈率",
        "期货结算价(活跃合约):螺纹钢": "螺纹钢期货",
        "期货结算价(活跃合约):动力煤": "动力煤期货",
        "中国:市场价:浮法平板玻璃(4.8/5mm)": "浮法玻璃价格(地产竣工需求)",
        "中国:开工率:石油沥青装置": "石油沥青开工率(道路建设景气)",
    }
    SUFFIX_CN = {
        "z": "相对近一年常见水平的偏离",
        "chg5": "近5日变化", "chg20": "近20日变化", "chg60": "近60日变化",
        "pct20": "近20日涨跌幅", "pct60": "近60日涨跌幅",
        "sum5_z": "近5日累计力度偏离", "sum20_z": "近20日累计力度偏离",
        "sum5_chg5": "近5日累计较前5日的变化",
    }
    FAMILY_PREFIX = re.compile(r"^(曲线|资金|股债|波动|联动|机构|趋势|事件|日历|汇率)__")

    def plain_name(name: str) -> str:
        """技术因子名 → 通俗中文描述"""
        base, suffix = name, ""
        m = re.search(r"__(z|chg\d+|pct\d+|sum\d+_z|sum\d+_chg\d+)$", name)
        if m:
            base, suffix = name[:m.start()], SUFFIX_CN.get(m.group(1), m.group(1))
        else:
            m2 = re.search(r"(chg\d+|pct\d+)$", base)   # 无__直连后缀: 斜率chg5
            if m2:
                base, suffix = base[:m2.start()], SUFFIX_CN.get(m2.group(1), m2.group(1))
        base = FAMILY_PREFIX.sub("", base)
        for k in sorted(FACTOR_CN, key=len, reverse=True):
            if k in base:
                base = base.replace(k, FACTOR_CN[k])
                break
        base = re.sub(r"^月频\.", "", base).replace("周频.", "")
        base = re.sub(r"#\d+", "", base).strip("_ ")
        return base + (f"（{suffix}）" if suffix else "")

    factor_rows, per_horizon = [], {}
    for _, r in sig.iterrows():
        h = int(r["周期"].lstrip("h"))
        m = xgb.XGBClassifier()
        m.load_model(mdir / f"h{h}_xgb.ubj")
        dm = xgb.DMatrix(x_row, feature_names=list(feats.columns))
        contrib = m.get_booster().predict(dm, pred_contribs=True)[0][:-1]
        top = np.argsort(-np.abs(contrib))[:TOPN]
        facts = []
        for fi in top:
            name = feats.columns[fi]
            cur = float(x_row.iloc[0, fi])
            prev = (float(feats.iloc[pos - 20, fi])
                    if pos >= 20 and pd.notna(feats.iloc[pos - 20, fi])
                    else np.nan)
            facts.append({"因子": name, "因子通俗": plain_name(name),
                          "贡献": round(float(contrib[fi]), 4),
                          "方向": "→下行" if contrib[fi] > 0 else "→上行",
                          "当前值": round(cur, 4) if pd.notna(cur) else None,
                          "20日前值": round(prev, 4) if pd.notna(prev) else None})
            factor_rows.append({"周期": r["周期"], "到期日": r["到期日"],
                                "融合概率": r["融合概率"],
                                **facts[-1]})
        per_horizon[r["周期"]] = {
            "融合概率": float(r["融合概率"]), "融合分级": r.get("融合分级", ""),
            "到期日": r["到期日"], "Top因子": facts}

    fac = pd.DataFrame(factor_rows)
    fac.to_csv(OUT / f"{asof.date()}主要因子.csv", index=False,
               encoding="utf-8-sig")
    print(fac[["周期", "因子通俗", "贡献", "方向"]].to_string(index=False))

    # ---- LLM 解读 ----
    key = get_key()
    if not key:
        print("无 key，跳过 LLM 解读")
        return
    lines = []
    for hz, d in per_horizon.items():
        lines.append(f"## {hz}（到期{d['到期日']}，融合下行概率{d['融合概率']:.3f}，"
                     f"分级:{d['融合分级']}）")
        for f in d["Top因子"]:
            chg = (f"（20日前 {f['20日前值']}）" if f["20日前值"] is not None else "")
            lines.append(f"- {f['因子通俗']}：贡献{f['贡献']:+.4f} {f['方向']}，"
                         f"当前值 {f['当前值']} {chg}")
    prompt = (
        "你是利率债研究员。以下是中国10年期国债收益率方向模型在"
        f"{asof.date()}对四个周期(未来5/10/15/20交易日,下行=收益率下降=债涨)的预测，"
        "以及每次预测贡献最大的因子（TreeSHAP贡献，正=推向下行）、因子当前值与"
        "20日前值。请解释：各周期预测结果主要由哪些因子驱动、因子近20日的"
        "变化如何影响方向判断、以及周期之间信号强弱差异的原因。"
        "用中文Markdown输出，结构：## 总体判断、## 各周期驱动因子(h5/h10/h15/h20)、"
        "## 因子变化解读、## 风险提示。因子名称已翻译为通俗说法，请沿用这些说法，"
        "面向非专业读者，不要引入z、pct等术语符号。合计600字以内，基于给定数据，不要编造。\n\n"
        + "\n".join(lines))
    mdfile = OUT / "主要因子LLM解读.md"
    reuse = (mdfile.exists()
             and asof.strftime("%Y-%m-%d") in mdfile.read_text(encoding="utf-8")[:60])
    if reuse:
        md = mdfile.read_text(encoding="utf-8")
        md = md[md.index("\n") + 1:]                    # 去掉首行标题
        md = re.sub(r"^#.*?\n+", "", md, flags=re.M)    # 去掉开头的 ## 标记行
        print("  [LLM] 复用已有解读（删除该md可重新生成）")
    else:
        header = (f"# {asof.date()} 预测主要因子与大模型解读\n\n"
                  f"方法：当日缓存XGB模型对该日预测的TreeSHAP逐因子贡献"
                  f"（正=推向下行），每周期取|贡献|Top{TOPN}。\n\n")
        md = ""
        for extra in ({"thinking": {"type": "disabled"}}, {}):   # 先关思考省token
            try:
                resp = requests.post(
                    API, headers={"Authorization": f"Bearer {key}"},
                    json={"model": MODEL, "temperature": 0.3,
                          "max_tokens": 4000,
                          "messages": [{"role": "user", "content": prompt}],
                          **extra},
                    timeout=240)
                j = resp.json()
                if "choices" in j and j["choices"][0]["message"].get("content"):
                    md = j["choices"][0]["message"]["content"]
                    break
                print("  [LLM重试]", str(j)[:150])
            except Exception as e:
                print("  [LLM异常]", e)
        mdfile.write_text(header + md, encoding="utf-8")
    print("\n===== LLM 解读 =====\n" + md[:600])
    print(f"\n输出: {OUT / f'{asof.date()}主要因子.csv'} / 主要因子LLM解读.md")

    # ---- 注入曲线页 HTML：因子贡献表 + LLM 解读卡片 ----
    html_path = OUT / "时点预测曲线.html"
    if not html_path.exists():
        return

    def esc_html(x) -> str:
        return (str(x).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))

    def md2html(md_text: str) -> str:
        out_lines, in_list = [], False
        for line in md_text.strip().split("\n"):
            t = esc_html(line.strip())
            t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
            if t.startswith("## "):
                if in_list:
                    out_lines.append("</ul>")
                    in_list = False
                out_lines.append(f"<h4 style='margin:12px 0 6px;'>{t[3:]}</h4>")
            elif t.startswith("- "):
                if not in_list:
                    out_lines.append("<ul style='margin:6px 0;padding-left:20px;'>")
                    in_list = True
                out_lines.append(f"<li style='margin:3px 0;'>{t[2:]}</li>")
            elif t.startswith(("1. ", "2. ", "3. ")):
                if not in_list:
                    out_lines.append("<ul style='margin:6px 0;padding-left:20px;'>")
                    in_list = True
                out_lines.append(f"<li style='margin:3px 0;'>{t[3:]}</li>")
            elif t:
                if in_list:
                    out_lines.append("</ul>")
                    in_list = False
                out_lines.append(f"<p style='margin:6px 0;'>{t}</p>")
        if in_list:
            out_lines.append("</ul>")
        return "".join(out_lines)

    horizon_tables = ""
    fac["_h"] = fac["周期"].str.lstrip("h").astype(int)
    for hz, grp in fac.sort_values("_h").groupby("_h"):
        rows = "".join(
            f"<tr><td title='{esc_html(f['因子'])}'><b>{esc_html(f['因子通俗'])}</b>"
            f"<br><small style='color:#94a3b8'>{esc_html(f['因子'])}</small></td>"
            f"<td style='color:{'#10b981' if f['贡献'] > 0 else '#ef4444'};"
            f"font-weight:600'>{f['贡献']:+.4f} {f['方向']}</td>"
            f"<td>{f['当前值']}</td><td>{f['20日前值']}</td></tr>"
            for _, f in grp.iterrows())
        horizon_tables += (
            f"<details{' open' if hz == 'h10' else ''}><summary>"
            f"<b>{hz}</b>（到期 {esc_html(grp['到期日'].iloc[0])}，融合概率 "
            f"{grp['融合概率'].iloc[0]:.3f}）——Top{TOPN} 因子</summary>"
            f"<table style='border-collapse:collapse;width:100%;margin:8px 0;'>"
            f"<tr style='background:#f5f7fa'><th>因子</th><th>TreeSHAP贡献</th>"
            f"<th>当前值</th><th>20日前值</th></tr>{rows}</table></details>")

    section = (
        f"<div id='factorSection' style='background:#fff;border:1px solid #e0e0e0;"
        f"border-radius:8px;padding:18px;margin:18px;'>"
        f"<h3 style='margin:0 0 10px;'>预测主要因子（TreeSHAP 逐因子贡献，"
        f"正=推向下行）与大模型解读</h3>"
        f"{horizon_tables}"
        f"<div style='background:#f8fafc;border-radius:6px;padding:12px 16px;"
        f"margin-top:14px;font-size:13.5px;'>"
        f"<b>大模型解读（GLM-5.3-Flash，仅供参考，未经历史回测验证）：</b>"
        f"{md2html(md)}</div></div>")

    html = html_path.read_text(encoding="utf-8")
    if "id='factorSection'" in html or 'id="factorSection"' in html:
        html = re.sub(r"<div id='?factorSection'?.*?</div>\s*(?=<|$)",
                      section, html, flags=re.S)
    else:
        html = html.replace("</body>", section + "\n</body>")
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML已注入因子与解读: {html_path}")

    # ---- 同步导出 PDF 版本（43 脚本；失败不影响 HTML） ----
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "m43", BASE / "model_new" / "43_导出PDF.py")
        m43 = _ilu.module_from_spec(spec)
        spec.loader.exec_module(m43)
        print(f"PDF已导出: {m43.export_pdf(html_path)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [PDF导出失败，不影响HTML] {exc}")


if __name__ == "__main__":
    main()
