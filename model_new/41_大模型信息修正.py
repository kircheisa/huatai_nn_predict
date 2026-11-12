# -*- coding: utf-8 -*-
"""
大模型信息修正：搜集国债相关公开信息 -> GLM 分析 -> 对融合信号做有界修正

流程（参考 FinanceMCP 思路：公开金融数据源 + LLM）:
  1. 读 data_new/预测记录.csv 最新发出日的四周期信号（XGB/NN/融合）
  2. 信息搜集（免费源直采，无需搜索资源包）:
     - 新浪财经滚动快讯（lid=2516 财经）
     - 东方财富 7x24 快讯（补 req_trace）
     按关键词过滤: 国债/债市/债券/央行/逆回购/MLF/降准/LPR/资金面/流动性/
     PMI/CPI/PPI/社融/美债/汇率/财政/发行/降息
  3. 调 GLM-5.3-Flash（智谱 API）: 新闻 + 模型信号 -> 严格 JSON 修正意见
  4. 有界修正: 修正概率 = clip(融合概率 + clip(调整量, ±0.08), 0.02, 0.98)
     （LLM 只做小幅倾斜，不可翻转量级；修正未经回测验证，仅作参考）

key 管理: 环境变量 ZHIPU_API_KEY，或 data_new/llm_key.txt（已 gitignore，
切勿写入本文件提交）。建议用后轮换。

输出: data_new/out/大模型修正.csv（同发出日覆盖）
      data_new/out/大模型修正.json（信息要点/理由/原始元数据）
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data_new"
OUT = DATA / "out"
LEDGER = DATA / "预测记录.csv"
KEY_FILE = DATA / "llm_key.txt"
API = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "glm-5.3-flash"

KEYWORDS = ["国债", "债市", "债券", "央行", "逆回购", "MLF", "降准", "LPR",
            "资金面", "流动性", "PMI", "CPI", "PPI", "社融", "美债", "汇率",
            "财政", "发行", "降息", "收益率"]
MAX_ADJ = 0.08


def get_key() -> str:
    import os
    key = os.environ.get("ZHIPU_API_KEY", "").strip()
    if key:
        return key
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    raise SystemExit("缺少API key: 设 ZHIPU_API_KEY 或创建 data_new/llm_key.txt")


def fetch_news() -> list[str]:
    """免费源直采，返回 ['[时间] 标题', ...]"""
    items: list[tuple[str, str]] = []          # (时间戳, 文本)
    try:                                        # 新浪财经滚动
        r = requests.get("https://feed.mix.sina.com.cn/api/roll/get",
                         params={"pageid": 153, "lid": 2516, "num": 50,
                                 "page": 1}, timeout=15)
        for d in r.json().get("data", []):
            t = d.get("ctime", "")
            items.append((t, d.get("title", "") + " " +
                          re.sub("<[^>]+>", "", d.get("intro", ""))[:120]))
    except Exception as e:
        print("  [新浪源失败]", e)
    try:                                        # 东方财富7x24
        r = requests.get(
            "https://np-listapi.eastmoney.com/comm/web/getFastNewsList",
            params={"client": "web", "biz": "web_724", "fastColumn": 102,
                    "sortEnd": "", "pageSize": 50,
                    "req_trace": int(time.time() * 1000)}, timeout=15)
        for d in (r.json().get("data") or {}).get("fastNewsList", []):
            items.append((d.get("showTime", ""), d.get("title", "") + " " +
                          (d.get("summary") or "")[:120]))
    except Exception as e:
        print("  [东财源失败]", e)

    now = time.time()
    picked, seen = [], set()
    for ts, txt in sorted(items, reverse=True):
        try:
            age = now - (dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                         .timestamp() if "-" in ts else int(ts))
        except Exception:
            age = 0
        if age > 4 * 86400:                     # 只取近4天
            continue
        if not any(k in txt for k in KEYWORDS) or txt[:40] in seen:
            continue
        seen.add(txt[:40])
        picked.append(f"[{ts}] {txt.strip()}")
        if len(picked) >= 30:
            break
    return picked


def call_llm(key: str, prompt: str) -> dict:
    body = {"model": MODEL, "temperature": 0.2, "max_tokens": 4000,
            "messages": [
                {"role": "system", "content":
                 "你是银行间市场的利率债研究员。只依据用户提供的新闻快讯与"
                 "量化模型信号作判断，不编造数据。必须只输出一个JSON对象，"
                 "不要输出其他文字。"},
                {"role": "user", "content": prompt}]}
    for extra in ({"thinking": {"type": "disabled"}}, {}):  # 先关思考省token
        try:
            r = requests.post(API, headers={"Authorization": f"Bearer {key}"},
                              json={**body, **extra}, timeout=180)
            j = r.json()
            if "choices" in j:
                return j
            print("  [LLM重试]", str(j)[:200])
        except Exception as e:
            print("  [LLM异常]", e)
    raise SystemExit("LLM 调用失败")


def extract_json(text: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        raise ValueError("响应中无JSON")
    return json.loads(m.group(0))


def main() -> None:
    led = pd.read_csv(LEDGER, dtype={"发出日": str})
    asof = led["发出日"].max()
    sig = led[led["发出日"] == asof].sort_values("周期")
    print(f"[1/4] 最新信号: 发出日 {asof}，{len(sig)} 个周期")

    print("[2/4] 搜集公开信息 ...")
    news = fetch_news()
    print(f"  命中关键词快讯 {len(news)} 条" +
          (f"，最新: {news[0][:60]}" if news else "（无，将仅用模型信号）"))

    print("[3/4] GLM-5.3-Flash 分析修正 ...")
    key = get_key()
    sig_txt = sig[["周期", "到期日", "下行概率", "NN概率", "融合概率",
                   "融合信号"]].to_csv(index=False)
    prompt = (f"今天是{dt.date.today().isoformat()}。量化模型在{asof}对"
              f"中国10年期国债收益率的方向预测如下（下行=收益率下降=债涨，"
              f"概率越接近1越看下行）：\n{sig_txt}\n\n"
              f"近几日相关快讯：\n" +
              "\n".join(news[:30]) +
              "\n\n请基于以上信息对每个周期的下行概率给出小幅修正"
              f"（范围±{MAX_ADJ}，正值=更看下行）。只输出JSON："
              '{"信息要点":["..."],"债市情绪":"偏多|中性|偏空(债)",'
              '"周期修正":{"h5":0.0,"h10":0.0,"h15":0.0,"h20":0.0},'
              '"理由":"...","信息质量":"高|中|低"}')
    resp = call_llm(key, prompt)
    content = resp["choices"][0]["message"].get("content", "")
    op = extract_json(content)
    print(f"  债市情绪: {op.get('债市情绪')}  信息质量: {op.get('信息质量')}")

    print("[4/4] 应用有界修正 ...")
    # 隐含点位换算幅度（与40口径一致：近2年该周期|Δ|中位数）
    raw = pd.read_excel(BASE / "data_ori" / "2026_ori_data_new.xlsx",
                        sheet_name="滚动稳健标准化日频", header=None)
    hi = raw.index[raw[0] == "指标名称"][0]
    names = list(raw.iloc[hi, 1:])
    d = raw.iloc[hi + 1:]
    d.columns = ["date"] + names
    d["date"] = pd.to_datetime(d["date"])
    y = (pd.to_numeric(d.set_index("date")["中国:国债到期收益率:10年"],
                       errors="coerce").replace(0, pd.NA).dropna().sort_index())
    y_now = float(sig["当日收益率"].iloc[0])

    adj = op.get("周期修正", {})
    rows = []
    for _, r in sig.iterrows():
        h = int(r["周期"].lstrip("h"))
        a = float(adj.get(r["周期"], 0.0) or 0.0)
        a = max(-MAX_ADJ, min(MAX_ADJ, a))
        p0 = float(r["融合概率"])
        p1 = min(0.98, max(0.02, p0 + a))
        med_bp = ((y.shift(-h) - y).abs().dropna()
                  [lambda s: s.index >= pd.Timestamp(asof)
                   - pd.Timedelta(days=730)].median() * 100)
        rows.append({"发出日": asof, "周期": r["周期"], "到期日": r["到期日"],
                     "融合概率": p0, "LLM修正量": round(a, 3),
                     "修正概率": round(p1, 3),
                     "修正信号": "下行(债涨)" if p1 >= 0.5 else "上行(债跌)",
                     "修正隐含点位(%)": round(y_now + (1 - 2 * p1) * med_bp / 100, 3),
                     "债市情绪": op.get("债市情绪", "")})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(OUT / "大模型修正.csv", index=False, encoding="utf-8-sig")
    (OUT / "大模型修正.json").write_text(json.dumps(
        {"发出日": asof, "分析": op, "新闻条数": len(news),
         "新闻样本": news[:30], "model": MODEL,
         "usage": resp.get("usage", {})}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n输出: {OUT / '大模型修正.csv'} / 大模型修正.json")

    # ---- 重建HTML：模型信号+LLM解读+修正结果并入同一页面 ----
    html_path = OUT / "时点预测曲线.html"
    if html_path.exists():
        sig_aug = sig.merge(
            res[["周期", "LLM修正量", "修正概率", "修正信号",
                 "修正隐含点位(%)"]], on="周期", how="left")
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "m40", BASE / "model_new" / "40_最新时点预测.py")
        m40 = _ilu.module_from_spec(spec)
        spec.loader.exec_module(m40)
        ymode = str(sig["特征口径"].iloc[0])
        pref = "xgbv4" if "含净买入" in ymode else "xgbv4nb"
        hist = {}
        for h, f in [(10, f"{pref}h10"), (15, "xgbv4h15"), (20, f"{pref}h20")]:
            p = OUT / f"{f}_test_predictions.csv"
            if p.exists():
                dd = pd.read_csv(p)
                dd["date"] = pd.to_datetime(dd["date"])
                hist[f"XGB h{h}"] = dd.set_index("date")["proba_down"]
        m40.make_html(pd.Timestamp(asof), ymode, sig_aug, y, hist,
                      llm={"model": MODEL, **op, "新闻样本": news})
        print(f"HTML已更新: {html_path}")
    print("注: LLM修正为有界小幅倾斜(±0.08)，未经历史回测验证，仅供参考。")


if __name__ == "__main__":
    main()
