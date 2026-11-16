#!/usr/bin/env python3
"""本地新闻代理服务。

用途：
1. 为前端页面提供同源可访问的 `/api/news-search` 接口，避免浏览器跨域限制；
2. 通过智谱官方 Remote MCP 的 `web_search_prime` 工具获取新闻结果；
3. 对返回内容做轻量标准化，便于前端展示与注入大模型提示词。
"""

from __future__ import annotations

import csv
import json
import os
import re
import socketserver
import sys
import time as _time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List
from urllib import error, request
from urllib.parse import urlparse


# 监听 0.0.0.0 以支持局域网内其他设备访问当前电脑上的新闻代理。
HOST = "0.0.0.0"
PORT = 8001
MCP_URL = "https://open.bigmodel.cn/api/mcp/web_search_prime/mcp"
MCP_PROTOCOL_VERSION = "2024-11-05"
def _load_local_search_api_key() -> str:
    """从本地未入库的 search_api.config 读取搜索密钥（避免真实密钥进入版本库）。"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_api.config")
    try:
        with open(config_path, "r", encoding="utf-8-sig") as config_file:
            return config_file.read().strip()
    except OSError:
        return ""


# 密钥优先级：环境变量 BIGMODEL_API_KEY > 本地 search_api.config 文件。
DEFAULT_SEARCH_API_KEY = os.environ.get("BIGMODEL_API_KEY") or _load_local_search_api_key()
DEFAULT_SEARCH_RECENCY = "oneWeek"
DEFAULT_CONTENT_SIZE = "medium"
DEFAULT_LOCATION = "cn"
# 站点定制配置：针对每个财经站点单独定制 query 与域名集合。
# 目的：优先走定制站点检索 + 本地强过滤，尽量不依赖 MCP 的全网模糊搜索。
SITE_PROFILES = [
    {
        "label": "新浪财经",
        "domains": [
            "stock.finance.sina.com.cn",
            "finance.sina.com.cn",
            "finance.sina.cn",
            "cj.sina.com.cn",
        ],
        # 新浪研报类页面（如固定收益策略报告）命中率较低，
        # 因此 query 显式加入“固定收益/研报/策略”，更贴近其研报频道语义。
        "queries": [
            "中国10年期国债收益率 新浪 固定收益 研报 债市策略",
            "10年期国债 收盘 评析 新浪财经 债券",
        ],
        "limit": 3,
    },
    {
        "label": "东方财富",
        "domains": [
            "finance.eastmoney.com",
            "bond.eastmoney.com",
            "quote.eastmoney.com",
            "stock.eastmoney.com",
        ],
        "queries": [
            "10年期国债收益率 东方财富 债市 资金面",
            "国债收益率 债券 东方财富",
        ],
        "limit": 3,
    },
    {
        "label": "财联社",
        "domains": ["cls.cn"],
        "queries": ["10年期国债收益率 财联社 债市早参"],
        "limit": 2,
    },
    {
        "label": "华尔街见闻",
        "domains": ["wallstreetcn.com"],
        "queries": ["中国10年期国债收益率 华尔街见闻 债市"],
        "limit": 2,
    },
    {
        "label": "中证网",
        "domains": ["cs.com.cn", "cnstock.com"],
        "queries": ["10年期国债收益率 中证网 债券"],
        "limit": 2,
    },
]

# 派生白名单：用于全局重排时的域名优先级（兼容旧逻辑）。
PREFERRED_DOMAINS: List[str] = []
for _profile in SITE_PROFILES:
    PREFERRED_DOMAINS.extend(_profile["domains"])


def build_bearer_token() -> str:
    token = os.environ.get("BIGMODEL_API_KEY") or DEFAULT_SEARCH_API_KEY
    return f"Bearer {token}"


def post_mcp(payload: Dict[str, Any], session_id: str | None = None) -> tuple[str, str | None]:
    headers = {
        "Authorization": build_bearer_token(),
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id

    req = request.Request(
        MCP_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with request.urlopen(req, timeout=40) as resp:
        content = resp.read().decode("utf-8", errors="replace")
        next_session_id = resp.headers.get("Mcp-Session-Id") or session_id
        return content, next_session_id


def parse_sse_json(content: str) -> Dict[str, Any]:
    for line in content.splitlines():
        if line.startswith("data:"):
            data_text = line[5:].strip()
            if not data_text:
                continue
            return json.loads(data_text)
    raise ValueError("未从 MCP 响应中解析到 data 负载。")


def mcp_initialize() -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "bond-dashboard-news-proxy",
                "version": "1.0.0",
            },
        },
    }
    content, session_id = post_mcp(payload)
    parsed = parse_sse_json(content)
    if parsed.get("error"):
        raise RuntimeError(parsed["error"])
    if not session_id:
        raise RuntimeError("初始化成功但未返回 MCP Session ID。")
    return session_id


def normalize_news_item(
    raw_item: Dict[str, Any],
    index: int,
    retrieval_mode: str = "mcp_web_search",
    retrieval_label: str = "MCP 全网搜索",
) -> Dict[str, str]:
    title = str(raw_item.get("title") or f"新闻 {index + 1}").strip()
    summary = str(raw_item.get("content") or raw_item.get("summary") or "").strip()
    url = str(raw_item.get("link") or raw_item.get("url") or "").strip()
    source = str(raw_item.get("siteName") or raw_item.get("site") or raw_item.get("source") or "未知来源").strip()
    raw_date = raw_item.get("publishTime") or raw_item.get("published_at") or raw_item.get("date") or ""
    date_text = ""
    if raw_date:
        try:
            date_text = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except ValueError:
            date_text = str(raw_date).strip()
    return {
        "title": title,
        "summary": summary,
        "url": url,
        "source": source,
        "dateText": date_text,
        "retrievalMode": retrieval_mode,
        "retrievalLabel": retrieval_label,
    }


def extract_tool_result(
    parsed_payload: Dict[str, Any],
    retrieval_mode: str = "mcp_web_search",
    retrieval_label: str = "MCP 全网搜索",
) -> List[Dict[str, str]]:
    result = parsed_payload.get("result") or {}
    if result.get("isError"):
        raise RuntimeError("智谱搜索 MCP 返回错误结果。")
    content_items = result.get("content") or []
    text_payload = ""
    for item in content_items:
        if item.get("type") == "text":
            text_payload = item.get("text", "")
            break
    if not text_payload:
        return []

    # 智谱 MCP 当前把结果序列化成一个 JSON 字符串，外层还多包了一层引号。
    current: Any = text_payload
    for _ in range(2):
        if isinstance(current, str):
            try:
                current = json.loads(current)
                continue
            except json.JSONDecodeError:
                pass
        break

    if not isinstance(current, list):
        return []

    return [
        normalize_news_item(
            item,
            index,
            retrieval_mode=retrieval_mode,
            retrieval_label=retrieval_label,
        )
        for index, item in enumerate(current)
        if isinstance(item, dict)
    ]


def merge_news_items(items_list: List[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    seen_keys = set()
    for item in items_list:
        key = (item.get("url") or "").strip().lower() or (item.get("title") or "").strip().lower()
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower().strip()
    except Exception:
        return ""


def get_domain_rank(url: str) -> int:
    domain = extract_domain(url)
    if not domain:
        return len(PREFERRED_DOMAINS) + 1
    for index, preferred in enumerate(PREFERRED_DOMAINS):
        if domain == preferred or domain.endswith(f".{preferred}"):
            return index
    return len(PREFERRED_DOMAINS) + 1


def url_matches_domains(url: str, domains: List[str]) -> bool:
    """本地强过滤：判断 URL 的域名是否真正属于给定域名集合。

    智谱 MCP 的 search_domain_filter 是“偏好”而非“硬约束”，
    因此这里必须再校验一次，避免非目标站点结果混入。
    """
    domain = extract_domain(url)
    if not domain:
        return False
    return any(domain == d or domain.endswith(f".{d}") for d in domains)


def search_site_profile(
    profile: Dict[str, Any],
    session_id: str,
    recency: str,
    start_request_id: int,
) -> tuple[List[Dict[str, str]], int]:
    """对单个站点配置执行定制 query 检索，并对结果做本地强过滤。

    返回 (命中结果, 下一个可用的 request_id)。
    每条命中结果会被标注为该站点的“定向命中”，覆盖默认的 MCP 标签。
    """
    domains = profile.get("domains") or []
    site_label = profile.get("label") or "站点"
    site_limit = int(profile.get("limit") or 3)
    queries = profile.get("queries") or []
    collected: List[Dict[str, str]] = []
    seen_keys: set[str] = set()
    rid = start_request_id

    for q in queries:
        for domain in domains:
            try:
                raw_items = call_search_once(
                    session_id=session_id,
                    query=q,
                    recency=recency,
                    domain_filter=domain,
                    request_id=rid,
                )
                rid += 1
            except Exception:
                rid += 1
                continue

            # 本地强过滤：只保留 URL 真正命中该站点域名集合的结果。
            for item in raw_items:
                if not url_matches_domains(item.get("url", ""), domains):
                    continue
                key = (item.get("url") or "").strip().lower()
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                # 覆写来源与获取方式标签，使其在前端一目了然。
                item["source"] = item.get("source") or site_label
                item["retrievalMode"] = "site_targeted"
                item["retrievalLabel"] = f"{site_label} 定向命中"
                # 打印每条定点命中，便于在终端直观看到各站点的召回情况。
                print(f"[site-hit] {site_label} | {item.get('title', '')[:40]} | {item.get('url', '')}")
                collected.append(item)
                if len(collected) >= site_limit:
                    return collected, rid
            if len(collected) >= site_limit:
                break
        if len(collected) >= site_limit:
            break

    return collected, rid


def rank_news_items(items_list: List[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    # 先去重，再按白名单域名优先、摘要完整度优先进行重排。
    deduped = merge_news_items(items_list, max(limit * 3, 20))
    ranked = sorted(
        deduped,
        key=lambda item: (
            get_domain_rank(item.get("url", "")),
            -len((item.get("summary") or "").strip()),
            item.get("title", ""),
        ),
    )
    return ranked[:limit]


def call_search_once(
    session_id: str,
    query: str,
    recency: str,
    domain_filter: str | None = None,
    request_id: int = 2,
) -> List[Dict[str, str]]:
    retrieval_mode = "mcp_site_search" if domain_filter else "mcp_web_search"
    retrieval_label = "MCP 定向网站搜索" if domain_filter else "MCP 全网搜索"
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {
            "name": "web_search_prime",
            "arguments": {
                "search_query": query,
                "search_recency_filter": recency or DEFAULT_SEARCH_RECENCY,
                "content_size": DEFAULT_CONTENT_SIZE,
                "location": DEFAULT_LOCATION,
            },
        },
    }
    if domain_filter:
        payload["params"]["arguments"]["search_domain_filter"] = domain_filter

    content, _ = post_mcp(payload, session_id=session_id)
    parsed = parse_sse_json(content)
    return extract_tool_result(
        parsed,
        retrieval_mode=retrieval_mode,
        retrieval_label=retrieval_label,
    )


def search_news(query: str, recency: str = DEFAULT_SEARCH_RECENCY, limit: int = 20) -> Dict[str, Any]:
    if not DEFAULT_SEARCH_API_KEY:
        # 未配置搜索 key：直接走免费源直采回退，保持接口可用
        merged = free_news_fallback(query, limit)
        return {
            "ok": True, "query": query, "items": merged, "count": len(merged),
            "siteResults": [], "siteResultsCount": 0,
            "webResultsCount": len(merged), "preferredDomains": PREFERRED_DOMAINS,
            "matchedDomains": [], "siteGroups": [],
            "retrievalNote": "智谱搜索 key 未配置，已回退到免费源直采（新浪/东财）。",
            "fetchedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    session_id = mcp_initialize()
    site_items: List[Dict[str, str]] = []
    site_groups: List[Dict[str, Any]] = []
    matched_domains: List[str] = []
    rid = 2

    # 阶段一：优先走定制站点检索 + 本地强过滤，尽量不依赖 MCP 全网搜索。
    for profile in SITE_PROFILES:
        items, rid = search_site_profile(profile, session_id, recency, rid)
        if not items:
            print(f"[site-summary] {profile.get('label', '站点')} | 命中 0 条")
            continue
        site_items.extend(items)
        site_groups.append({
            "label": profile.get("label"),
            "domains": profile.get("domains"),
            "count": len(items),
            "items": items,
        })
        for d in profile.get("domains") or []:
            if d not in matched_domains:
                matched_domains.append(d)
        print(f"[site-summary] {profile.get('label', '站点')} | 命中 {len(items)} 条")
        for hit in items:
            print(f"[site-url] {hit.get('url', '')}")
        if len(site_items) >= limit:
            break

    # 阶段二：仅当定制站点结果不足时，才回退到一次全网 MCP 搜索补充。
    web_items: List[Dict[str, str]] = []
    if len(site_items) < limit:
        try:
            web_items = call_search_once(
                session_id=session_id,
                query=query,
                recency=recency,
                domain_filter=None,
                request_id=rid,
            )
            rid += 1
        except Exception:
            web_items = []

    # 合并：站点命中结果永远排在全网补充前面。
    final_limit = max(1, min(limit, 10))
    merged = merge_news_items(site_items + web_items, final_limit)

    return {
        "ok": True,
        "query": query,
        "items": merged,
        "count": len(merged),
        "siteResults": site_items,
        "siteResultsCount": len(site_items),
        "webResultsCount": len(merged) - len(site_items),
        "preferredDomains": PREFERRED_DOMAINS,
        "matchedDomains": matched_domains,
        "siteGroups": site_groups,
        "fetchedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# =====================================================================
# 真实模型信号端点：读取预测主项目的产出（40/41 脚本台账与 LLM 修正）
# =====================================================================
# 项目数据目录：环境变量 BOND_PROJECT_DIR，或默认同级的 43_华泰证券预处理
PROJECT_DIR = os.environ.get("BOND_PROJECT_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except OSError:
        return []


def load_signal() -> Dict[str, Any]:
    ledger = _read_csv_rows(os.path.join(PROJECT_DIR, "data_new", "预测记录.csv"))
    fixes = _read_csv_rows(os.path.join(PROJECT_DIR, "data_new", "out", "大模型修正.csv"))
    if not ledger:
        return {"ok": False,
                "error": "未找到预测台账，请先运行 40_最新时点预测.py",
                "projectDir": os.path.abspath(PROJECT_DIR)}
    asof = max(r.get("发出日", "") for r in ledger)
    rows = [r for r in ledger if r.get("发出日") == asof]
    rows = sorted(rows, key=lambda r: r.get("到期日", ""))   # 按到期日递增
    fix_map = {r.get("周期"): r for r in fixes if r.get("发出日") == asof}
    signals = []
    for r in rows:
        s = {
            "horizon": r.get("周期"), "maturity": r.get("到期日"),
            "xgb": float(r["下行概率"]) if r.get("下行概率") else None,
            "nn": float(r["NN概率"]) if r.get("NN概率") else None,
            "fusion": float(r["融合概率"]) if r.get("融合概率") else None,
            "fusionSignal": r.get("融合信号", ""),
            "xgbLevel": r.get("隐含点位估算(%)", ""),
            "fusionLevel": r.get("融合隐含点位(%)", ""),
        }
        f = fix_map.get(r.get("周期"))
        if f:
            s.update({"llmAdj": float(f["LLM修正量"]) if f.get("LLM修正量") else None,
                      "llmProb": float(f["修正概率"]) if f.get("修正概率") else None,
                      "llmSignal": f.get("修正信号", ""),
                      "llmLevel": f.get("修正隐含点位(%)", "")})
        signals.append(s)
    history = sorted(ledger, key=lambda r: (r.get("发出日", ""), r.get("周期", "")),
                     reverse=True)[:12]
    return {
        "ok": True, "asof": asof,
        "yieldNow": rows[0].get("当日收益率", "") if rows else "",
        "mode": rows[0].get("特征口径", "") if rows else "",
        "signals": signals, "history": history,
        "llmSentiment": (fixes[-1].get("债市情绪") if fixes else ""),
        "projectDir": os.path.abspath(PROJECT_DIR),
    }


# =====================================================================
# 免费源回退：智谱 MCP 不可用（无 key / 无资源包）时直采公开快讯
# =====================================================================
FREE_KEYWORDS = ["国债", "债市", "债券", "央行", "逆回购", "MLF", "降准", "LPR",
                 "资金面", "流动性", "PMI", "CPI", "PPI", "社融", "美债", "汇率",
                 "财政", "发行", "降息", "收益率", "国开", "政金债", "同业存单",
                 "Shibor", "公开市场"]


def free_news_fallback(query: str, limit: int = 10) -> List[Dict[str, str]]:
    items: List[Dict[str, Any]] = []
    try:
        url = ("https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2516"
               f"&num=100&page=1&t={int(_time.time() * 1000)}")
        with request.urlopen(url, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8", "ignore"))
        for d in data.get("data") or []:
            items.append({"title": d.get("title", ""),
                          "summary": re.sub(r"<[^>]+>", "", d.get("intro", "") or "")[:120],
                          "url": d.get("url", ""), "source": "新浪财经",
                          "publishTime": d.get("ctime", "")})
    except Exception as exc:  # noqa: BLE001
        print(f"[free-source] sina failed: {exc}")
    try:
        url = ("https://np-listapi.eastmoney.com/comm/web/getFastNewsList"
               f"?client=web&biz=web_724&fastColumn=102&sortEnd=&pageSize=100"
               f"&req_trace={int(_time.time() * 1000)}")
        with request.urlopen(url, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8", "ignore"))
        for d in (data.get("data") or {}).get("fastNewsList") or []:
            items.append({"title": d.get("title", ""),
                          "summary": (d.get("summary") or "")[:120],
                          "url": d.get("newsLink", "") or "", "source": "东方财富",
                          "publishTime": d.get("showTime", "")})
    except Exception as exc:  # noqa: BLE001
        print(f"[free-source] eastmoney failed: {exc}")

    picked, seen = [], set()
    for raw in sorted(items, key=lambda x: str(x.get("publishTime", "")), reverse=True):
        text = raw.get("title", "") + raw.get("summary", "")
        if not any(k in text for k in FREE_KEYWORDS):
            continue
        if raw["title"][:40] in seen:
            continue
        seen.add(raw["title"][:40])
        raw["siteName"] = raw.pop("source")
        picked.append(normalize_news_item(
            raw, len(picked), "free_direct",
            "免费源直采（新浪/东财快讯回退）"))
        if len(picked) >= max(1, min(limit, 15)):
            break
    print(f"[free-source] fallback matched {len(picked)} items")
    return picked


class NewsProxyHandler(BaseHTTPRequestHandler):
    server_version = "BondNewsProxy/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("[news-proxy] " + format % args + "\n")

    def _set_headers(self, status_code: int = 200) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self._set_headers(204)

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._set_headers(200)
            self.wfile.write(json.dumps({"ok": True, "service": "news-proxy"}).encode("utf-8"))
            return
        if self.path == "/api/signal":
            try:
                result = load_signal()
                code = 200 if result.get("ok") else 404
            except Exception as exc:  # noqa: BLE001
                result, code = {"ok": False, "error": str(exc)}, 500
            self._set_headers(code)
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
            return
        self._set_headers(404)
        self.wfile.write(json.dumps({"ok": False, "error": "Not Found"}).encode("utf-8"))

    def do_POST(self) -> None:
        if self.path != "/api/news-search":
            self._set_headers(404)
            self.wfile.write(json.dumps({"ok": False, "error": "Not Found"}).encode("utf-8"))
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
            payload = json.loads(raw_body or "{}")
            query = str(payload.get("query") or "").strip()
            recency = str(payload.get("recency") or DEFAULT_SEARCH_RECENCY).strip()
            limit = int(payload.get("limit") or 10)

            if not query:
                raise ValueError("缺少 query 参数。")

            result = search_news(query=query, recency=recency, limit=limit)
            self._set_headers(200)
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._set_headers(exc.code or 502)
            self.wfile.write(json.dumps({
                "ok": False,
                "error": f"智谱 MCP 调用失败：HTTP {exc.code}",
                "detail": detail,
            }, ensure_ascii=False).encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            # MCP 链路失败（无资源包/超时等）时降级到免费源直采
            try:
                merged = free_news_fallback(str(payload.get("query") or query), limit)
                self._set_headers(200)
                self.wfile.write(json.dumps({
                    "ok": True, "query": query, "items": merged,
                    "count": len(merged), "siteResults": [],
                    "siteResultsCount": 0, "webResultsCount": len(merged),
                    "preferredDomains": PREFERRED_DOMAINS, "matchedDomains": [],
                    "siteGroups": [],
                    "retrievalNote": f"MCP 调用失败已回退免费源：{exc}",
                    "fetchedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }, ensure_ascii=False).encode("utf-8"))
            except Exception:  # noqa: BLE001
                self._set_headers(500)
                self.wfile.write(json.dumps({
                    "ok": False,
                    "error": str(exc),
                }, ensure_ascii=False).encode("utf-8"))


class SafeThreadingHTTPServer(ThreadingHTTPServer):
    """绕过 Python 3.7 在 Windows 下 getfqdn 的编码问题。"""

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = port


def main() -> None:
    server = SafeThreadingHTTPServer((HOST, PORT), NewsProxyHandler)
    print(f"[INFO] News proxy listening at http://{HOST}:{PORT}")
    print("[INFO] Endpoints: GET /api/health , GET /api/signal , POST /api/news-search")
    print(f"[INFO] Project dir: {os.path.abspath(PROJECT_DIR)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] News proxy stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
