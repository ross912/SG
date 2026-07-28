"""面向问答区的资讯检索：Tavily 主源、RSS 降级与短时缓存。"""
from __future__ import annotations

import html
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus, urlparse

import requests


_AUTO_SEARCH_PATTERN = re.compile(
    r"最新|近期|今天|今日|刚刚|目前|现在|本周|昨日|昨天|"
    r"新闻|资讯|消息|公告|政策|财报|舆情|利好|利空|联网|搜索|查一下"
)
_MARKET_EVENT_PATTERN = re.compile(r"原因|影响|主线|异动|大涨|大跌|停牌|复牌")
_STOCK_CODE_PATTERN = re.compile(r"(?<!\d)\d{6}(?!\d)")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_SPACE_PATTERN = re.compile(r"\s+")
_TITLE_SUFFIX_PATTERN = re.compile(r"\s+[-–—]\s+[^-–—]{1,24}$")
_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_TAVILY_FINANCE_DOMAINS = [
    "eastmoney.com", "cls.cn", "finance.sina.com.cn", "cnstock.com",
    "stcn.com", "cs.com.cn", "10jqka.com.cn", "sse.com.cn", "szse.cn",
]


def should_search_web(question: str, mode: str = "auto") -> bool:
    """根据用户选择与问题时效性判断是否联网。"""
    selected = str(mode or "auto").strip().lower()
    if selected == "on":
        return True
    if selected == "off":
        return False
    text = str(question or "").strip()
    if not text:
        return False
    if _AUTO_SEARCH_PATTERN.search(text):
        return True
    return bool(
        _MARKET_EVENT_PATTERN.search(text)
        and (_STOCK_CODE_PATTERN.search(text) or any(word in text for word in ("股票", "个股", "公司")))
    )


def search_latest_news(question: str, *, max_results: int = 8) -> dict[str, Any]:
    """优先查询 Tavily，失败或结果不足时再并行查询两个 RSS 源。"""
    query = _build_query(question)
    limit = min(max(int(max_results), 1), 12)
    provider_mode = os.getenv("SG_QUANT_WEB_SEARCH_PROVIDER", "auto").strip().lower()
    if provider_mode not in {"auto", "rss"}:
        provider_mode = "auto"
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    cache_key = f"{provider_mode}|{bool(tavily_key)}|{query}|{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        cached["cached"] = True
        return cached

    timeout = min(max(float(os.getenv("SG_QUANT_WEB_SEARCH_TIMEOUT", "8")), 2.0), 20.0)
    collected: list[dict[str, str]] = []
    successful: list[str] = []
    errors: list[str] = []

    use_tavily = provider_mode == "auto" and bool(tavily_key)
    if use_tavily:
        try:
            collected.extend(
                _fetch_tavily(
                    _build_tavily_query(question),
                    api_key=tavily_key,
                    timeout=timeout,
                    max_results=limit,
                ),
            )
            successful.append("Tavily")
        except Exception as error:
            errors.append(f"Tavily: {type(error).__name__}")

    minimum_primary_results = min(limit, 4)
    needs_rss = provider_mode == "rss" or len(_deduplicate(collected)) < minimum_primary_results
    if needs_rss:
        rss_rows, rss_successful, rss_errors = _fetch_rss_fallbacks(query, timeout=timeout)
        collected.extend(rss_rows)
        successful.extend(rss_successful)
        errors.extend(rss_errors)

    results = _deduplicate(collected)[:limit]
    payload = {
        "query": query,
        "searched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "providers": successful,
        "results": results,
        "errors": errors,
        "cached": False,
    }
    _cache_put(cache_key, payload)
    return _copy_payload(payload)


def _fetch_tavily(
    query: str,
    *,
    api_key: str,
    timeout: float,
    max_results: int,
) -> list[dict[str, str]]:
    response = requests.post(
        "https://api.tavily.com/search",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SGQuant/1.0 (+private-research-terminal)",
        },
        json={
            "query": query,
            "topic": "finance",
            "search_depth": "basic",
            "time_range": "week",
            "max_results": min(max(int(max_results), 1), 12),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "include_domains": _TAVILY_FINANCE_DOMAINS,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    raw_results = body.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Tavily response missing results")
    rows = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = _clean_text(item.get("title"))
        url = _clean_text(item.get("url"))
        if not title or not url.startswith(("https://", "http://")):
            continue
        suffix = _TITLE_SUFFIX_PATTERN.search(title)
        source = suffix.group(0).strip(" -–—") if suffix else urlparse(url).netloc
        rows.append({
            "title": title,
            "url": url,
            "source": source.removeprefix("www.")[:80] or "Tavily",
            "published_at": _normalize_date(item.get("published_date")),
            "snippet": _clean_text(item.get("content"))[:500],
            "provider": "Tavily",
        })
    return rows


def _fetch_rss_fallbacks(
    query: str,
    *,
    timeout: float,
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    providers = {
        "Google News": lambda: _fetch_google_news(query, timeout=timeout),
        "Bing News": lambda: _fetch_bing_news(query, timeout=timeout),
    }
    collected: list[dict[str, str]] = []
    successful: list[str] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=len(providers), thread_name_prefix="sgq-search") as executor:
        future_map = {executor.submit(fetcher): name for name, fetcher in providers.items()}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                collected.extend(future.result())
                successful.append(name)
            except Exception as error:
                errors.append(f"{name}: {type(error).__name__}")
    return collected, successful, errors


def _fetch_google_news(query: str, *, timeout: float) -> list[dict[str, str]]:
    search_query = f"{query} when:7d"
    url = (
        "https://news.google.com/rss/search"
        f"?q={quote_plus(search_query)}&hl=zh-CN&gl=CN&ceid=CN%3Azh-Hans"
    )
    return _fetch_rss(url, provider="Google News", timeout=timeout)


def _fetch_bing_news(query: str, *, timeout: float) -> list[dict[str, str]]:
    url = (
        "https://www.bing.com/news/search"
        f"?q={quote_plus(query)}&format=rss&setlang=zh-cn"
    )
    return _fetch_rss(url, provider="Bing News", timeout=timeout)


def _fetch_rss(url: str, *, provider: str, timeout: float) -> list[dict[str, str]]:
    response = requests.get(
        url,
        headers={
            "User-Agent": "SGQuant/1.0 (+private-research-terminal)",
            "Accept": "application/rss+xml, application/xml, text/xml",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.content
    if len(content) > 2_000_000:
        raise ValueError("RSS response too large")
    return _parse_rss(content, provider=provider)


def _parse_rss(content: bytes | str, *, provider: str) -> list[dict[str, str]]:
    root = ET.fromstring(content)
    rows = []
    for item in root.findall(".//item")[:40]:
        title = _clean_text(item.findtext("title"))
        url = _clean_text(item.findtext("link"))
        if not title or not url.startswith(("https://", "http://")):
            continue
        source = _clean_text(item.findtext("source"))
        if not source:
            source = _find_namespaced_text(item, "Source")
        if not source:
            suffix = _TITLE_SUFFIX_PATTERN.search(title)
            source = suffix.group(0).strip(" -–—") if suffix else provider
        description = _clean_text(item.findtext("description"))
        rows.append({
            "title": title,
            "url": url,
            "source": source[:80],
            "published_at": _normalize_date(item.findtext("pubDate")),
            "snippet": description[:360],
            "provider": provider,
        })
    return rows


def _build_query(question: str) -> str:
    text = _SPACE_PATTERN.sub(" ", str(question or "")).strip()
    text = re.sub(r"^(请|帮我|能否|可以|麻烦)?\s*(搜索|联网搜索|查一下|分析|总结)\s*", "", text)
    text = text[:120].strip(" ，。？！,?!")
    if not text:
        text = "A股 市场"
    has_specific_target = bool(_STOCK_CODE_PATTERN.search(text)) or any(
        word in text for word in ("股份", "科技", "集团", "银行", "证券", "药业", "能源")
    )
    if not has_specific_target:
        if any(word in text for word in ("政策", "公告", "公司消息", "公司新闻")):
            return "A股 (政策 OR 公告 OR 公司)"
        if any(word in text for word in ("新闻", "资讯", "消息", "主线", "热点")):
            return "A股 (市场 OR 财经 OR 政策)"
        if any(word in text for word in ("市场", "指数", "行情", "强弱", "涨跌")):
            return "A股 市场"
    if not any(word in text for word in ("A股", "股票", "指数", "市场", "公司", "财经")):
        text = f"{text} A股 财经"
    return text


def _build_tavily_query(question: str) -> str:
    """Tavily 更适合自然语言，不复用面向 RSS 的布尔查询。"""
    text = _SPACE_PATTERN.sub(" ", str(question or "")).strip()
    text = re.sub(r"^(请|帮我|能否|可以|麻烦)?\s*(搜索|联网搜索|查一下)\s*", "", text)
    text = text[:160].strip(" ，。？！,?!")
    has_specific_target = bool(_STOCK_CODE_PATTERN.search(text)) or any(
        word in text for word in ("股份", "科技", "集团", "银行", "证券", "药业", "能源")
    )
    if has_specific_target:
        return f"{text}，查找中国A股相关的最新公告和财经报道"
    if any(word in text for word in ("政策", "公告", "公司消息", "公司新闻")):
        return "中国A股市场最近一周的重要政策、监管动态和上市公司公告"
    if any(word in text for word in ("新闻", "资讯", "消息", "主线", "热点")):
        return "中国A股市场最近一周的财经新闻、市场主线和重要政策"
    if any(word in text for word in ("市场", "指数", "行情", "强弱", "涨跌")):
        return "中国A股市场今日行情、主要指数表现和最新市场动态"
    return f"{text or '中国A股市场'}，查找最近一周的相关财经资讯"


def _deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    seen = set()
    for row in sorted(rows, key=lambda item: item.get("published_at", ""), reverse=True):
        key = re.sub(r"[\W_]+", "", _TITLE_SUFFIX_PATTERN.sub("", row.get("title", "")).lower())
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = _HTML_TAG_PATTERN.sub(" ", text)
    return _SPACE_PATTERN.sub(" ", text).strip()


def _find_namespaced_text(item: ET.Element, suffix: str) -> str:
    for child in item:
        if child.tag.rsplit("}", 1)[-1].lower() == suffix.lower():
            return _clean_text(child.text)
    return ""


def _normalize_date(value: str | None) -> str:
    if not value:
        return ""
    text = _clean_text(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone().isoformat(
            timespec="seconds",
        )
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        parsed = parsedate_to_datetime(text)
        return parsed.astimezone().isoformat(timespec="seconds")
    except (TypeError, ValueError, OverflowError):
        return text[:80]


def _cache_get(key: str) -> dict[str, Any] | None:
    ttl = min(max(int(os.getenv("SG_QUANT_WEB_SEARCH_CACHE_SECONDS", "600")), 30), 3600)
    now = time.monotonic()
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None or now - entry[0] > ttl:
            _CACHE.pop(key, None)
            return None
        return _copy_payload(entry[1])


def _cache_put(key: str, payload: dict[str, Any]) -> None:
    with _CACHE_LOCK:
        if len(_CACHE) >= 100:
            oldest = min(_CACHE, key=lambda item: _CACHE[item][0])
            _CACHE.pop(oldest, None)
        _CACHE[key] = (time.monotonic(), _copy_payload(payload))


def _copy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "providers": list(payload.get("providers", [])),
        "results": [dict(item) for item in payload.get("results", [])],
        "errors": list(payload.get("errors", [])),
    }
