"""多源财经快讯聚合、去重和本地缓存。"""
import hashlib
import html
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

from config import DATA_DIR
from services.storage import read_json, write_json


INTELLIGENCE_DIR = DATA_DIR / "intelligence"
LATEST_NEWS_PATH = INTELLIGENCE_DIR / "latest_news.json"
SOURCE_URLS = {
    "东方财富": "https://kuaixun.eastmoney.com/7_24.html",
    "新浪财经": "https://finance.sina.com.cn/7x24",
    "财联社": "https://www.cls.cn/telegraph",
}
_TAG_PATTERN = re.compile(r"<[^>]+>")
_SPACE_PATTERN = re.compile(r"\s+")
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
    ),
}


def refresh_news(hours: int = 24, limit: int = 50) -> dict[str, Any]:
    """抓取最近财经快讯，保留来源原标题并持久化。"""
    records, errors = fetch_latest_news(hours=hours, limit=limit)
    previous = load_latest_news()
    if not records and previous.get("items"):
        previous["last_attempt_at"] = _now_iso()
        previous["refresh_errors"] = errors or ["新闻源未返回新数据"]
        write_json(LATEST_NEWS_PATH, previous)
        return previous

    for item in records:
        item["title"] = item["original_title"]

    generated_at = _now_iso()
    payload = {
        "schema_version": 2,
        "generated_at": generated_at,
        "window_hours": int(hours),
        "item_count": len(records),
        "sources": sorted({item["source"] for item in records}),
        "refresh_errors": errors,
        "items": records,
    }
    daily_path = INTELLIGENCE_DIR / f"news_{generated_at[:10]}.json"
    write_json(daily_path, payload)
    write_json(LATEST_NEWS_PATH, payload)
    return payload


def fetch_latest_news(hours: int = 24, limit: int = 50) -> tuple[list[dict], list[str]]:
    """抓取三家快讯并统一字段；单个源失败不影响其他源。"""
    now = datetime.now().astimezone()
    cutoff = now - timedelta(hours=max(1, int(hours)))
    records: list[dict] = []
    errors: list[str] = []
    sources = (
        ("东方财富", _fetch_eastmoney),
        ("新浪财经", _fetch_sina),
        ("财联社", _fetch_cls),
    )
    for source_name, fetcher in sources:
        try:
            records.extend(fetcher())
        except Exception as error:  # 第三方字段/网络异常需隔离
            errors.append(f"{source_name}：{type(error).__name__}")

    deduplicated: dict[str, dict] = {}
    for item in records:
        published = _parse_datetime(item.get("published_at"))
        if published is not None and published < cutoff:
            continue
        fingerprint = _fingerprint(item)
        item["fingerprint"] = fingerprint
        item["id"] = hashlib.sha256(
            f"{item['source']}|{fingerprint}|{item.get('published_at', '')}".encode("utf-8")
        ).hexdigest()[:16]
        existing = deduplicated.get(fingerprint)
        if existing is None or len(item.get("content", "")) > len(existing.get("content", "")):
            deduplicated[fingerprint] = item

    ordered = sorted(
        deduplicated.values(),
        key=lambda item: item.get("published_at", ""),
        reverse=True,
    )
    return ordered[:max(1, int(limit))], errors


def load_latest_news() -> dict[str, Any]:
    payload = read_json(LATEST_NEWS_PATH, {})
    return payload if isinstance(payload, dict) else {}


def _fetch_eastmoney() -> list[dict]:
    payload = _get_json(
        "https://np-weblist.eastmoney.com/comm/web/getFastNewsList",
        params={
            "client": "web", "biz": "web_724", "fastColumn": "102",
            "sortEnd": "", "pageSize": "200", "req_trace": "1710315450384",
        },
    )
    output = []
    for row in payload.get("data", {}).get("fastNewsList", []):
        title = _plain(row.get("title"))
        content = _plain(row.get("summary")) or title
        if not title and not content:
            continue
        code = _plain(row.get("code"))
        output.append(_record(
            source="东方财富",
            title=title or _derive_title(content),
            content=content,
            published=row.get("showTime"),
            url=f"https://finance.eastmoney.com/a/{code}.html" if code else SOURCE_URLS["东方财富"],
        ))
    return output


def _fetch_sina() -> list[dict]:
    payload = _get_json(
        "https://zhibo.sina.com.cn/api/zhibo/feed",
        params={
            "page": "1", "page_size": "50", "zhibo_id": "152", "tag_id": "0",
            "dire": "f", "dpc": "1", "pagesize": "50", "type": "1",
        },
    )
    output = []
    rows = payload.get("result", {}).get("data", {}).get("feed", {}).get("list", [])
    for row in rows:
        content = _plain(row.get("rich_text"))
        if not content:
            continue
        output.append(_record(
            source="新浪财经",
            title=_derive_title(content),
            content=content,
            published=row.get("create_time"),
            url=SOURCE_URLS["新浪财经"],
        ))
    return output


def _fetch_cls() -> list[dict]:
    params = {
        "app": "CailianpressWeb",
        "category": "",
        "last_time": int(time.time()),
        "os": "web",
        "refresh_type": "1",
        "rn": "50",
        "sv": "8.4.6",
    }
    params["sign"] = hashlib.md5(
        hashlib.sha1(urlencode(params).encode("utf-8")).hexdigest().encode("utf-8")
    ).hexdigest()
    payload = _get_json("https://www.cls.cn/v1/roll/get_roll_list", params=params)
    output = []
    for row in payload.get("data", {}).get("roll_data", []):
        title = _plain(row.get("title"))
        content = _plain(row.get("content")) or title
        if not title and not content:
            continue
        output.append(_record(
            source="财联社",
            title=title or _derive_title(content),
            content=content,
            published=row.get("ctime"),
            url=SOURCE_URLS["财联社"],
        ))
    return output


def _record(*, source: str, title: str, content: str, published: Any, url: str) -> dict:
    parsed = _parse_datetime(published)
    return {
        "source": source,
        "original_title": _clean_headline(title),
        "content": _plain(content)[:1200],
        "published_at": parsed.isoformat(timespec="seconds") if parsed else "",
        "url": _safe_url(url, SOURCE_URLS.get(source, "")),
    }


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)) and float(value) > 1_000_000_000:
            stamp = pd.to_datetime(value, unit="s", utc=True)
        else:
            stamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(stamp):
            return None
        python_value = stamp.to_pydatetime()
        if python_value.tzinfo is None:
            python_value = python_value.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return python_value.astimezone()
    except (TypeError, ValueError, OverflowError):
        return None


def _plain(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = html.unescape(_TAG_PATTERN.sub(" ", str(value)))
    return _SPACE_PATTERN.sub(" ", text).strip()


def _derive_title(content: str) -> str:
    first = re.split(r"[。！？；\n]", _plain(content), maxsplit=1)[0]
    return _clean_headline(first or content)


def _clean_headline(value: Any) -> str:
    text = _plain(value).strip(" \"'“”‘’。")
    return text if len(text) <= 68 else f"{text[:67]}…"


def _fingerprint(item: dict) -> str:
    basis = _SPACE_PATTERN.sub("", item.get("original_title") or item.get("content", ""))
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _safe_url(value: Any, fallback: str = "") -> str:
    url = _plain(value)
    if re.match(r"^https?://", url, flags=re.IGNORECASE):
        return url
    return fallback if re.match(r"^https?://", fallback, flags=re.IGNORECASE) else ""


def _get_json(url: str, *, params: dict[str, str] | None = None) -> dict:
    timeout = max(3.0, float(os.getenv("SG_QUANT_HTTP_TIMEOUT", "15")))
    response = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("新闻源返回格式异常")
    return payload


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
