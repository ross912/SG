"""为最终榜单补充股票概念标签；标签不参与评分、排名或过滤。"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import DATA_DIR
from data.fetcher import _code_to_tushare, _get_tushare_pro, _tushare_wait_token

_logger = logging.getLogger(__name__)
_CACHE_PATH = DATA_DIR / "concept_tags.json"
_MAX_CONCEPTS = 8


def get_concept_tags(
    codes: list[str] | set[str],
    max_workers: int = 4,
    allow_network: bool = True,
) -> dict[str, list[str]]:
    """返回代码到概念名列表的映射。接口不可用时返回已有缓存或空列表。"""
    unique_codes = sorted({str(code).zfill(6) for code in codes})
    cache = _load_cache()
    missing = [code for code in unique_codes if code not in cache]
    pro = _get_tushare_pro() if allow_network else None
    if pro is not None and missing:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, pro, code): code for code in missing}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    cache[code] = future.result()
                except Exception as error:
                    _logger.debug("概念标签获取失败 %s: %s", code, error)
                    # 权限或网络失败不能永久缓存为空，否则权限恢复后也不会重试。
                    continue
        _save_cache(cache)
    return {code: cache.get(code, []) for code in unique_codes}


def apply_concept_tags(frame, concept_map: dict[str, list[str]]):
    """给输出表格增加概念文字，不改变行顺序和综合得分。"""
    if frame.empty:
        return frame
    enriched = frame.copy()
    enriched["所属概念"] = enriched["代码"].astype(str).str.zfill(6).map(
        lambda code: "、".join(concept_map.get(code, []))
    )
    return enriched


def _fetch_one(pro, code: str) -> list[str]:
    _tushare_wait_token()
    data = pro.concept_detail(ts_code=_code_to_tushare(code), fields="ts_code,concept_name")
    if data is None or data.empty or "concept_name" not in data.columns:
        return []
    names = [str(value).strip() for value in data["concept_name"].dropna() if str(value).strip()]
    return list(dict.fromkeys(names))[:_MAX_CONCEPTS]


def _load_cache() -> dict[str, list[str]]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        with _CACHE_PATH.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return {str(code).zfill(6): list(names) for code, names in value.items()}
    except Exception:
        return {}


def _save_cache(cache: dict[str, list[str]]) -> None:
    try:
        with _CACHE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception as error:
        _logger.debug("概念标签缓存写入失败: %s", error)
