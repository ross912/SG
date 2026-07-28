"""AkShare 调用的随机延迟、重试和轻量熔断。"""
import random
import time
from collections import defaultdict
from functools import wraps


class Source:
    AKSHARE = "akshare"


_failures: dict[str, int] = defaultdict(int)
_opened_at: dict[str, float] = defaultdict(float)
_THRESHOLD = 5
_COOLDOWN_SECONDS = 300


def _random_delay(min_sec: float = 0.8, max_sec: float = 2.5) -> float:
    """返回指定范围的随机等待时间。"""
    return random.uniform(min_sec, max_sec)


def with_retry(source: str = Source.AKSHARE, max_retries: int = 3):
    """为网络调用增加指数退避；连续失败时短暂熔断。"""
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if _failures[source] >= _THRESHOLD:
                if time.time() - _opened_at[source] < _COOLDOWN_SECONDS:
                    raise ConnectionError(f"数据源 [{source}] 暂时熔断")
                _failures[source] = 0

            last_error = None
            for attempt in range(max_retries):
                try:
                    result = function(*args, **kwargs)
                    _failures[source] = 0
                    return result
                except Exception as error:
                    last_error = error
                    _failures[source] += 1
                    if _failures[source] >= _THRESHOLD:
                        _opened_at[source] = time.time()
                    if attempt < max_retries - 1:
                        delay = min(2 ** (attempt + 1) + random.random(), 60)
                        print(f"  [{source}] 第 {attempt + 1} 次重试，等待 {delay:.1f}s: {error}")
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator
