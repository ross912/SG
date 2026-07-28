"""
数据获取封装：股票列表/不复权数据用 Tushare，复权日线用 AkShare。
所有函数内置重试和延迟，避免触发反爬。
"""
import os
import time
import threading
import pandas as pd
import akshare as ak
import requests
from datetime import datetime
from typing import Optional

from data.rate_guard import with_retry, Source, _random_delay
from config import TUSHARE_TOKEN


def _install_default_http_timeout() -> None:
    """给未显式设置超时的第三方 requests 调用补上上限，避免工作线程永久卡住。"""
    request_method = requests.sessions.Session.request
    if getattr(request_method, "_sg_quant_timeout", False):
        return
    try:
        timeout = max(1.0, float(os.getenv("SG_QUANT_HTTP_TIMEOUT", "15")))
    except ValueError:
        timeout = 15.0

    def bounded_request(session, method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return request_method(session, method, url, **kwargs)

    bounded_request._sg_quant_timeout = True
    requests.sessions.Session.request = bounded_request


_install_default_http_timeout()

# ── 按数据源分类的请求间隔 ──────────────────────────────────────────
AKSHARE_MIN = 2.0
AKSHARE_MAX = 5.0

# ── Tushare 令牌桶限流器（免费版 daily 50次/分钟，留 5次余量）───────
_TUSHARE_RATE = 45          # 每分钟最大请求数
_TUSHARE_WINDOW = 60.0      # 时间窗口（秒）
_tushare_tokens = _TUSHARE_RATE
_tushare_last_refill = time.time()
_tushare_rate_lock = threading.Lock()
_tushare_rate_hits = 0              # 连续触发限频的次数
_tushare_disabled_until = 0         # 熔断到期时间戳（0=未熔断）

# 新浪历史接口内部首次创建 JavaScript 运行时；并发首次初始化会让
# py-mini-racer 在 macOS 直接终止进程。只串行化第一次成功调用，之后可并发。
_akshare_js_init_lock = threading.Lock()
_akshare_js_ready = False
_market_daily_cache: dict[str, pd.DataFrame] = {}
_market_daily_cache_lock = threading.Lock()


def _tushare_record_rate_hit():
    """记录一次 Tushare 限频命中，达到阈值后熔断 5 分钟。"""
    global _tushare_rate_hits, _tushare_disabled_until
    with _tushare_rate_lock:
        _tushare_rate_hits += 1
        if _tushare_rate_hits >= 3:
            _tushare_disabled_until = time.time() + 300
            print(f"  [tushare] 连续 {_tushare_rate_hits} 次限频，熔断 5 分钟，切换 akshare")


def _tushare_wait_token():
    """获取一个 Tushare 请求令牌（线程安全，阻塞式）。"""
    global _tushare_tokens, _tushare_last_refill
    refill_rate = _TUSHARE_RATE / _TUSHARE_WINDOW
    while True:
        with _tushare_rate_lock:
            now = time.time()
            elapsed = now - _tushare_last_refill
            _tushare_tokens = min(
                _TUSHARE_RATE,
                _tushare_tokens + elapsed * refill_rate,
            )
            _tushare_last_refill = now

            if _tushare_tokens >= 1.0:
                _tushare_tokens -= 1.0
                return

            wait = (1.0 - _tushare_tokens) / refill_rate
        # 释放锁后等待；醒来后重新竞争令牌，避免多个线程同时突发请求。
        time.sleep(wait + 0.05)


# ── Tushare Pro 全局客户端（懒加载单例）──────────────────────────────
_tushare_pro = None
_tushare_lock = threading.Lock()


def _get_tushare_pro():
    """获取 Tushare Pro 客户端（线程安全单例）。"""
    global _tushare_pro
    if not TUSHARE_TOKEN:
        return None
    if _tushare_pro is None:
        with _tushare_lock:
            if _tushare_pro is None:
                import tushare as ts
                ts.set_token(TUSHARE_TOKEN)
                _tushare_pro = ts.pro_api(timeout=15)
                # 1.4.x 客户端仍默认旧域名 api.waditu.com；本机实测其 DNS 不可用。
                # 改用同一官方服务的 api.tushare.pro，服务器可通过环境变量覆盖。
                if hasattr(_tushare_pro, "_DataApi__http_url"):
                    _tushare_pro._DataApi__http_url = os.getenv(
                        "TUSHARE_API_URL", "http://api.tushare.pro/dataapi"
                    )
    return _tushare_pro


def _code_to_tushare(code: str) -> str:
    """将纯数字代码转为 Tushare 格式：000001 → 000001.SZ"""
    if code.startswith("6"):
        return f"{code}.SH"
    elif code.startswith(("4", "8", "9")):
        return f"{code}.BJ"
    else:
        return f"{code}.SZ"


def _code_to_daily_symbol(code: str) -> str:
    """将纯数字代码转为 stock_zh_a_daily 所需的带交易所前缀格式（akshare fallback 用）。"""
    if code.startswith("6"):
        return f"sh{code}"
    elif code.startswith(("4", "8", "9")):
        return f"bj{code}"
    else:
        return f"sz{code}"


def _normalize_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """统一列名为英文，确保 date/open/high/low/close/volume/amount 存在。"""
    if df is None or df.empty:
        return None

    col_map = {}
    turnover_is_percent = False
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ("date", "日期", "时间", "time", "trade_date", "trade date"):
            col_map[c] = "date"
        elif cl in ("open", "开盘"):
            col_map[c] = "open"
        elif cl in ("high", "最高"):
            col_map[c] = "high"
        elif cl in ("low", "最低"):
            col_map[c] = "low"
        elif cl in ("close", "收盘"):
            col_map[c] = "close"
        elif cl in ("volume", "vol", "成交量"):
            col_map[c] = "volume"
        elif cl in ("amount", "成交额"):
            col_map[c] = "amount"
        elif cl in ("pct_chg", "pct_change", "涨跌幅"):
            col_map[c] = "pct_change"
        elif cl in ("amplitude", "振幅"):
            col_map[c] = "amplitude"
        elif cl in ("pre_close", "昨收"):
            col_map[c] = "pre_close"
        elif cl in ("turnover", "换手率", "换手"):
            col_map[c] = "turnover"
            if cl in ("换手率", "换手"):
                turnover_is_percent = True
        elif cl in ("outstanding_share", "流通股"):
            col_map[c] = "outstanding_share"
    if col_map:
        df = df.rename(columns=col_map)

    # 确保 date 列存在
    if "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 内部统一用小数表示换手率：0.005 = 0.5%。中文行情接口通常返回百分数。
    if "turnover" in df.columns:
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
        if turnover_is_percent:
            df["turnover"] = df["turnover"] / 100.0

    # 计算缺失列
    if "pct_change" not in df.columns:
        df["pct_change"] = df["close"].pct_change() * 100

    if "amplitude" not in df.columns and "high" in df.columns and "low" in df.columns:
        df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100

    return df


# ══════════════════════════════════════════════════════════════════════
# Tushare 日K线（不复权主线；复权仅在 adj_factor 可用时成立）
# ══════════════════════════════════════════════════════════════════════

def _fetch_kline_tushare(code: str, start_date: str, end_date: str,
                          adjust: str = "qfq") -> Optional[pd.DataFrame]:
    """通过 Tushare Pro 获取日K线，含前复权处理。"""
    global _tushare_rate_hits, _tushare_disabled_until

    # 熔断检查
    with _tushare_rate_lock:
        if _tushare_disabled_until > 0 and time.time() < _tushare_disabled_until:
            return None  # 熔断中，直接跳过，让调用方降级到 akshare
        if _tushare_disabled_until > 0 and time.time() >= _tushare_disabled_until:
            _tushare_disabled_until = 0
            _tushare_rate_hits = 0

    if not TUSHARE_TOKEN:
        return None

    pro = _get_tushare_pro()
    if pro is None:
        return None
    ts_code = _code_to_tushare(code)

    # 令牌桶限流（Tushare 免费版 daily 50次/分钟）
    _tushare_wait_token()

    # 1) 拉取未复权日线数据
    try:
        df = pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
        )
    except Exception as e:
        msg = str(e)
        if "频率" in msg or "frequency" in msg.lower() or "50次" in msg or "超限" in msg:
            _tushare_record_rate_hit()
        else:
            print(f"  [tushare:{code}] daily() 失败: {e}")
        return None

    if df is None or df.empty:
        return None

    # Tushare 默认按日期倒序返回。复权和涨跌幅计算前必须先转为正序。
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 2) 拉取复权因子
    if adjust == "qfq":
        try:
            # adj_factor 也是一次独立 API 请求，必须计入限频。
            _tushare_wait_token()
            adj_df = pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
        except Exception:
            adj_df = None

        if adj_df is None or adj_df.empty:
            return None

        # 合并复权因子（统一用 trade_date 做 key）
        adj_df["trade_date"] = pd.to_datetime(adj_df["trade_date"])
        df = df.merge(
            adj_df[["trade_date", "adj_factor"]],
            on="trade_date", how="left",
        )
        if "adj_factor" not in df.columns or not df["adj_factor"].notna().any():
            return None

        latest_factor = df.loc[df["trade_date"].idxmax(), "adj_factor"]
        if not latest_factor or latest_factor <= 0:
            return None
        factor_ratio = df["adj_factor"] / latest_factor
        for col in ["open", "high", "low", "close", "pre_close"]:
            if col in df.columns:
                df[col] = df[col] * factor_ratio
    # 无论复权因子是否成功，都按日期正序重新计算涨跌幅，避免使用倒序收益。
    if "close" in df.columns:
        df["pct_chg"] = df["close"].pct_change() * 100

    # Tushare vol=手 → 股, amount=千元 → 元（与 akshare 对齐）
    if "vol" in df.columns:
        df["vol"] = df["vol"] * 100
    if "amount" in df.columns:
        df["amount"] = df["amount"] * 1000

    # 清理 tushare 特有列
    df.drop(columns=["ts_code"], inplace=True, errors="ignore")

    return df


# ══════════════════════════════════════════════════════════════════════
# AkShare 复权日K线（生产主线）
# ══════════════════════════════════════════════════════════════════════

def _fetch_kline_akshare(code: str, start_date: str, end_date: str,
                          adjust: str = "qfq") -> Optional[pd.DataFrame]:
    """通过 AkShare 获取复权日线，首次 JS 初始化串行、后续并发。"""
    global _akshare_js_ready

    symbol = _code_to_daily_symbol(code)

    df = None
    for attempt in range(3):
        try:
            time.sleep(_random_delay(0.1, 0.35))
            if not _akshare_js_ready:
                with _akshare_js_init_lock:
                    if not _akshare_js_ready:
                        df = ak.stock_zh_a_daily(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                            adjust=adjust,
                        )
                        if df is not None and not df.empty:
                            _akshare_js_ready = True
            if _akshare_js_ready and (df is None or df.empty):
                df = ak.stock_zh_a_daily(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )
            if df is not None and not df.empty:
                break
        except Exception as e:
            if attempt == 2:
                print(f"  [{code}] stock_zh_a_daily failed: {e}, trying fallback...")
            else:
                print(f"  [{code}] retry {attempt+1}/3: {e}")
                time.sleep(1.5 * (attempt + 1))

    # 备用：东财接口不使用 JavaScript 运行时；部分网络环境会主动断开。
    if df is None or df.empty:
        for attempt in range(2):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                    timeout=15,
                )
                if df is not None and not df.empty:
                    break
            except Exception:
                if attempt == 0:
                    time.sleep(1.0)

    return df


# ══════════════════════════════════════════════════════════════════════
# 统一入口：按是否复权选择可靠数据源
# ══════════════════════════════════════════════════════════════════════

def get_kline_daily(code: str, start_date: str = "20200101",
                     end_date: str | None = None,
                     adjust: str = "qfq") -> Optional[pd.DataFrame]:
    """获取单只股票日K线（前复权）。

    复权数据优先 AkShare；不复权数据优先 Tushare Pro。

    旧 Token 的 daily 为 50 次/分钟，而 adj_factor 仅 1 次/小时，不能把
    daily 原始价冒充前复权价格。调整模式因此直接使用能返回完整复权序列的
    AkShare 新浪接口；其失败时再尝试东财接口。
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y%m%d")

    if adjust in ("qfq", "hfq"):
        df = _fetch_kline_akshare(code, start_date, end_date, adjust)
        return _normalize_dataframe(df)

    # ── 不复权主线：Tushare Pro ──
    df = _fetch_kline_tushare(code, start_date, end_date, adjust)
    if df is not None and not df.empty:
        df = _normalize_dataframe(df)
        if df is not None:
            return df
    # _fetch_kline_tushare 内部已打印错误，此处静默降级

    # ── 备线：AkShare ──
    df = _fetch_kline_akshare(code, start_date, end_date, adjust)
    return _normalize_dataframe(df)


# ══════════════════════════════════════════════════════════════════════
# 股票列表：Tushare 主线，AkShare 备线
# ══════════════════════════════════════════════════════════════════════

@with_retry(source=Source.AKSHARE, max_retries=3)
def get_all_a_stocks() -> pd.DataFrame:
    """获取全部上市 A 股的代码与名称，不按 ST、流动性或涨跌停筛选。"""
    pro = _get_tushare_pro()
    if pro is not None:
        try:
            _tushare_wait_token()
            result = pro.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,industry,market,list_date",
            )
            if result is not None and not result.empty:
                result = result.rename(columns={"symbol": "code"})
                result["code"] = result["code"].astype(str).str.zfill(6)
                return result.drop_duplicates("code").reset_index(drop=True)
        except Exception as error:
            print(f"  [tushare] 股票列表失败，切换 AkShare: {error}")

    time.sleep(_random_delay(AKSHARE_MIN, AKSHARE_MAX))
    result = ak.stock_info_a_code_name()
    result = result.rename(columns={
        "stock_code": "code", "symbol": "code",
        "stock_name": "name", "short_name": "name",
    })
    if "code" not in result.columns or "name" not in result.columns:
        raise ValueError("股票列表缺少 code/name 列")
    if "industry" not in result.columns:
        result["industry"] = ""
    result["code"] = result["code"].astype(str).str.zfill(6)
    return result.drop_duplicates("code").reset_index(drop=True)


def get_market_daily_snapshot(trade_date: str) -> pd.DataFrame:
    """一次获取全市场指定交易日的未复权日线，用于补齐缓存中缺少的单日。"""
    compact_date = str(trade_date).replace("-", "")
    with _market_daily_cache_lock:
        cached = _market_daily_cache.get(compact_date)
        if cached is not None:
            return cached.copy()

    pro = _get_tushare_pro()
    if pro is None:
        raise RuntimeError("Tushare 未配置，无法执行全市场单日增量更新")
    try:
        _tushare_wait_token()
        result = pro.daily(
            trade_date=compact_date,
            fields=(
                "ts_code,trade_date,open,high,low,close,pre_close,"
                "pct_chg,vol,amount"
            ),
        )
    except Exception as error:
        raise RuntimeError(
            f"Tushare 全市场单日行情获取失败：{type(error).__name__}"
        ) from error
    if result is None or result.empty:
        raise RuntimeError(f"{trade_date} 未取得全市场单日行情")

    frame = result.rename(columns={
        "ts_code": "code",
        "trade_date": "date",
        "pct_chg": "pct_change",
        "vol": "volume",
    }).copy()
    frame["code"] = frame["code"].astype(str).str.split(".").str[0].str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") * 100
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce") * 1000
    for column in (
        "open", "high", "low", "close", "pre_close", "pct_change",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop_duplicates("code").reset_index(drop=True)
    with _market_daily_cache_lock:
        _market_daily_cache[compact_date] = frame.copy()
    return frame


def get_stock_missing_daily(
    code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """通过 Tushare 只获取单只股票缺失的日期区间，不重抓已有历史。"""
    pro = _get_tushare_pro()
    if pro is None:
        raise RuntimeError("Tushare 未配置，无法补齐多日缺口")
    try:
        _tushare_wait_token()
        result = pro.daily(
            ts_code=_code_to_tushare(code),
            start_date=str(start_date).replace("-", ""),
            end_date=str(end_date).replace("-", ""),
            fields=(
                "ts_code,trade_date,open,high,low,close,pre_close,"
                "pct_chg,vol,amount"
            ),
        )
    except Exception as error:
        raise RuntimeError(
            f"{code} 缺失日线获取失败：{type(error).__name__}"
        ) from error
    if result is None or result.empty:
        raise RuntimeError(f"{code} 在缺失区间未取得日线")
    frame = result.rename(columns={
        "ts_code": "code",
        "trade_date": "date",
        "pct_chg": "pct_change",
        "vol": "volume",
    }).copy()
    frame["code"] = code
    frame["date"] = pd.to_datetime(frame["date"])
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") * 100
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce") * 1000
    for column in (
        "open", "high", "low", "close", "pre_close", "pct_change",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def get_trading_codes(trade_date: str) -> set[str]:
    """返回指定交易日有实际成交的股票代码，用于排除停牌股后校验数据完整性。"""
    try:
        result = get_market_daily_snapshot(trade_date)
        volume = pd.to_numeric(result["volume"], errors="coerce").fillna(0)
        amount = pd.to_numeric(result["amount"], errors="coerce").fillna(0)
        active = result[(volume > 0) | (amount > 0)]
        return set(active["code"].astype(str))
    except Exception as error:
        print(f"  [tushare] 当日交易状态获取失败，切换 AkShare: {type(error).__name__}")

    try:
        time.sleep(_random_delay(0.5, 1.2))
        result = ak.stock_zh_a_spot_em()
        code_column = next(
            (column for column in ("代码", "code", "symbol") if column in result.columns),
            None,
        )
        if code_column is None or result.empty:
            raise ValueError("实时行情缺少股票代码")
        volume = pd.to_numeric(
            result["成交量"] if "成交量" in result.columns else 0,
            errors="coerce",
        )
        amount = pd.to_numeric(
            result["成交额"] if "成交额" in result.columns else 0,
            errors="coerce",
        )
        active = result[(volume.fillna(0) > 0) | (amount.fillna(0) > 0)]
        codes = {
            str(value).split(".")[0].zfill(6)
            for value in active[code_column].astype(str)
        }
        if codes:
            return codes
    except Exception as error:
        raise RuntimeError(
            f"无法验证 {trade_date} 的股票交易状态：{type(error).__name__}"
        ) from error
    raise RuntimeError(f"{trade_date} 未取得任何有成交股票，无法验证行情完整性")
