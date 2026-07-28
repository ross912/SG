"""
市场环境评估：基于大盘指数判断牛/熊/震荡，仅用于因子权重与仓位参考。

核心逻辑（道氏理论 + 均线系统）：
- 牛市 (bull)   : 指数收盘 > MA20 > MA60，且 MA20 向上 → 全力选股
- 熊市 (bear)   : 指数收盘 < MA60 → 使用熊市权重
- 震荡 (sideways): 介于两者之间 → 正常选股但降低趋势权重
"""
import json
from datetime import datetime
from typing import Any

import pandas as pd
from dataclasses import asdict, dataclass

from config import DATA_DIR
from data.storage import expected_market_date


# 指数代码映射
INDEX_CODES = {
    "sh000001": "上证综指",
    "sz399001": "深证成指",
    "sh000300": "沪深300",
    "sh000016": "上证50",
    "sh000905": "中证500",
    "sz399006": "创业板指",
    "sh000688": "科创50",
}
DISPLAY_INDICES = (
    ("shanghai", "000001", "上证指数"),
    ("shenzhen", "399001", "深证成指"),
    ("chinext", "399006", "创业板指"),
    ("star50", "000688", "科创50"),
)
_REGIME_CACHE = DATA_DIR / "market_regime.json"
_OVERVIEW_CACHE = DATA_DIR / "market_overview.json"
_REGIME_LABELS = {"bull": "牛市", "sideways": "震荡", "bear": "熊市", "unknown": "待更新"}


@dataclass
class MarketRegime:
    """市场状态评估结果。"""
    regime: str           # bull / sideways / bear
    score: float          # 综合评分 0-100
    index_name: str       # 主要参考指数
    index_close: float    # 指数最新价
    ma20: float           # 20日均线
    ma60: float           # 60日均线
    ma20_trend: str       # MA20趋势方向 up/down/flat
    details: str          # 一句话描述


def load_cached_market_regime() -> MarketRegime | None:
    """读取最近一次 online 保存的市场状态；离线模式不触发网络。"""
    if not _REGIME_CACHE.exists():
        return None
    try:
        with _REGIME_CACHE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return MarketRegime(**payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_market_regime(regime: MarketRegime) -> None:
    """保存 online 市场状态，供离线复算使用。"""
    try:
        with _REGIME_CACHE.open("w", encoding="utf-8") as handle:
            json.dump(asdict(regime), handle, ensure_ascii=False, indent=2)
    except OSError:
        pass


def load_cached_market_overview() -> dict[str, Any]:
    """读取四指数最新状态；页面只读缓存，不临时请求行情源。"""
    try:
        with _OVERVIEW_CACHE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def save_market_overview(overview: dict[str, Any]) -> None:
    """保存四指数状态，供 Dashboard、日报和问答共用。"""
    try:
        temporary = _OVERVIEW_CACHE.with_name(f".{_OVERVIEW_CACHE.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(overview, handle, ensure_ascii=False, indent=2, allow_nan=False)
        temporary.replace(_OVERVIEW_CACHE)
    except OSError:
        pass


def default_market_regime() -> MarketRegime:
    """没有 online 缓存时使用中性状态，不伪造在线行情。"""
    return MarketRegime(
        regime="sideways",
        score=50.0,
        index_name="离线默认",
        index_close=0.0,
        ma20=0.0,
        ma60=0.0,
        ma20_trend="flat",
        details="未找到已保存的市场状态，离线模式使用中性权重",
    )


def _code_to_index_symbol(code: str) -> str:
    """将指数代码转为 stock_zh_index_daily 所需的 sh/sz 前缀格式。

    指数前缀规则：000xxx（上证系列）→ sh，399xxx（深证系列）→ sz。
    """
    if code.startswith("sh") or code.startswith("sz"):
        return code
    # 上证系列指数: 000xxx
    if code.startswith("000"):
        return f"sh{code}"
    # 深证/创业板系列指数: 399xxx
    return f"sz{code}"


def fetch_index_kline(code: str = "000001", days: int = 250) -> pd.DataFrame | None:
    """获取指数日K线数据（使用新浪财经指数接口）。"""
    import akshare as ak
    try:
        symbol = _code_to_index_symbol(code)
        df = ak.stock_zh_index_daily(symbol=symbol)
        if df is None or df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df.tail(days)
    except Exception:
        return None


def fetch_index_spot_snapshot() -> pd.DataFrame:
    """一次取得沪深指数收盘快照，用于历史日线尚未落库时补齐当天。"""
    import akshare as ak

    frame = ak.stock_zh_index_spot_sina()
    required = {"代码", "最新价", "昨收", "今开", "最高", "最低"}
    if frame is None or frame.empty or not required.issubset(frame.columns):
        raise RuntimeError("指数实时快照字段不完整")
    output = frame.rename(columns={
        "代码": "symbol",
        "最新价": "close",
        "昨收": "pre_close",
        "今开": "open",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }).copy()
    output["symbol"] = output["symbol"].astype(str)
    for column in ("close", "pre_close", "open", "high", "low", "volume", "amount"):
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _append_index_spot_bar(
    frame: pd.DataFrame,
    spot_row: pd.Series,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """仅在历史源缺少目标日时追加一根真实指数收盘 K 线。"""
    output = frame.copy()
    latest = pd.to_datetime(output["date"]).max().normalize()
    if latest >= target_date:
        return output
    values = {
        "date": target_date,
        "open": spot_row.get("open"),
        "high": spot_row.get("high"),
        "low": spot_row.get("low"),
        "close": spot_row.get("close"),
        "volume": spot_row.get("volume", 0),
        "amount": spot_row.get("amount", 0),
    }
    if any(pd.isna(values[column]) for column in ("open", "high", "low", "close")):
        return output
    return (
        pd.concat([output, pd.DataFrame([values])], ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _ma_trend(series: pd.Series, lookback: int = 5) -> str:
    """判断均线方向。"""
    if len(series) < lookback + 1:
        return "flat"
    recent = series.iloc[-lookback:]
    if recent.iloc[-1] > recent.iloc[0] * 1.005:
        return "up"
    elif recent.iloc[-1] < recent.iloc[0] * 0.995:
        return "down"
    return "flat"


def _assess_single_index(df: pd.DataFrame, index_name: str) -> dict:
    """对单个指数做牛熊判断，返回 regime/score/details。"""
    close = df["close"]
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    latest_close = close.iloc[-1]
    latest_ma20 = ma20.iloc[-1]
    latest_ma60 = ma60.iloc[-1]
    trend = _ma_trend(ma20)

    above_ma20 = latest_close > latest_ma20
    above_ma60 = latest_close > latest_ma60
    ma20_above_ma60 = latest_ma20 > latest_ma60
    ma20_rising = trend == "up"

    if above_ma20 and above_ma60 and ma20_above_ma60 and ma20_rising:
        regime = "bull"
        score = 85.0
        deviation = (latest_close / latest_ma20 - 1) * 100
        if 3 <= deviation <= 8:
            score += 5
        details = f"{index_name}牛市确认：均线多头排列，MA20上行"
    elif not above_ma60:
        regime = "bear"
        score = 25.0
        if not ma20_above_ma60 and trend == "down":
            score -= 10
        details = f"{index_name}熊市：价格在MA60下方"
    else:
        regime = "sideways"
        score = 55.0
        if above_ma20 and not ma20_above_ma60:
            details = f"{index_name}震荡偏多"
            score = 62.0
        elif not above_ma20 and above_ma60:
            details = f"{index_name}震荡偏弱"
            score = 48.0
        else:
            details = f"{index_name}震荡市：方向不明"

    previous_close = close.iloc[-2] if len(close) >= 2 else latest_close
    change_pct = (latest_close / previous_close - 1) * 100 if previous_close else 0.0
    latest_date = pd.to_datetime(df["date"].iloc[-1]).strftime("%Y-%m-%d")
    return {
        "regime": regime, "score": score, "details": details,
        "close": latest_close, "ma20": latest_ma20, "ma60": latest_ma60, "trend": trend,
        "date": latest_date, "change_pct": change_pct,
    }


def _unavailable_index(key: str, code: str, name: str) -> dict[str, Any]:
    return {
        "key": key, "code": code, "name": name, "regime": "unknown",
        "label": _REGIME_LABELS["unknown"], "score": 50.0, "close": 0.0,
        "change_pct": 0.0, "ma20": 0.0, "ma60": 0.0, "trend": "flat",
        "date": "", "details": f"{name}数据不足",
    }


def _display_index_result(key: str, code: str, name: str, result: dict) -> dict[str, Any]:
    return {
        "key": key,
        "code": code,
        "name": name,
        "regime": result["regime"],
        "label": _REGIME_LABELS[result["regime"]],
        "score": round(float(result["score"]), 1),
        "close": round(float(result["close"]), 2),
        "change_pct": round(float(result["change_pct"]), 2),
        "ma20": round(float(result["ma20"]), 2),
        "ma60": round(float(result["ma60"]), 2),
        "trend": result["trend"],
        "date": result["date"],
        "details": result["details"],
    }


def _score_to_hue(score: float) -> float:
    """用户指定的反向股市配色：低分绿、中性黄、高分红。"""
    bounded = min(max(float(score), 0.0), 100.0)
    if bounded <= 50:
        return round(138 - bounded / 50 * 90, 1)
    return round(48 - (bounded - 50) / 50 * 48, 1)


def assess_market_overview() -> tuple[MarketRegime, dict[str, Any]]:
    """评估四个指数；平台主牛熊状态严格只由上证指数决定。"""
    target_date = expected_market_date(datetime.now()).normalize()
    frames: dict[str, pd.DataFrame | None] = {
        code: fetch_index_kline(code)
        for _key, code, _name in DISPLAY_INDICES
    }
    stale_codes = [
        code for code, frame in frames.items()
        if frame is None
        or frame.empty
        or pd.to_datetime(frame["date"]).max().normalize() < target_date
    ]
    if stale_codes:
        try:
            spot = fetch_index_spot_snapshot()
        except Exception:
            spot = pd.DataFrame()
        for code in stale_codes:
            frame = frames.get(code)
            symbol = _code_to_index_symbol(code)
            rows = spot[spot["symbol"] == symbol] if not spot.empty else pd.DataFrame()
            if frame is not None and not frame.empty and len(rows) == 1:
                frames[code] = _append_index_spot_bar(frame, rows.iloc[0], target_date)

    bad_dates = {}
    for _key, code, name in DISPLAY_INDICES:
        frame = frames.get(code)
        actual = ""
        if frame is not None and not frame.empty:
            actual = pd.to_datetime(frame["date"]).max().strftime("%Y-%m-%d")
        if actual != target_date.strftime("%Y-%m-%d"):
            bad_dates[name] = actual or "缺失"
    if bad_dates:
        detail = "、".join(f"{name}={date}" for name, date in bad_dates.items())
        raise RuntimeError(
            f"四指数日期不一致：要求 {target_date.strftime('%Y-%m-%d')}，{detail}"
        )

    indices = []
    for key, code, name in DISPLAY_INDICES:
        frame = frames[code]
        if frame is None or len(frame) < 120:
            indices.append(_unavailable_index(key, code, name))
            continue
        indices.append(_display_index_result(key, code, name, _assess_single_index(frame, name)))

    shanghai = next(item for item in indices if item["key"] == "shanghai")
    if shanghai["regime"] == "unknown":
        regime = default_market_regime()
        regime.index_name = "上证指数（数据待更新）"
        regime.details = "上证指数数据不足，主状态暂按震荡处理"
    else:
        regime = MarketRegime(
            regime=shanghai["regime"],
            score=shanghai["score"],
            index_name="上证指数",
            index_close=shanghai["close"],
            ma20=shanghai["ma20"],
            ma60=shanghai["ma60"],
            ma20_trend=shanghai["trend"],
            details=shanghai["details"],
        )

    overview = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "market_date": shanghai.get("date", ""),
        "primary_index": "shanghai",
        "regime": regime.regime,
        "label": _REGIME_LABELS[regime.regime],
        "score": round(float(regime.score), 1),
        "color_hue": _score_to_hue(regime.score),
        "color_rule": "绿色偏熊，黄色震荡，红色偏牛",
        "details": regime.details,
        "indices": indices,
    }
    return regime, overview


def assess_market_regime(index_code: str = "000001") -> MarketRegime:
    """
    兼容旧调用入口。主市场状态严格只使用上证指数；
    其他三个指数只用于 Dashboard 的分板展示。
    """
    if index_code != "000001":
        frame = fetch_index_kline(index_code)
        index_name = INDEX_CODES.get(_code_to_index_symbol(index_code), index_code)
        if frame is None or len(frame) < 120:
            return default_market_regime()
        result = _assess_single_index(frame, index_name)
        return MarketRegime(
            regime=result["regime"], score=result["score"], index_name=index_name,
            index_close=round(float(result["close"]), 2),
            ma20=round(float(result["ma20"]), 2), ma60=round(float(result["ma60"]), 2),
            ma20_trend=result["trend"], details=result["details"],
        )
    regime, _overview = assess_market_overview()
    return regime


def print_market_regime(regime: MarketRegime) -> None:
    """控制台打印市场状态。"""
    icon = {"bull": "[牛市]", "sideways": "[震荡]", "bear": "[熊市]"}.get(regime.regime, "")
    print(f"\n{'─'*50}")
    print(f"  {icon} 市场状态: {regime.index_name} {regime.index_close:.0f}")
    print(f"  MA20: {regime.ma20:.0f} | MA60: {regime.ma60:.0f}")
    print(f"  综合评分: {regime.score:.0f}/100 | {regime.details}")
    print(f"{'─'*50}")
