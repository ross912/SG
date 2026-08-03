"""面向未来的公开证据缓存：业绩预告与机构盈利预测。

行业景气和市场份额变化由评分层使用最新财报横截面计算；本模块只保存
可批量验证的公司级前瞻证据，不使用大模型猜测研发、订单或市场空间。
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta
from typing import Any

import akshare as ak
import numpy as np
import pandas as pd

from config import DATA_DIR


FORWARD_PATH = DATA_DIR / "fundamentals" / "forward_outlook_latest.csv"
FORWARD_META_PATH = DATA_DIR / "fundamentals" / "forward_outlook_meta.json"

_POSITIVE_RD = (
    "加大研发", "研发投入增加", "研发项目落地", "研发成果",
)
_NEGATIVE_RD = ("研发费用下降", "研发投入减少", "削减研发")
_GENERIC_RD = ("研发",)
_POSITIVE_ORDER = (
    "在手订单", "订单增长", "订单充足", "新增订单", "产能释放",
    "扩产项目投产", "产销量增长", "产量增长", "市场份额提升",
)
_NEGATIVE_ORDER = (
    "订单减少", "订单下降", "需求不足", "产能利用不足", "项目延期",
    "市场份额下降",
)
_GENERIC_ORDER = ("订单", "产能", "扩产", "产量", "产销", "市场份额")


def load_forward_cache() -> tuple[pd.DataFrame, dict[str, Any]]:
    if FORWARD_PATH.exists():
        try:
            frame = pd.read_csv(FORWARD_PATH, dtype={"code": str})
            frame["code"] = frame["code"].astype(str).str.zfill(6)
        except Exception:
            frame = pd.DataFrame()
    else:
        frame = pd.DataFrame()
    if FORWARD_META_PATH.exists():
        try:
            metadata = json.loads(FORWARD_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    else:
        metadata = {}
    return frame, metadata


def refresh_forward_cache(
    *,
    allow_network: bool = True,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """刷新前瞻证据；失败时保留最近一次成功缓存。"""
    cached, cached_meta = load_forward_cache()
    if not force and not cached.empty and not _expired(cached_meta):
        return cached, cached_meta
    if not allow_network:
        return cached, cached_meta

    warnings: list[str] = []
    official = pd.DataFrame()
    official_period = ""
    for period in _forecast_period_candidates(date.today()):
        try:
            candidate = _normalize_performance_forecast(
                ak.stock_yjyg_em(date=period),
            )
        except Exception as error:
            warnings.append(
                f"业绩预告 {period} 获取失败：{type(error).__name__}",
            )
            continue
        if len(candidate) > len(official):
            official = candidate
            official_period = period
        if candidate["code"].nunique() >= 500:
            break

    try:
        analyst = _normalize_analyst_forecast(
            ak.stock_profit_forecast_em(symbol=""),
            current_year=date.today().year,
        )
    except Exception as error:
        analyst = pd.DataFrame()
        warnings.append(f"机构盈利预测获取失败：{type(error).__name__}")

    official_columns = [
        "code", "forecast_metric", "official_profit_forecast",
        "official_profit_yoy", "forecast_reason", "forecast_type",
        "forecast_notice_date", "rd_evidence_score", "order_evidence_score",
        "rd_evidence", "order_evidence",
    ]
    analyst_columns = [
        "code", "research_report_count", "analyst_eps_base_year",
        "analyst_eps_next_year", "analyst_eps_base", "analyst_eps_next",
        "analyst_eps_growth", "positive_rating_ratio",
    ]
    if official.empty and not cached.empty:
        available = [column for column in official_columns if column in cached]
        if "official_profit_forecast" in available or "official_profit_yoy" in available:
            official = cached[available].copy()
            evidence = pd.Series(False, index=official.index)
            for column in ("official_profit_forecast", "official_profit_yoy"):
                if column in official:
                    evidence |= official[column].notna()
            official = official.loc[evidence]
            warnings.append("业绩预告沿用最近成功缓存")
    if analyst.empty and not cached.empty:
        available = [column for column in analyst_columns if column in cached]
        if "research_report_count" in available:
            analyst = cached[available].copy()
            analyst = analyst.loc[
                pd.to_numeric(
                    analyst["research_report_count"], errors="coerce",
                ).fillna(0).gt(0)
            ]
            warnings.append("机构盈利预测沿用最近成功缓存")

    if official.empty and analyst.empty:
        if not cached.empty:
            cached_meta = dict(cached_meta)
            cached_meta["refresh_warning"] = "；".join(warnings) or "前瞻数据刷新失败"
            return cached, cached_meta
        return pd.DataFrame(), {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "row_count": 0,
            "warnings": warnings or ["没有取得前瞻数据"],
        }

    if official.empty:
        combined = analyst
    elif analyst.empty:
        combined = official
    else:
        combined = official.merge(analyst, on="code", how="outer")
    combined["code"] = combined["code"].astype(str).str.zfill(6)
    combined = combined.drop_duplicates("code").sort_values("code")

    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "row_count": int(len(combined)),
        "official_count": int(len(official)),
        "analyst_count": int(len(analyst)),
        "official_period": official_period,
        "source": "东方财富业绩预告与机构盈利预测（AkShare）",
        "warnings": warnings,
        "limitations": (
            "研发和订单仅识别公司公告原因中的明确文字证据；未提及不等于没有。"
            "机构预测存在覆盖偏差，不能视为公司承诺。"
        ),
    }
    FORWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(FORWARD_PATH, index=False, encoding="utf-8-sig")
    FORWARD_META_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return combined, metadata


def _normalize_performance_forecast(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.rename(columns={
        "股票代码": "code",
        "预测指标": "forecast_metric",
        "预测数值": "official_profit_forecast",
        "业绩变动幅度": "official_profit_yoy",
        "业绩变动原因": "forecast_reason",
        "预告类型": "forecast_type",
        "公告日期": "forecast_notice_date",
    }).copy()
    required = [
        "code", "forecast_metric", "official_profit_forecast",
        "official_profit_yoy", "forecast_reason", "forecast_type",
        "forecast_notice_date",
    ]
    for column in required:
        if column not in data:
            data[column] = np.nan
    data["code"] = data["code"].astype(str).str.extract(r"(\d{6})", expand=False)
    data = data.dropna(subset=["code"])
    metric = data["forecast_metric"].fillna("").astype(str)
    data["_priority"] = np.select(
        [
            metric.str.contains("归属于上市公司股东的净利润")
            & ~metric.str.contains("扣除"),
            metric.str.contains("净利润") & ~metric.str.contains("扣除"),
            metric.str.contains("扣除.*净利润", regex=True),
        ],
        [0, 1, 2],
        default=3,
    )
    data["forecast_notice_date"] = pd.to_datetime(
        data["forecast_notice_date"], errors="coerce",
    )
    data = data.sort_values(
        ["code", "_priority", "forecast_notice_date"],
        ascending=[True, True, False],
    ).drop_duplicates("code")
    data["official_profit_forecast"] = pd.to_numeric(
        data["official_profit_forecast"], errors="coerce",
    )
    data["official_profit_yoy"] = pd.to_numeric(
        data["official_profit_yoy"], errors="coerce",
    )
    data["forecast_reason"] = data["forecast_reason"].fillna("").astype(str)
    data["rd_evidence_score"] = data["forecast_reason"].map(
        lambda text: _text_signal(text, _POSITIVE_RD, _NEGATIVE_RD, _GENERIC_RD),
    )
    data["order_evidence_score"] = data["forecast_reason"].map(
        lambda text: _text_signal(
            text, _POSITIVE_ORDER, _NEGATIVE_ORDER, _GENERIC_ORDER,
        ),
    )
    data["rd_evidence"] = data["forecast_reason"].map(
        lambda text: _evidence_excerpt(text, _GENERIC_RD),
    )
    data["order_evidence"] = data["forecast_reason"].map(
        lambda text: _evidence_excerpt(text, _GENERIC_ORDER),
    )
    data["forecast_notice_date"] = data["forecast_notice_date"].dt.strftime("%Y-%m-%d")
    return data[required + [
        "rd_evidence_score", "order_evidence_score",
        "rd_evidence", "order_evidence",
    ]]


def _normalize_analyst_forecast(
    frame: pd.DataFrame,
    *,
    current_year: int,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.rename(columns={
        "代码": "code",
        "研报数": "research_report_count",
        "机构投资评级(近六个月)-买入": "rating_buy",
        "机构投资评级(近六个月)-增持": "rating_add",
        "机构投资评级(近六个月)-中性": "rating_neutral",
        "机构投资评级(近六个月)-减持": "rating_reduce",
        "机构投资评级(近六个月)-卖出": "rating_sell",
    }).copy()
    year_columns: dict[int, str] = {}
    for column in data.columns:
        match = re.fullmatch(r"(\d{4})预测每股收益", str(column))
        if match:
            year_columns[int(match.group(1))] = str(column)
    usable_years = sorted(year for year in year_columns if year >= current_year)
    base_year = usable_years[0] if usable_years else None
    next_year = usable_years[1] if len(usable_years) > 1 else None
    data["analyst_eps_base_year"] = base_year
    data["analyst_eps_next_year"] = next_year
    data["analyst_eps_base"] = (
        pd.to_numeric(data[year_columns[base_year]], errors="coerce")
        if base_year is not None else np.nan
    )
    data["analyst_eps_next"] = (
        pd.to_numeric(data[year_columns[next_year]], errors="coerce")
        if next_year is not None else np.nan
    )
    data["analyst_eps_growth"] = np.where(
        data["analyst_eps_base"] > 0,
        (data["analyst_eps_next"] / data["analyst_eps_base"] - 1) * 100,
        np.nan,
    )
    rating_columns = [
        "rating_buy", "rating_add", "rating_neutral", "rating_reduce", "rating_sell",
    ]
    for column in ["research_report_count", *rating_columns]:
        if column not in data:
            data[column] = 0
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    rating_total = data[rating_columns].sum(axis=1)
    data["positive_rating_ratio"] = np.where(
        rating_total > 0,
        (data["rating_buy"] + data["rating_add"]) / rating_total,
        np.nan,
    )
    data["code"] = data["code"].astype(str).str.extract(r"(\d{6})", expand=False)
    data = data.dropna(subset=["code"]).drop_duplicates("code")
    return data[[
        "code", "research_report_count", "analyst_eps_base_year",
        "analyst_eps_next_year", "analyst_eps_base", "analyst_eps_next",
        "analyst_eps_growth", "positive_rating_ratio",
    ]]


def _text_signal(
    text: str,
    positive: tuple[str, ...],
    negative: tuple[str, ...],
    generic: tuple[str, ...],
) -> float:
    normalized = str(text or "")
    if any(term in normalized for term in negative):
        return 0.2
    if any(term in normalized for term in positive):
        return 1.0
    if any(term in normalized for term in generic):
        return 0.6
    return np.nan


def _evidence_excerpt(text: str, terms: tuple[str, ...]) -> str:
    normalized = re.sub(r"\s+", "", str(text or ""))
    positions = [normalized.find(term) for term in terms if term in normalized]
    if not positions:
        return ""
    start = max(0, min(positions) - 24)
    return normalized[start:start + 90]


def _forecast_period_candidates(today: date) -> list[str]:
    year = today.year
    if today.month <= 4:
        periods = [f"{year - 1}1231", f"{year}0331"]
    elif today.month <= 8:
        periods = [f"{year}0630", f"{year}0331"]
    elif today.month <= 10:
        periods = [f"{year}0930", f"{year}0630"]
    else:
        periods = [f"{year}1231", f"{year}0930"]
    return periods


def _expired(metadata: dict[str, Any]) -> bool:
    try:
        generated = datetime.fromisoformat(str(metadata.get("generated_at", "")))
        if generated.tzinfo is None:
            generated = generated.astimezone()
    except (TypeError, ValueError):
        return True
    days = max(1, int(os.getenv("SG_QUANT_FORWARD_CACHE_DAYS", "1")))
    return datetime.now().astimezone() - generated >= timedelta(days=days)
