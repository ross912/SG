"""基本面与估值数据缓存。

财务报表来自 AkShare 的东方财富全市场报表接口；估值来自 Tushare
``daily_basic``。财务报表按低频缓存，交易日只刷新估值。
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd
import requests

from config import DATA_DIR
from data.fetcher import _get_tushare_pro, _tushare_wait_token


FUNDAMENTAL_DIR = DATA_DIR / "fundamentals"
FUNDAMENTAL_PATH = FUNDAMENTAL_DIR / "fundamentals_latest.csv"
FUNDAMENTAL_META_PATH = FUNDAMENTAL_DIR / "fundamentals_meta.json"
VALUATION_PATH = FUNDAMENTAL_DIR / "valuation_latest.csv"
VALUATION_META_PATH = FUNDAMENTAL_DIR / "valuation_meta.json"


def load_fundamental_cache() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _read_csv(FUNDAMENTAL_PATH)
    return frame, _read_json(FUNDAMENTAL_META_PATH)


def load_valuation_cache() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _read_csv(VALUATION_PATH)
    return frame, _read_json(VALUATION_META_PATH)


def refresh_fundamental_cache(
    *,
    universe_count: int,
    allow_network: bool = True,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """低频刷新全市场财务报表与股权质押数据。

    缓存默认 7 天有效。当前季度披露不完整时，自动退回覆盖率达到 75%
    的最近报告期，避免把“尚未披露”误判为“基本面缺失”。
    """
    cached, metadata = load_fundamental_cache()
    if (
        not force
        and not cached.empty
        and not _cache_expired(metadata, _fundamental_ttl_days())
    ):
        return cached, metadata
    if not allow_network:
        return cached, metadata

    minimum_coverage = max(500, int(max(1, universe_count) * 0.75))
    chosen_income: pd.DataFrame | None = None
    chosen_period = ""
    coverage_trials: list[dict[str, Any]] = []
    best_income: pd.DataFrame | None = None
    best_period = ""

    for period in _candidate_report_periods(date.today()):
        try:
            income = _normalize_income(ak.stock_lrb_em(date=period), period)
        except Exception as error:
            coverage_trials.append({
                "period": period, "rows": 0,
                "error": f"{type(error).__name__}: {error}"[:240],
            })
            continue
        row_count = int(income["code"].nunique()) if not income.empty else 0
        coverage_trials.append({"period": period, "rows": row_count, "error": ""})
        if best_income is None or row_count > len(best_income):
            best_income, best_period = income, period
        if row_count >= minimum_coverage:
            chosen_income, chosen_period = income, period
            break

    if chosen_income is None:
        chosen_income, chosen_period = best_income, best_period
    if chosen_income is None or chosen_income.empty:
        if not cached.empty:
            metadata = dict(metadata)
            metadata["refresh_warning"] = "财务报表刷新失败，继续使用最近一次缓存"
            return cached, metadata
        raise RuntimeError("没有取得可用的全市场财务报表")

    balance = _fetch_balance_sheet(chosen_period)
    cashflow = _normalize_cashflow(
        ak.stock_xjll_em(date=chosen_period), chosen_period,
    )
    balance = balance.drop(columns=["report_period"], errors="ignore")
    cashflow = cashflow.drop(columns=["report_period"], errors="ignore")
    frame = chosen_income.merge(balance, on="code", how="outer")
    frame = frame.merge(cashflow, on="code", how="outer")

    pledge_date = _previous_friday(date.today()).strftime("%Y%m%d")
    pledge_warning = ""
    try:
        pledge = _normalize_pledge(
            ak.stock_gpzy_pledge_ratio_em(date=pledge_date),
        )
    except Exception as error:
        pledge = pd.DataFrame(columns=["code", "pledge_ratio", "pledge_date"])
        pledge_warning = f"股权质押数据不可用：{type(error).__name__}"
    frame = frame.merge(pledge, on="code", how="left")
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame = frame.drop_duplicates("code").sort_values("code").reset_index(drop=True)

    now_iso = _now_iso()
    metadata = {
        "schema_version": 1,
        "generated_at": now_iso,
        "report_period": chosen_period,
        "row_count": int(len(frame)),
        "minimum_coverage": minimum_coverage,
        "coverage_trials": coverage_trials,
        "pledge_date": pledge_date,
        "pledge_warning": pledge_warning,
        "source": "AkShare/东方财富公开财务报表与质押统计",
        "limitations": (
            "当前免费数据源不覆盖财务审计意见、监管调查和未公开突发事件；"
            "风险等级不能等同于无暴雷风险。"
        ),
    }
    _write_csv(FUNDAMENTAL_PATH, frame)
    _write_json(FUNDAMENTAL_META_PATH, metadata)
    return frame, metadata


def refresh_valuation_cache(
    trade_date: str,
    *,
    allow_network: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """刷新指定交易日的全市场估值；失败时退回最近缓存并明确标记日期。"""
    compact_date = str(trade_date).replace("-", "")
    cached, metadata = load_valuation_cache()
    if metadata.get("trade_date") == compact_date and not cached.empty:
        return cached, metadata
    if not allow_network:
        return cached, metadata

    pro = _get_tushare_pro()
    tushare_error = ""
    if pro is None:
        raw = pd.DataFrame()
        tushare_error = "TushareNotConfigured"
    else:
        try:
            _tushare_wait_token()
            raw = pro.daily_basic(
                trade_date=compact_date,
                fields=(
                    "ts_code,trade_date,close,turnover_rate,pe_ttm,pb,ps_ttm,"
                    "dv_ttm,total_mv,circ_mv"
                ),
            )
        except Exception as error:
            raw = pd.DataFrame()
            tushare_error = f"{type(error).__name__}: {error}"[:240]

    if raw is not None and not raw.empty:
        frame = raw.rename(columns={"ts_code": "code"}).copy()
        frame["code"] = frame["code"].map(_normalize_code)
        frame["pe_basis"] = "PE(TTM)"
        source = "Tushare daily_basic"
        warning = ""
    else:
        try:
            frame = _normalize_spot_valuation(
                ak.stock_zh_a_spot_em(), compact_date,
            )
            source = "AkShare/东方财富全市场快照"
            warning = (
                "Tushare daily_basic 不可用，估值降级为动态 PE + PB；"
                "PS 与股息率本次不参与评分"
            )
            if tushare_error:
                warning += f"（{tushare_error.split(':', 1)[0]}）"
        except Exception:
            try:
                frame = _fetch_sina_valuation(compact_date)
                source = "新浪财经全市场行情"
                warning = (
                    "Tushare daily_basic 与东方财富快照不可用，"
                    "估值降级为新浪动态 PE + PB；PS 与股息率本次不参与评分"
                )
            except Exception as sina_error:
                if not cached.empty:
                    metadata = dict(metadata)
                    metadata["refresh_warning"] = (
                        f"{compact_date} 估值刷新失败，继续使用最近缓存："
                        f"{type(sina_error).__name__}"
                    )
                    return cached, metadata
                raise RuntimeError(
                    f"{compact_date} 没有可用估值数据"
                ) from sina_error

    numeric_columns = [
        "close", "turnover_rate", "pe_ttm", "pb", "ps_ttm",
        "dv_ttm", "total_mv", "circ_mv",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop_duplicates("code").sort_values("code").reset_index(drop=True)
    metadata = {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "trade_date": compact_date,
        "row_count": int(len(frame)),
        "source": source,
        "refresh_warning": warning,
    }
    _write_csv(VALUATION_PATH, frame)
    _write_json(VALUATION_META_PATH, metadata)
    return frame, metadata


def _fetch_balance_sheet(period: str) -> pd.DataFrame:
    main = _normalize_balance(ak.stock_zcfz_em(date=period), period)
    try:
        bj = _normalize_balance(ak.stock_zcfz_bj_em(date=period), period)
    except Exception:
        bj = pd.DataFrame()
    if bj.empty:
        return main
    return (
        pd.concat([main, bj], ignore_index=True)
        .drop_duplicates("code")
        .reset_index(drop=True)
    )


def _normalize_income(raw: pd.DataFrame, period: str) -> pd.DataFrame:
    mapping = {
        "股票代码": "code",
        "股票简称": "statement_name",
        "净利润": "net_profit",
        "净利润同比": "net_profit_yoy",
        "营业总收入": "revenue",
        "营业总收入同比": "revenue_yoy",
        "营业总支出-营业支出": "operating_cost",
        "营业利润": "operating_profit",
        "公告日期": "income_ann_date",
    }
    return _normalize_report(raw, mapping, period)


def _normalize_balance(raw: pd.DataFrame, period: str) -> pd.DataFrame:
    mapping = {
        "股票代码": "code",
        "资产-货币资金": "cash",
        "资产-应收账款": "accounts_receivable",
        "资产-存货": "inventory",
        "资产-总资产": "total_assets",
        "资产-总资产同比": "total_assets_yoy",
        "负债-总负债": "total_liabilities",
        "资产负债率": "debt_to_assets",
        "股东权益合计": "equity",
        "公告日期": "balance_ann_date",
    }
    return _normalize_report(raw, mapping, period)


def _normalize_cashflow(raw: pd.DataFrame, period: str) -> pd.DataFrame:
    mapping = {
        "股票代码": "code",
        "净现金流-净现金流": "net_cashflow",
        "净现金流-同比增长": "net_cashflow_yoy",
        "经营性现金流-现金流量净额": "operating_cashflow",
        "公告日期": "cashflow_ann_date",
    }
    return _normalize_report(raw, mapping, period)


def _normalize_report(
    raw: pd.DataFrame,
    mapping: dict[str, str],
    period: str,
) -> pd.DataFrame:
    if raw is None or raw.empty or "股票代码" not in raw.columns:
        return pd.DataFrame(columns=["code"])
    keep = [column for column in mapping if column in raw.columns]
    frame = raw[keep].rename(columns=mapping).copy()
    frame["code"] = frame["code"].map(_normalize_code)
    for column in frame.columns:
        if column == "code" or column.endswith("_date") or column == "statement_name":
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in [c for c in frame if c.endswith("_date")]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["report_period"] = period
    return frame.drop_duplicates("code").reset_index(drop=True)


def _normalize_pledge(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty or "股票代码" not in raw.columns:
        return pd.DataFrame(columns=["code", "pledge_ratio", "pledge_date"])
    frame = raw.rename(columns={
        "股票代码": "code",
        "质押比例": "pledge_ratio",
        "交易日期": "pledge_date",
    })[["code", "pledge_ratio", "pledge_date"]].copy()
    frame["code"] = frame["code"].map(_normalize_code)
    frame["pledge_ratio"] = pd.to_numeric(frame["pledge_ratio"], errors="coerce")
    frame["pledge_date"] = pd.to_datetime(
        frame["pledge_date"], errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    return frame.drop_duplicates("code").reset_index(drop=True)


def _normalize_spot_valuation(raw: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if raw is None or raw.empty or "代码" not in raw.columns:
        raise ValueError("东方财富全市场快照缺少代码")
    frame = raw.rename(columns={
        "代码": "code",
        "最新价": "close",
        "换手率": "turnover_rate",
        "市盈率-动态": "pe_ttm",
        "市净率": "pb",
        "总市值": "total_mv",
        "流通市值": "circ_mv",
    }).copy()
    for column in (
        "close", "turnover_rate", "pe_ttm", "pb", "total_mv", "circ_mv",
    ):
        if column not in frame:
            frame[column] = pd.NA
    frame["code"] = frame["code"].map(_normalize_code)
    frame["trade_date"] = trade_date
    frame["ps_ttm"] = pd.NA
    frame["dv_ttm"] = pd.NA
    frame["pe_basis"] = "动态PE"
    # 东方财富市值单位为元，统一为 Tushare 的万元。
    frame["total_mv"] = pd.to_numeric(frame["total_mv"], errors="coerce") / 10_000
    frame["circ_mv"] = pd.to_numeric(frame["circ_mv"], errors="coerce") / 10_000
    return frame[
        [
            "code", "trade_date", "close", "turnover_rate", "pe_ttm", "pb",
            "ps_ttm", "dv_ttm", "total_mv", "circ_mv", "pe_basis",
        ]
    ].drop_duplicates("code").reset_index(drop=True)


def _fetch_sina_valuation(trade_date: str) -> pd.DataFrame:
    """新浪全市场行情降级源，保留 AkShare 默认函数丢弃的 PE/PB 字段。"""
    base = (
        "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    )
    count_response = requests.get(
        base + "Market_Center.getHQNodeStockCount",
        params={"node": "hs_a"},
        timeout=15,
    )
    count_response.raise_for_status()
    digits = "".join(character for character in count_response.text if character.isdigit())
    total = int(digits or "0")
    if total <= 0:
        raise ValueError("新浪全市场股票数量为空")

    rows: list[dict[str, Any]] = []
    page_size = 100
    for page in range(1, math.ceil(total / page_size) + 1):
        params = {
            "page": str(page), "num": str(page_size), "sort": "symbol",
            "asc": "1", "node": "hs_a", "symbol": "", "_s_r_a": "page",
        }
        page_error: Exception | None = None
        for attempt in range(2):
            try:
                response = requests.get(
                    base + "Market_Center.getHQNodeData",
                    params=params,
                    timeout=15,
                )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    rows.extend(payload)
                    page_error = None
                    break
                raise ValueError("新浪行情页返回格式异常")
            except Exception as error:
                page_error = error
                time.sleep(0.25 * (attempt + 1))
        if page_error is not None:
            raise page_error
        time.sleep(0.03)

    raw = pd.DataFrame(rows)
    required = {"code", "trade", "per", "pb"}
    if raw.empty or not required.issubset(raw.columns):
        raise ValueError("新浪行情缺少估值字段")
    frame = raw.rename(columns={
        "trade": "close",
        "turnoverratio": "turnover_rate",
        "per": "pe_ttm",
        "mktcap": "total_mv",
        "nmc": "circ_mv",
    }).copy()
    frame["code"] = frame["code"].map(_normalize_code)
    frame["trade_date"] = trade_date
    frame["ps_ttm"] = pd.NA
    frame["dv_ttm"] = pd.NA
    frame["pe_basis"] = "动态PE"
    for column in (
        "close", "turnover_rate", "pe_ttm", "pb", "total_mv", "circ_mv",
    ):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame[
        [
            "code", "trade_date", "close", "turnover_rate", "pe_ttm", "pb",
            "ps_ttm", "dv_ttm", "total_mv", "circ_mv", "pe_basis",
        ]
    ].drop_duplicates("code").reset_index(drop=True)


def _candidate_report_periods(today: date) -> list[str]:
    periods: list[date] = []
    for year in range(today.year, today.year - 3, -1):
        periods.extend([
            date(year, 12, 31),
            date(year, 9, 30),
            date(year, 6, 30),
            date(year, 3, 31),
        ])
    # 报告期之后至少留 20 天，避免在季度刚结束时请求必然不完整的数据。
    eligible = [
        value for value in periods
        if value + timedelta(days=20) <= today
    ]
    return [value.strftime("%Y%m%d") for value in sorted(eligible, reverse=True)[:6]]


def _previous_friday(today: date) -> date:
    offset = (today.weekday() - 4) % 7
    return today - timedelta(days=offset)


def _fundamental_ttl_days() -> int:
    try:
        return max(1, int(os.getenv("SG_QUANT_FUNDAMENTAL_CACHE_DAYS", "7")))
    except ValueError:
        return 7


def _cache_expired(metadata: dict[str, Any], ttl_days: int) -> bool:
    try:
        generated = datetime.fromisoformat(str(metadata.get("generated_at", "")))
    except ValueError:
        return True
    if generated.tzinfo is not None:
        generated = generated.astimezone().replace(tzinfo=None)
    return datetime.now() - generated > timedelta(days=ttl_days)


def _normalize_code(value: Any) -> str:
    text = str(value or "").strip().split(".")[0]
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, dtype={"code": str, "report_period": str})
        if "report_period" in frame:
            frame["report_period"] = frame["report_period"].str.replace(
                r"\.0$", "", regex=True,
            )
        return frame
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
